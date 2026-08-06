import argparse
import datetime as dt
import json
import logging
import os
import shutil
import time
import warnings
from copy import deepcopy

import pytz

from common.io_handler import IOHandler, format_time_dependent_paths
from common.logging_handler import set_logging_stream
from common.settings import Settings
from hydro.flood_gridded_return_period import CalculateFloodThresholdLevels
from hydro.flood_hazard_mapping import (
    FloodHazardOverlayMerge,
    classify_admin_hazard,
    convert_hazard_classes,
    write_binary_flood_map,
)
from hydro.flood_impact_assessment import ImpactAssessment


class GriddedForecastInputError(RuntimeError):
    """Raised when expected gridded forecast input files are missing or invalid."""


class SafeFormatDict(dict):
    """Keep unresolved template tags unchanged during string formatting."""

    def __missing__(self, key):
        return "{" + key + "}"


def parse_algorithm_time(alg_time: str) -> dt.datetime:
    """Parse the algorithm time in UTC."""
    return pytz.utc.localize(dt.datetime.strptime(alg_time, "%Y-%m-%d %H:%M"))


def _output_path(output_settings: dict) -> str:
    """Build an output path from the standard folder/file-name structure."""
    return os.path.join(output_settings["folder"], output_settings["file_name"])


def _format_forecast_input_file(forecast_template: str, time_step) -> str:
    """Build a forecast input file path from the workflow template and lead time."""
    template_values = SafeFormatDict({
        "step": time_step,
        "time_step": time_step,
        "leadtime": time_step,
        "lead_time": time_step,
        "leadtime_hour": time_step,
    })

    try:
        forecast_file = forecast_template.format_map(template_values)
    except Exception:
        forecast_file = forecast_template

    # Support legacy placeholder styles occasionally used in file templates.
    for placeholder in [
        "[step]", "[time_step]", "[leadtime]", "[lead_time]", "[leadtime_hour]",
        "%STEP", "%TIME_STEP", "%LEADTIME", "%LEAD_TIME", "%LEADTIME_HOUR",
    ]:
        forecast_file = forecast_file.replace(placeholder, str(time_step))

    return forecast_file


def _check_forecast_input_files(input_cfg: dict) -> None:
    """Validate that all GLOFAS NetCDF inputs expected by the workflow exist."""
    forecast_template = input_cfg["forecast_template"]
    time_steps = input_cfg["time_steps"]

    logging.info(" --> Check GLOFAS NetCDF forecast inputs...")
    logging.info(f" ---> Forecast input template: {forecast_template}")

    missing_files = []
    empty_files = []
    expected_files = []

    for time_step in time_steps:
        forecast_file = _format_forecast_input_file(forecast_template, time_step)
        expected_files.append(forecast_file)

        if not os.path.isfile(forecast_file):
            missing_files.append(forecast_file)
        elif os.path.getsize(forecast_file) == 0:
            empty_files.append(forecast_file)

    if missing_files or empty_files:
        n_expected = len(expected_files)
        n_missing = len(missing_files)
        n_empty = len(empty_files)

        if missing_files:
            first_problem = missing_files[0]
            problem_kind = "missing"
        else:
            first_problem = empty_files[0]
            problem_kind = "empty"

        logging.error(" ==> ERROR! GLOFAS NetCDF input check failed before IBF processing.")
        logging.error(f" ----> Expected input files: {n_expected}")
        logging.error(f" ----> Missing input files: {n_missing}")
        logging.error(f" ----> Empty input files: {n_empty}")
        logging.error(f" ----> First {problem_kind} input file: {first_problem}")
        logging.error(
            " ----> Reason: the GLOFAS downloader did not produce all required "
            "average NetCDF files for this run, or the operational forecast is "
            "not available yet. Run or retry the GLOFAS downloader before this workflow."
        )

        for file_path in missing_files[:10]:
            logging.error(f" ----> Missing: {file_path}")
        for file_path in empty_files[:10]:
            logging.error(f" ----> Empty: {file_path}")

        raise GriddedForecastInputError(
            "required GLOFAS NetCDF input files are missing or empty; "
            f"expected={n_expected}, missing={n_missing}, empty={n_empty}, "
            f"first_problem={first_problem}"
        )

    logging.info(" --> Check GLOFAS NetCDF forecast inputs...DONE")


def _summarize_exception(error: Exception) -> str:
    """Return an operator-friendly one-line explanation for the final ERROR log."""
    if isinstance(error, GriddedForecastInputError):
        return (
            "required GLOFAS NetCDF inputs are missing or empty; this usually means "
            "the GLOFAS downloader did not complete, failed, or the requested forecast "
            f"is not available yet. Details: {error}"
        )

    if isinstance(error, FileNotFoundError):
        return (
            "a required input file was not found during IBF processing. This is likely "
            "an upstream missing-input problem. Details: " + str(error)
        )

    return str(error)


def _restore_legacy_overlay_placeholders(settings_file: str, settings: dict) -> None:
    """
    Preserve AOI placeholders used by early GLOFAS beta configurations.

    ``Settings`` treats ``{domain}`` as the workflow domain everywhere. The
    former GLOFAS configuration also used that token for each AOI. For these
    two overlay templates only, restore the intended late-bound placeholder.
    """
    with open(settings_file, "r") as file_handle:
        source_settings = json.load(file_handle)

    source_impacts = source_settings.get("static_data", {}).get("impacts", {})
    runtime_impacts = settings.get("static_data", {}).get("impacts", {})
    path_pairs = (
        (source_impacts.get("aoi", {}), runtime_impacts.get("aoi", {}), "domain_map"),
        (source_impacts.get("flood_maps", {}), runtime_impacts.get("flood_maps", {}), "file_name"),
    )
    for source_section, runtime_section, key in path_pairs:
        source_value = source_section.get(key)
        if isinstance(source_value, str) and "{domain}" in source_value:
            runtime_section[key] = source_value.replace("{domain}", "{aoi}")


def main(settings_file: str, alg_time: str, domain: str = None) -> None:
    """
    Run threshold classification, flood mapping, and impact assessment for a
    gridded discharge product such as GLOFAS.

    All new processing paths are selected explicitly by this workflow. Existing
    flood and FANFAR workflows continue to call their original classes/methods.
    """
    raw_settings = Settings(settings_file, domain).settings
    _restore_legacy_overlay_placeholders(settings_file, raw_settings)
    date_now = parse_algorithm_time(alg_time)
    settings = format_time_dependent_paths(
        deepcopy(raw_settings),
        date_now,
        raw_settings.get("template", {}),
    )

    logger_level = logging.DEBUG if settings["flags"].get("debug", False) else logging.INFO
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    set_logging_stream(
        logger_folder=settings["log"]["folder"],
        logger_file=settings["log"]["file_name"],
        logger_level=logger_level,
    )

    start_time = time.time()
    logging.info(" ============================================================================ ")
    logging.info(" ==> START GRIDDED DISCHARGE IBF PROCESSING")
    logging.info(f" ==> Time now: {alg_time}")

    input_cfg = settings["input"]
    static_cfg = settings["static_data"]
    outcome_cfg = settings["outcome"]
    flags = settings["flags"]

    _check_forecast_input_files(input_cfg)

    alert_levels_raw = CalculateFloodThresholdLevels(
        forecast_template=input_cfg["forecast_template"],
        time_steps=input_cfg["time_steps"],
        area_file=static_cfg["area"],
        discharge_thresholds=static_cfg["discharge_thresholds"],
        thresholds=settings["thresholds"],
        variable_name=input_cfg.get("variable_name", "dis24"),
    ).run()

    alert_levels_out = alert_levels_raw
    if flags.get("convert_hazard_classes", False):
        alert_levels_out = convert_hazard_classes(
            alert_levels_raw,
            settings["hazard"]["conversion_table"],
        )

    levels_file = _output_path(outcome_cfg["levels"])
    IOHandler.create_directories([os.path.dirname(levels_file)])
    IOHandler.write_tif(
        data=alert_levels_out.values,
        lon=alert_levels_out.lon.values,
        lat=alert_levels_out.lat.values,
        out_filename=levels_file,
        nodata=0,
        dtype="int16",
    )

    admin_gdf = IOHandler.read_vector(static_cfg["warning_regions"])

    if flags.get("hazard_assessment", False):
        classify_admin_hazard(
            admin_gdf=admin_gdf,
            alert_map=alert_levels_raw,
            min_warning_pixel=settings["thresholds"]["min_warning_pixel"],
            output_file=_output_path(outcome_cfg["hazard_shapefile"]),
            output_column=settings["hazard"].get("output_column", "level_GLOFAS"),
        )

    if flags.get("impact_assessment", False):
        impact_settings = static_cfg["impacts"]
        flood_temp_folder = os.path.join(settings["ancillary"]["folder"], "flood_maps")
        flood_map, weighted_map = FloodHazardOverlayMerge(
            overlay_settings=impact_settings,
            temp_folder=flood_temp_folder,
        ).run(alert_levels_raw)

        write_binary_flood_map(
            flood_map,
            _output_path(outcome_cfg["flood_map"]),
        )

        impact_assessment = ImpactAssessment(admin_shape=admin_gdf)
        if os.path.isfile(weighted_map):
            impact_gdf = impact_assessment.run_overlay(
                weighted_flood_map=weighted_map,
                impact_settings=impact_settings,
                hazard=impact_settings.get("hazard_prefix", "GLfl"),
            )
        else:
            logging.warning(
                "Weighted flood-map mosaic not found. Writing a zero-impact shapefile."
            )
            impact_gdf = impact_assessment.empty_overlay(
                hazard=impact_settings.get("hazard_prefix", "GLfl")
            )

        impact_file = _output_path(outcome_cfg["impact_shapefile"])
        IOHandler.create_directories([os.path.dirname(impact_file)])
        impact_gdf.to_file(impact_file)

    if flags.get("clear_ancillary", False) and not flags.get("debug", False):
        logging.info(f"Cleaning ancillary folder: {settings['ancillary']['folder']}")
        shutil.rmtree(settings["ancillary"]["folder"], ignore_errors=True)

    time_elapsed = round(time.time() - start_time, 1)
    logging.info(f" ==> TIME ELAPSED: {time_elapsed} seconds")
    logging.info(" ==> END GRIDDED DISCHARGE IBF PROCESSING")
    logging.info(" ============================================================================ ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-settings_file", required=True, help="Path to the settings file")
    parser.add_argument(
        "-time",
        required=True,
        help='Algorithm time in "YYYY-MM-DD HH:MM" format',
    )
    parser.add_argument(
        "-domain",
        required=False,
        help="Domain to use, overrides settings file",
    )
    args = parser.parse_args()
    try:
        main(args.settings_file, args.time, args.domain)
    except Exception as exc:
        logging.exception(" ==> ERROR! AMHEWAS Continental Watch Hydro failed.")
        logging.error(
            " ==> ERROR! AMHEWAS Continental Watch Hydro failed. Last error summary: %s",
            _summarize_exception(exc),
        )
        raise SystemExit(1)
