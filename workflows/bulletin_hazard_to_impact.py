import argparse
import datetime as dt
import logging
import os
import time
import warnings
from copy import deepcopy

import pytz
import xarray as xr

from common.io_handler import IOHandler, format_time_dependent_paths
from common.logging_handler import log_workflow_exception, set_logging_stream
from common.settings import Settings
from hydro.flood_impact_assessment import ImpactAssessment
from meteo.meteo_impact import MeteoImpactAssessment
from meteo.meteo_merger import MeteoImpactMerger


class _SafeFormatDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def parse_algorithm_time(alg_time: str) -> dt.datetime:
    """Parse the algorithm time in UTC."""
    return pytz.utc.localize(dt.datetime.strptime(alg_time, "%Y-%m-%d %H:%M"))


def _format_tokens(value: str, **tokens) -> str:
    """Format runtime tokens while preserving placeholders used later."""
    return value.format_map(_SafeFormatDict(tokens))


def _path(path_settings: dict, **tokens) -> str:
    return _format_tokens(
        os.path.join(path_settings["folder"], path_settings["file_name"]),
        **tokens,
    )


def _rename_output_columns(impact_gdf, output_settings: dict):
    column_map = output_settings.get("rename_columns", {})
    missing_columns = set(column_map) - set(impact_gdf.columns)
    if missing_columns:
        raise KeyError(
            "Cannot rename missing impact columns: "
            + ", ".join(sorted(missing_columns))
        )
    return impact_gdf.rename(columns=column_map) if column_map else impact_gdf


def run_weighted_flood_overlay(settings: dict) -> None:
    """Overlay a precomputed weighted flood raster on warning regions."""
    weighted_map = settings["input"]["weighted_flood_map"]
    if not os.path.isfile(weighted_map):
        raise FileNotFoundError(
            "required weighted flood map not found; Continental Watch Hydro must "
            f"complete successfully before this overlay. File: {weighted_map}"
        )

    static_cfg = settings["static_data"]
    impact_settings = static_cfg["impacts"]
    logging.info("Using precomputed weighted flood map: %s", weighted_map)

    admin_gdf = IOHandler.read_vector(static_cfg["warning_regions"])
    impact_gdf = ImpactAssessment(admin_shape=admin_gdf).run_overlay(
        weighted_flood_map=weighted_map,
        impact_settings=impact_settings,
        hazard=impact_settings.get("hazard_prefix", "GLfl"),
    )

    impact_output = settings["outcome"]["impact_shapefile"]
    impact_gdf = _rename_output_columns(impact_gdf, impact_output)
    impact_file = _path(impact_output)
    IOHandler.create_directories([os.path.dirname(impact_file)])
    impact_gdf.to_file(impact_file)


def _meteo_output_paths(settings: dict, model: str) -> list[str]:
    hazards = settings["settings"]["hazards"]
    output_cfg = settings["outcome"]["impact_shapefile"]
    return [
        _path(
            output_cfg,
            model=model,
            hazard=hazard,
            HAZARD=hazard.upper(),
        )
        for hazard in hazards
    ]


def run_meteo_classified_overlay(settings: dict) -> None:
    """Overlay preclassified meteo datasets and optionally merge the models."""
    set_cfg = settings["settings"]
    flags = settings.get("flags", {})
    models = settings["input"]["models"]
    hazards = list(set_cfg["hazards"])
    hazards_short = list(set_cfg.get("hazards_short", hazards))
    static_cfg = settings["static_data"]
    impact_settings = static_cfg["impacts"]
    output_cfg = settings["outcome"]["impact_shapefile"]
    exposed_element = set_cfg.get("exposed_element", "population")
    overwrite = bool(flags.get("overwrite", False))
    raise_if_missing = bool(flags.get("raise_error_if_missing", True))

    admin_gdf = IOHandler.read_vector(static_cfg["warning_regions"])
    impact_handler = MeteoImpactAssessment()
    completed_models: dict[str, dict] = {}
    missing_models: list[str] = []

    for model, model_cfg in models.items():
        outputs = _meteo_output_paths(settings, model)
        if not overwrite and all(os.path.isfile(path) for path in outputs):
            logging.info("All impact outputs already exist for model %s", model)
            completed_models[model] = model_cfg
            continue

        classified_map = model_cfg["classified_map"]
        if not os.path.isfile(classified_map):
            logging.warning("Missing classified map for model %s: %s", model, classified_map)
            missing_models.append(model)
            continue

        logging.info("Processing classified meteo map for model %s: %s", model, classified_map)
        with xr.open_dataset(classified_map) as alert_daily:
            absent_hazards = [hazard for hazard in hazards if hazard not in alert_daily]
            if absent_hazards:
                raise KeyError(
                    f"Model {model} is missing hazards: {', '.join(absent_hazards)}"
                )
            impact_outputs = impact_handler.classify_warning_levels_impact_based_multi_hazard(
                hazards=hazards,
                hazards_short=hazards_short,
                admin_gdf=admin_gdf,
                alert_daily=alert_daily,
                impact_settings=impact_settings,
                exposed_element=exposed_element,
                progress_every=int(set_cfg.get("impact_progress_every", 25)),
            )

        for hazard, output_path in zip(hazards, outputs):
            IOHandler.create_directories([os.path.dirname(output_path)])
            impact_handler.save(impact_outputs[hazard], output_path)
        completed_models[model] = model_cfg

    if missing_models and raise_if_missing:
        raise FileNotFoundError(
            "Missing classified maps for models: " + ", ".join(missing_models)
        )

    merger_cfg = settings.get("merger")
    if merger_cfg and flags.get("run_multimodel_merger", True):
        if not completed_models:
            raise RuntimeError("No meteo model outputs are available for the merger")
        MeteoImpactMerger().run(
            hazards=hazards,
            hazards_short=hazards_short,
            models=completed_models,
            file_template=merger_cfg["input_file"],
            risk_thresholds=impact_settings["risk_thresholds"],
            out_file_template=merger_cfg["output_file"],
            raise_error_if_missing=bool(merger_cfg.get("raise_error_if_missing", True)),
        )


def main(settings_file: str, alg_time: str, domain: str = None) -> None:
    raw_settings = Settings(settings_file, domain).settings
    date_now = parse_algorithm_time(alg_time)
    settings = format_time_dependent_paths(
        deepcopy(raw_settings),
        date_now,
        raw_settings.get("template", {}),
    )

    logger_level = logging.DEBUG if settings.get("flags", {}).get("debug", False) else logging.INFO
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    set_logging_stream(
        logger_folder=settings["log"]["folder"],
        logger_file=settings["log"]["file_name"],
        logger_level=logger_level,
    )

    operation = settings["settings"]["operation"]
    start_time = time.time()
    logging.info(" ============================================================================ ")
    logging.info(" ==> START HAZARD-TO-IMPACT WORKFLOW")
    logging.info(" ==> Operation: %s", operation)
    logging.info(" ==> Time now: %s", alg_time)

    if operation == "weighted_flood_overlay":
        run_weighted_flood_overlay(settings)
    elif operation == "meteo_classified_overlay":
        run_meteo_classified_overlay(settings)
    else:
        raise ValueError(f"Unsupported hazard-to-impact operation: {operation}")

    logging.info(" ==> TIME ELAPSED: %s seconds", round(time.time() - start_time, 1))
    logging.info(" ==> END HAZARD-TO-IMPACT WORKFLOW")
    logging.info(" ============================================================================ ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-settings_file", required=True, help="Path to the settings file")
    parser.add_argument("-time", required=True, help='Algorithm time in "YYYY-MM-DD HH:MM" format')
    parser.add_argument("-domain", required=False, help="Domain override")
    args = parser.parse_args()
    try:
        main(args.settings_file, args.time, args.domain)
    except Exception as exc:
        log_workflow_exception("Hazard-to-impact workflow", exc)
        raise SystemExit(1)
