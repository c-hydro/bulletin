import argparse
import datetime as dt
import logging
import os
import time
import warnings

from common.settings import Settings
from common.logging_handler import set_logging_stream, reset_logging_stream
from common.io_handler import IOHandler, format_path_with_time, update_file_paths
from common.time_handler import parse_algorithm_time, build_time_tokens
from meteo.meteo_hazard import MeteoVariable, MeteoForecastInput, MeteoHazardAssessment
from meteo.meteo_impact import MeteoImpactAssessment


def build_variables(variables_settings: dict, date_now: dt.datetime, tokens: dict | None = None) -> dict[str, MeteoVariable]:
    """
    Build variable objects from settings.
    """
    variables = {}
    tokens = tokens or {}
    for variable_key, variable_settings in variables_settings.items():
        filename = variable_settings.get("filename", variable_settings.get("file"))
        if filename is not None:
            filename = format_path_with_time(update_file_paths(filename, tokens), date_now)
        variables[variable_key] = MeteoVariable(
            name=variable_key,
            varname=variable_settings["name"],
            filename=filename,
            level=variable_settings.get("level"),
            date_selected=variable_settings.get("date_selected"),
            accumulated=bool(variable_settings.get("accumulated", False)),
        )
    return variables


def setup_logger(settings: dict, date_now: dt.datetime) -> None:
    """
    Set up logging from settings.
    """
    log_cfg = settings.get("log", {})
    log_folder = format_path_with_time(log_cfg.get("folder", "./logs"), date_now)
    log_file = format_path_with_time(log_cfg.get("file_name", "bulletin_meteo_ibf.txt"), date_now)
    logger_level = logging.DEBUG if settings.get("flags", {}).get("debug", False) else logging.INFO
    set_logging_stream(logger_folder=log_folder, logger_file=log_file, logger_level=logger_level)


def resolve_model_name(settings: dict, model_name: str | None = None) -> str:
    """
    Resolve the model to run from input.models.
    """
    models = settings.get("input", {}).get("models", {})
    if not models:
        raise KeyError("input.models missing")

    if model_name is not None:
        if model_name not in models:
            raise KeyError(f"Model '{model_name}' not found in input.models")
        return model_name

    if len(models) == 1:
        return next(iter(models.keys()))

    raise ValueError("Multiple input.models configured. Use -model for the single-model workflow.")


def get_model_settings(settings: dict, model_name: str) -> dict:
    """
    Get settings for one model.
    """
    return settings["input"]["models"][model_name]


def get_model_flags(settings: dict, model_settings: dict) -> dict:
    """
    Merge global flags with optional model-specific flags.
    """
    flags = dict(settings.get("flags", {}))
    flags.update(model_settings.get("flags", {}))
    return flags


def get_model_time_tokens(settings: dict, model_name: str, model_settings: dict, date_now: dt.datetime) -> dict:
    """
    Build time and model tokens, allowing model-specific template overrides.
    """
    template_settings = dict(settings.get("settings", {}).get("template", {}))
    template_settings.update(model_settings.get("template", {}))
    return build_time_tokens(template_settings, date_now, extra_tokens={"model": model_name})


def format_path_from_cfg(path_cfg: dict, tokens: dict, date_now: dt.datetime) -> str:
    """
    Build a dated path from a {"folder", "file_name"} settings block.
    """
    return format_path_with_time(
        update_file_paths(os.path.join(path_cfg["folder"], path_cfg["file_name"]), tokens),
        date_now,
    )


def build_gridded_hazard_path(settings: dict, date_now: dt.datetime, model_name: str) -> str:
    """
    Build the gridded hazard path from outcome.gridded_hazard.
    """
    model_settings = get_model_settings(settings, model_name)
    tokens = get_model_time_tokens(settings, model_name, model_settings, date_now)
    return format_path_from_cfg(settings["outcome"]["gridded_hazard"], tokens, date_now)


def run_hazard_stage(settings: dict, date_now: dt.datetime, model_name: str | None = None) -> tuple[str, str, object]:
    """
    Run forecast input and gridded hazard classification.
    """
    set_cfg = settings.get("settings", {})
    static_cfg = settings.get("static_data", {})
    ancillary_cfg = settings.get("ancillary", {})
    outcome_cfg = settings.get("outcome", {})

    model_name = resolve_model_name(settings, model_name)
    model_settings = get_model_settings(settings, model_name)
    flags = get_model_flags(settings, model_settings)
    hazards = set_cfg["hazards"]
    forecast_length_h = int(set_cfg["forecast_length_h"])
    forecast_resolution_h = float(set_cfg.get("forecast_resolution_h", 1))
    rain_window_h = float(set_cfg.get("rain_window_h", 24))
    forecast_end = date_now + dt.timedelta(hours=forecast_length_h - 1)

    tokens = get_model_time_tokens(settings, model_name, model_settings, date_now)

    source = str(model_settings.get("source", "local")).lower()
    variables_settings = model_settings["variables"]
    variables = build_variables(variables_settings, date_now, tokens=tokens)

    input_handler = MeteoForecastInput()
    if source == "drops2":
        drops_cfg = model_settings["drops2"]
        date_from = date_now - dt.timedelta(hours=int(drops_cfg.get("past_time_search_window_h", 0)))
        date_to = date_now + dt.timedelta(hours=int(drops_cfg.get("future_time_search_window_h", 0)))
        variables_dic, date_ref = input_handler.read_drops_variables(variables, drops_cfg, date_from, date_to)
        date_ref = date_ref.replace(tzinfo=dt.timezone.utc) if date_ref.tzinfo is None else date_ref
    elif source == "local":
        variables_dic = input_handler.read_local_variables(variables, time_tokens=tokens)
        date_ref = date_now
    else:
        raise NotImplementedError("Model source must be 'local' or 'drops2'")

    data = input_handler.build_dataset(variables_dic)
    data = input_handler.crop_bbox(data, set_cfg.get("bbox"))
    data = input_handler.slice_time(data, forecast_end)

    ancillary_forecast = ancillary_cfg.get("forecast")
    if ancillary_forecast is not None and flags.get("save_ancillary", True):
        ancillary_forecast_path = format_path_from_cfg(ancillary_forecast, tokens, date_now)
        IOHandler.create_directories([os.path.dirname(ancillary_forecast_path)])
        data.to_netcdf(ancillary_forecast_path)

    processed = input_handler.preprocess_hazards(
        data=data,
        hazards=hazards,
        variables_settings=variables_settings,
        forecast_resolution_h=forecast_resolution_h,
        rain_window_h=rain_window_h,
    )
    daily = input_handler.daily_maxima(processed, date_ref, forecast_end)

    ancillary_daily = ancillary_cfg.get("daily_maxima")
    if ancillary_daily is not None and flags.get("save_ancillary", True):
        ancillary_daily_path = format_path_from_cfg(ancillary_daily, tokens, date_now)
        IOHandler.create_directories([os.path.dirname(ancillary_daily_path)])
        daily.to_netcdf(ancillary_daily_path)

    hazard_handler = MeteoHazardAssessment()
    thresholds_cfg = static_cfg.get("thresholds", static_cfg.get("tresholds"))
    if thresholds_cfg is None:
        raise KeyError("static_data.thresholds missing")
    alert_daily = hazard_handler.classify(daily, hazards, thresholds_cfg)
    if flags.get("mask_sea", False):
        alert_daily = hazard_handler.apply_sea_mask(alert_daily, static_cfg.get("sea_mask"))

    alert_out = outcome_cfg["gridded_hazard"]
    alert_path = format_path_from_cfg(alert_out, tokens, date_now)
    IOHandler.create_directories([os.path.dirname(alert_path)])
    alert_daily.to_netcdf(alert_path)

    return alert_path, str(date_ref), alert_daily


def run_admin_stage(settings: dict, date_now: dt.datetime, alert_daily, model_name: str | None = None) -> None:
    """
    Run admin-level hazard and impact assessment.
    """
    set_cfg = settings.get("settings", {})
    static_cfg = settings.get("static_data", {})
    outcome_cfg = settings.get("outcome", {})

    model_name = resolve_model_name(settings, model_name)
    model_settings = get_model_settings(settings, model_name)
    flags = get_model_flags(settings, model_settings)
    hazards = set_cfg["hazards"]
    hazards_short = set_cfg.get("hazards_short", [hazard[:4] for hazard in hazards])
    tokens = get_model_time_tokens(settings, model_name, model_settings, date_now)

    admin_gdf = IOHandler.read_vector(static_cfg["warning_regions"])
    impact_handler = MeteoImpactAssessment()

    if flags.get("hazard_assessment", True):
        hazard_out = outcome_cfg["shape_hazard"]
        for hazard, hazard_short in zip(hazards, hazards_short):
            out_path = format_path_with_time(
                update_file_paths(os.path.join(hazard_out["folder"], hazard_out["file_name"]), tokens | {"hazard": hazard, "HAZARD": hazard.upper()}),
                date_now,
            )
            out_gdf = impact_handler.classify_warning_levels_pure_hazard(
                hazard=hazard,
                hazard_short=hazard_short,
                admin_gdf=admin_gdf,
                alert_daily=alert_daily,
                min_warning_threshold=int(set_cfg.get("min_warning_pixel", 1)),
            )
            impact_handler.save(out_gdf, out_path)

    if flags.get("impact_assessment", True):
        impact_out = outcome_cfg["shape_impacts"]
        impact_settings = static_cfg["impacts"]
        for exposed_element in impact_settings["exposed_map"].keys():
            for hazard, hazard_short in zip(hazards, hazards_short):
                out_path = format_path_with_time(
                    update_file_paths(
                        os.path.join(impact_out["folder"], impact_out["file_name"]),
                        tokens | {
                            "hazard": hazard,
                            "HAZARD": hazard.upper(),
                            "exposed_element": exposed_element,
                        },
                    ),
                    date_now,
                )
                out_gdf = impact_handler.classify_warning_levels_impact_based(
                    hazard=hazard,
                    hazard_short=hazard_short,
                    admin_gdf=admin_gdf,
                    alert_daily=alert_daily,
                    impact_settings=impact_settings,
                    exposed_element=exposed_element,
                )
                impact_handler.save(out_gdf, out_path)


def main(settings_file: str, alg_time: str, domain: str | None = None, model: str | None = None) -> None:
    """
    Main function to run a single-model meteo IBF bulletin.
    """
    warnings.filterwarnings("ignore")
    settings = Settings(settings_file, domain).settings
    date_now = parse_algorithm_time(alg_time)
    setup_logger(settings, date_now)
    model_name = resolve_model_name(settings, model)
    model_flags = get_model_flags(settings, get_model_settings(settings, model_name))

    start_time = time.time()
    logging.info(" ============================================================================ ")
    logging.info(" ==> START ... ")
    logging.info(f" --> Time now : {alg_time}")

    try:
        alert_daily = None
        if model_flags.get("run_hazard", True):
            alert_path, date_ref, alert_daily = run_hazard_stage(settings, date_now, model_name=model_name)
            logging.info(f"Hazard output written: {alert_path}")
            logging.info(f"Forecast reference time: {date_ref}")
        else:
            alert_path = build_gridded_hazard_path(settings, date_now, model_name)
            logging.info(f"Reading existing gridded hazard output: {alert_path}")
            alert_daily = IOHandler.open_netcdf_dataset(alert_path)

        if model_flags.get("run_admin", True):
            run_admin_stage(settings, date_now, alert_daily, model_name=model_name)

        time_elapsed = round(time.time() - start_time, 1)
        logging.info(" ")
        logging.info(" ==> bulletin - meteo IBF single (Version: 1.0.0 Release_Date: 2026-06-11)")
        logging.info(f" ==> TIME ELAPSED: {time_elapsed} seconds")
        logging.info(" ==> ... END")
        logging.info(" ==> Bye, Bye")
        logging.info(" ============================================================================ ")
    finally:
        reset_logging_stream("logger")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-settings_file", required=True, help="Path to the settings file")
    parser.add_argument("-time", required=True, help="Algorithm time in 'YYYY-MM-DD HH:MM' format")
    parser.add_argument("-domain", required=False, help="Domain to use, overrides settings file")
    parser.add_argument("-model", required=False, help="Optional model token for path templates")
    args = parser.parse_args()
    main(args.settings_file, args.time, args.domain, args.model)
