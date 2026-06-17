import argparse
import datetime as dt
import logging
import time
import warnings

from common.settings import Settings
from common.logging_handler import set_logging_stream, reset_logging_stream
from common.io_handler import IOHandler, format_path_with_time
from common.time_handler import parse_algorithm_time
from meteo.meteo_merger import MeteoImpactMerger
from workflows.bulletin_meteo_ibf_single import (
    build_gridded_hazard_path,
    get_model_flags,
    run_hazard_stage,
    run_admin_stage,
)


def setup_logger(settings: dict, date_now: dt.datetime) -> None:
    """
    Set up logging from settings.
    """
    log_cfg = settings.get("log", {})
    log_folder = format_path_with_time(log_cfg.get("folder", "./logs"), date_now)
    log_file = format_path_with_time(log_cfg.get("file_name", "bulletin_meteo_ibf_multimodel.txt"), date_now)
    logger_level = logging.DEBUG if settings.get("flags", {}).get("debug", False) else logging.INFO
    set_logging_stream(logger_folder=log_folder, logger_file=log_file, logger_level=logger_level)


def run_merger(settings: dict, date_now: dt.datetime) -> None:
    """
    Merge model impact outputs already produced by the single-model stage.
    """
    flags = settings.get("flags", {})
    if not flags.get("run_merger", True):
        return

    merger_cfg = settings.get("merger")
    if merger_cfg is None:
        logging.info("No merger settings found. Skipping merger stage.")
        return

    hazards = settings["settings"]["hazards"]
    hazards_short = settings["settings"].get("hazards_short", [hazard[:4] for hazard in hazards])
    models = settings["input"]["models"]

    file_template = format_path_with_time(merger_cfg["input_file"], date_now)
    out_template = format_path_with_time(merger_cfg["output_file"], date_now)
    risk_thresholds = settings["static_data"]["impacts"]["risk_thresholds"]

    merger = MeteoImpactMerger()
    merger.run(
        hazards=hazards,
        hazards_short=hazards_short,
        models=models,
        file_template=file_template,
        risk_thresholds=risk_thresholds,
        out_file_template=out_template,
        raise_error_if_missing=bool(merger_cfg.get("raise_error_if_missing", False)),
    )


def main(settings_file: str, alg_time: str, domain: str | None = None) -> None:
    """
    Main function to run a multimodel meteo IBF bulletin.
    """
    warnings.filterwarnings("ignore")
    settings = Settings(settings_file, domain).settings
    date_now = parse_algorithm_time(alg_time)
    setup_logger(settings, date_now)

    start_time = time.time()
    logging.info(" ============================================================================ ")
    logging.info(" ==> START ... ")
    logging.info(f" --> Time now : {alg_time}")

    try:
        models = settings["input"]["models"]
        for model_name, model_cfg in models.items():
            logging.info(f" --> Run model: {model_name}")
            model_flags = get_model_flags(settings, model_cfg)
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

        run_merger(settings, date_now)

        time_elapsed = round(time.time() - start_time, 1)
        logging.info(" ")
        logging.info(" ==> bulletin - meteo IBF multimodel (Version: 1.0.0 Release_Date: 2026-06-11)")
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
    args = parser.parse_args()
    main(args.settings_file, args.time, args.domain)
