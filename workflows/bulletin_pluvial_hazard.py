import argparse
import datetime as dt
import pytz
import logging
import warnings
import os

from common.settings import Settings
from common.logging_handler import set_logging_stream, reset_logging_stream
from common.io_handler import IOHandler, format_path_with_time

from pluvial.pluvial_input import PluvialInputManager, ModelInput
from pluvial.pluvial_hazard import PluvialHazardAssessment
from pluvial.pluvial_impact import PluvialImpactAssessment


def parse_algorithm_time(alg_time: str) -> dt.datetime:
    """
    Parse algorithm time as UTC datetime.
    Expected format: 'YYYY-MM-DD HH:MM'
    """
    logging.info(f"Parsing algorithm time: {alg_time}")
    return pytz.utc.localize(dt.datetime.strptime(alg_time, "%Y-%m-%d %H:%M"))


def pick_point_id_field(points_gdf) -> str:
    """
    Pick a point id field if present; otherwise create one.
    """
    for candidate in ["id", "ID", "name", "Name"]:
        if candidate in points_gdf.columns:
            return candidate

    points_gdf["id"] = [str(i) for i in range(len(points_gdf))]
    return "id"


def main(settings_file: str, alg_time: str, domain: str | None = None) -> None:
    warnings.filterwarnings("ignore")

    settings_obj = Settings(settings_file=settings_file, domain=domain)
    settings = settings_obj.settings

    date_now = parse_algorithm_time(alg_time)

    # ---------- Configure logging (compatible with your logging_handler.py)
    logger_name = "logger"
    log_cfg = settings.get("log", {})
    log_folder = format_path_with_time(log_cfg.get("folder", ""), date_now)
    log_name = log_cfg.get("file_name", "hazard_assessment.txt")
    IOHandler.create_directories([log_folder])

    set_logging_stream(
        logger_folder=log_folder,
        logger_file=log_name,
        logger_name=logger_name
    )

    try:
        # ---------- Flags
        flags = settings.get("flags", {})
        skip_missing_models = bool(flags.get("skip_missing_models", True))

        # ---------- General settings
        set_cfg = settings.get("settings", {})
        model_blend = str(set_cfg.get("model_blend", "weighted_mean")).lower()
        rolling_mode = str(set_cfg.get("rolling", "sum")).lower()

        # Scenario merging: accept both keys for backward compatibility
        scenario_merging = str(
            set_cfg.get("scenario_merging", set_cfg.get("scenario_merge", "combine"))
        ).lower()

        # Water depth threshold (used to count affected cells)
        water_depth_threshold = float(set_cfg.get("water_depth_threshold", 0.0))

        # Optional point spatial aggregation (e.g., 3x3 weighted quantile)
        pixel_spatial_aggregation = set_cfg.get("pixel_spatial_aggregation", None)

        # ---------- Static data
        static = settings.get("static_data", {})
        points_path = static["shapefile_point"]
        admin_path = static["admin_map"]
        flood_template = static["flood_scenarios"]
        hazard_classification = static["hazard_classification"]

        thresholds_by_duration = static.get("rain_thresholds", static.get("rain_tresholds"))
        if thresholds_by_duration is None:
            raise KeyError("static_data.rain_thresholds missing")
        duration_keys = list(thresholds_by_duration.keys())

        # ---------- Outputs
        out_worst = settings["outcome"]["worst_scenario"]
        out_worst_folder = format_path_with_time(out_worst["folder"], date_now)
        out_worst_name = format_path_with_time(out_worst["file_name"], date_now)
        IOHandler.create_directories([out_worst_folder])
        out_worst_path = os.path.join(out_worst_folder, out_worst_name)

        out_class = settings["outcome"]["classified_maps"]
        out_class_folder = format_path_with_time(out_class["folder"], date_now)
        out_class_name = format_path_with_time(out_class["file_name"], date_now)
        IOHandler.create_directories([out_class_folder])
        out_class_path = os.path.join(out_class_folder, out_class_name)

        # ---------- Read vectors
        points_gdf = IOHandler.read_vector(points_path)
        admin_gdf = IOHandler.read_vector(admin_path)

        point_id_field = pick_point_id_field(points_gdf)

        # ---------- Build model objects from JSON
        models_cfg = settings.get("input", {})
        models: list[ModelInput] = []
        for model_name, model_cfg in models_cfg.items():
            nc_path = format_path_with_time(model_cfg["filename"], date_now)
            varname = model_cfg["varname"]
            weight = float(model_cfg.get("weight", 1.0))
            decumulate = bool(model_cfg.get("decumulate_rain", False))

            models.append(
                ModelInput(
                    name=model_name,
                    filename=nc_path,
                    varname=varname,
                    weight=weight,
                    decumulate_rain=decumulate,
                )
            )

        # ---------- Stage 1: Input
        input_mgr = PluvialInputManager()
        maxima_by_model, weights_by_model = input_mgr.run(
            models=models,
            points_gdf=points_gdf,
            point_id_field=point_id_field,
            duration_keys=duration_keys,
            rolling_mode=rolling_mode,
            point_agg="max",
            skip_missing_models=skip_missing_models,
            spatial_aggregation=pixel_spatial_aggregation,
            extraction_method="nearest",
        )

        # ---------- Stage 2: Hazard
        hazard = PluvialHazardAssessment()
        blended, scenario_by_duration, hazard_da = hazard.run(
            maxima_by_model=maxima_by_model,
            weights_by_model=weights_by_model,
            thresholds_by_duration=thresholds_by_duration,
            duration_keys=duration_keys,
            flood_map_template=flood_template,
            model_blend=model_blend,
            scenario_merging=scenario_merging,
            out_raster_path=out_worst_path,
            nodata=None,
        )

        logging.info(f"Blended maxima by duration: {blended}")
        logging.info(f"Selected scenarios by duration: {scenario_by_duration}")

        if hazard_da is None:
            logging.warning("No duration exceeded any threshold -> no hazard raster generated.")
            return

        # ---------- Stage 3: Impacts
        impacts = PluvialImpactAssessment()
        impacts.run(
            admin_gdf=admin_gdf,
            hazard_raster_path=out_worst_path,
            hazard_classification=hazard_classification,
            out_vector_path=out_class_path,
            affected_value_threshold=water_depth_threshold,
        )

        logging.info(f"Hazard raster written: {out_worst_path}")
        logging.info(f"Admin impacts written: {out_class_path}")

    finally:
        # Always reset logging streams/handlers
        reset_logging_stream(logger_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--settings_file", required=True, help="Path to settings JSON")
    parser.add_argument("-t", "--time", required=True, help="Algorithm time 'YYYY-MM-DD HH:MM' (UTC)")
    parser.add_argument("-d", "--domain", required=False, default=None, help="Domain name (optional)")

    args = parser.parse_args()
    main(settings_file=args.settings_file, alg_time=args.time, domain=args.domain)
