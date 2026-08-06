"""
FloodPROOFS with MUL impact-based forecasting workflow.

2026-08-06: Restored the original workflow after the GLOFAS merge and improved
            logging, input checks and final error reporting.
"""

import argparse
import datetime as dt
import logging
import os
import warnings

import geopandas as gpd
import pandas as pd
import pytz

from common.classification import ImpactClassifier
from common.io_handler import IOHandler, format_path_with_time, update_file_paths
from common.logging_handler import set_logging_stream
from common.settings import Settings
from hydro.flood_gridded_return_period import CalculateFloodReturnPeriod
from hydro.flood_hazard_mapping import FloodHazardMerge
from hydro.flood_impact_assessment import ImpactAssessment, initialize_subdomain_inputs


# Helpers ---------------------------------------------------------------------

def parse_algorithm_time(
    alg_time: str,
    forecast_length_h: int,
) -> tuple[dt.datetime, dt.datetime]:
    """Parse the algorithm time and calculate the forecast end time."""
    if forecast_length_h <= 0:
        raise ValueError("forecast_length_h must be greater than zero")

    try:
        date_now = pytz.utc.localize(dt.datetime.strptime(alg_time, "%Y-%m-%d %H:%M"))
    except ValueError as error:
        raise ValueError(
            f"Invalid algorithm time '{alg_time}'. Expected format: YYYY-MM-DD HH:MM"
        ) from error

    forecast_end = date_now + dt.timedelta(hours=forecast_length_h - 1)
    return date_now, forecast_end


# Main ------------------------------------------------------------------------

def main(settings_file: str, alg_time: str, domain: str | None = None) -> None:
    """Run the FloodPROOFS with MUL impact-based forecasting workflow."""

    # Initialize settings
    settings = Settings(settings_file, domain).settings

    # Set up logging
    logger_level = logging.DEBUG if settings["flags"]["debug"] else logging.INFO
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    set_logging_stream(
        logger_folder=settings["log"]["folder"],
        logger_file=settings["log"]["file_name"],
        logger_level=logger_level,
    )

    logging.info(" ============================================================================ ")
    logging.info(" ==> START FLOODPROOFS WITH MUL IBF PROCESSING")
    logging.info(" ==> Algorithm time: %s", alg_time)
    logging.info(" ==> Domain: %s", settings["settings"]["domain"])

    # Parse algorithm time
    date_now, forecast_end = parse_algorithm_time(
        alg_time,
        settings["settings"]["forecast_length_h"],
    )

    # Calculate flood return period
    logging.info(" --> Calculate flood return period")
    return_period_file = CalculateFloodReturnPeriod(
        forecast_length_h=settings["settings"]["forecast_length_h"],
        thresholds=settings["settings"]["thresholds"],
        distribution=settings["settings"]["distribution"],
        static_data=settings["static_data"]["hydro"],
        input_data=settings["input"],
        ancillary_folder=settings["ancillary"]["folder"],
        outcome_folder=settings["outcome"]["return_period"]["folder"],
        outcome_filename=settings["outcome"]["return_period"]["file_name"],
        clear_ancillary_flag=settings["flags"]["clear_ancillary"],
        skip_missing_models=settings["flags"]["skip_missing_models"],
        save_return_period_shapefile=settings["flags"]["save_return_period_shapefile"],
        shapefile_folder=settings["outcome"]["return_period_shapefile"]["folder"],
        shapefile_filename=settings["outcome"]["return_period_shapefile"]["file_name"],
    ).run(date_now, forecast_end)

    # Create flood hazard map
    logging.info(" --> Create flood hazard map")
    hazard_settings = settings["static_data"]["hazard"]
    flood_map, levels_sections = FloodHazardMerge(
        section_map=hazard_settings["section_file"]["file_name"],
        section_map_field=hazard_settings["section_file"]["field"],
        return_periods=hazard_settings["return_period"],
        flood_maps_template=hazard_settings["flood_maps"],
        decode_map=hazard_settings["decode_map"],
        outcome_folder=settings["outcome"]["flood_map"]["folder"],
        outcome_filename=settings["outcome"]["flood_map"]["file_name"],
        skip_empty_maps=settings["flags"]["skip_empty_floodmaps"],
    ).run(date_now, return_period_file)
    logging.info(" --> Flood map saved to: %s", flood_map)

    # Read administrative shapefile
    logging.info(" --> Read administrative shapefile")
    impact_settings = settings["static_data"]["impacts"]
    admin_settings = impact_settings["admin"]
    admin_shapefile = admin_settings["shapefile"]["filename"]
    admin_column = admin_settings["shapefile"]["admin_column"]

    if not os.path.isfile(admin_shapefile):
        raise FileNotFoundError(f"Administrative shapefile not found: {admin_shapefile}")

    domain_shape = gpd.read_file(admin_shapefile)
    if admin_column not in domain_shape.columns:
        raise KeyError(f"Column '{admin_column}' not found in {admin_shapefile}")
    domain_shape.set_index(admin_column, inplace=True)

    # Initialize impacts table
    impacts_table = pd.DataFrame(index=domain_shape.index)
    exposed_elements = []
    mul_settings = impact_settings["MUL"]

    for exposed_element, element_settings in mul_settings.items():
        impacts_table[f"flood_tot_{exposed_element}"] = 0.0
        exposed_elements.append(exposed_element)

        if isinstance(element_settings["files"], dict):
            for sub_category in element_settings["files"]:
                impacts_table[f"flood_tot_{exposed_element}_{sub_category}"] = 0.0

    # Perform impact assessment
    logging.info(" --> Perform impact assessment")
    for subdomain in admin_settings["subdomains"]:
        logging.info(" ---> Processing subdomain: %s", subdomain)
        hydro_to_admin, impact_files = initialize_subdomain_inputs(
            settings["settings"]["domain"],
            subdomain,
            domain_shape,
            mul_settings,
            admin_settings["hydro_to_admin_table"],
        )

        subdomain_impacts = ImpactAssessment(admin_shape=domain_shape).run(
            levels_sections,
            hydro_to_admin,
            impact_files,
            apply_defense=settings["flags"]["apply_flood_protection"],
        )
        impacts_table = impacts_table.add(subdomain_impacts, fill_value=0)

    # Save impact shapefiles
    logging.info(" --> Save impact shapefiles")
    impact_output = settings["outcome"]["impact_shapefile"]

    for exposed_element in exposed_elements:
        logging.info(" ---> Saving impact shapefile for: %s", exposed_element)
        IOHandler.save_impact_shapefiles(
            exposed_element=exposed_element,
            folder_name=format_path_with_time(
                update_file_paths(impact_output["folder"], {"element": exposed_element}),
                date_now,
            ),
            file_name=format_path_with_time(
                update_file_paths(impact_output["file_name"], {"element": exposed_element}),
                date_now,
            ),
            domain_shape=domain_shape.copy(),
            impacts_table=impacts_table,
            hazard="flood",
            rounding=mul_settings.get(exposed_element, {}).get("rounding", False),
        )

    # Classify impact levels
    logging.info(" --> Classify impact levels")
    for element, classification_info in settings["static_data"]["classification"].items():
        logging.info(" ---> Classifying impact levels for: %s", element)
        shapefile_path = format_path_with_time(
            update_file_paths(
                os.path.join(impact_output["folder"], impact_output["file_name"]),
                {"element": element},
            ),
            date_now,
        )
        ImpactClassifier(shapefile_path, classification_info, "flood").classify()

    logging.info(" ==> END FLOODPROOFS WITH MUL IBF PROCESSING")
    logging.info(" ============================================================================ ")


# Command line ----------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FloodPROOFS with MUL IBF workflow")
    parser.add_argument("-settings_file", required=True, help="Path to the settings file")
    parser.add_argument("-time", required=True, help='Algorithm time in "YYYY-MM-DD HH:MM" format')
    parser.add_argument("-domain", required=False, help="Domain override")
    args = parser.parse_args()

    try:
        main(args.settings_file, args.time, args.domain)
    except KeyboardInterrupt:
        logging.warning(" ==> Workflow interrupted by user")
        raise SystemExit(130)
    except Exception:
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
        logging.exception(" ==> ERROR! FloodPROOFS with MUL workflow failed")
        raise SystemExit(1)
