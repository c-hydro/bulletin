import argparse
import datetime as dt
import pytz
import logging
import warnings
import os
import pandas as pd
import geopandas as gpd
from common.settings import Settings
from common.logging_handler import set_logging_stream
from hydro.flood_return_period import CalculateFloodReturnPeriod
from hydro.flood_hazard_mapping import FloodHazardMerge
from hydro.flood_impact_assessment import ImpactAssessment, initialize_subdomain_inputs
from common.classification import ImpactClassifier
from common.io_handler import IOHandler, format_path_with_time, update_file_paths

def parse_algorithm_time(alg_time: str, forecast_length_h: int) -> tuple[dt.datetime, dt.datetime]:
    """
    Parse the algorithm time.

    :param alg_time: Algorithm time in "YYYY-MM-DD HH:MM" format.
    :param forecast_length_h: Forecast length in hours.
    :return: Tuple of current date and forecast end date.
    """
    logging.info(f"Parsing algorithm time: {alg_time}")
    date_now = pytz.utc.localize(dt.datetime.strptime(alg_time, "%Y-%m-%d %H:%M"))
    forecast_end = date_now + dt.timedelta(hours=forecast_length_h - 1)
    return date_now, forecast_end

def main(settings_file: str, alg_time: str, domain: str = None) -> None:
    """
    Main function to run the flood impact-based forecasting.

    :param settings_file: Path to the settings file.
    :param alg_time: Algorithm time in "YYYY-MM-DD HH:MM" format.
    :param domain: Domain to use, overrides settings file.
    """
    # Initialize settings
    settings = Settings(settings_file, domain).settings

    # Set up logging
    logger_level = logging.DEBUG if settings['flags']['debug'] else logging.INFO
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    set_logging_stream(
        logger_folder=settings['log']['folder'],
        logger_file=settings['log']['file_name'],
        logger_level=logger_level
    )

    # Parse algorithm time
    date_now, forecast_end = parse_algorithm_time(alg_time, settings['settings']['forecast_length_h'])

    # Create HazardAssessment instance
    rp_file = CalculateFloodReturnPeriod(
         forecast_length_h=settings['settings']['forecast_length_h'],
         thresholds=settings['settings']['thresholds'],
         distribution=settings['settings']['distribution'],
         static_data=settings['static_data']['hydro'],
         input_data=settings['input'],
         ancillary_folder=settings['ancillary']['folder'],
         outcome_folder=settings['outcome']['return_period']['folder'],
         outcome_filename=settings['outcome']['return_period']['file_name'],
         clear_ancillary_flag=settings['flags']['clear_ancillary'],
         skip_missing_models=settings['flags']['skip_missing_models']
     ).run(date_now, forecast_end)

    # Run flood hazard mapping
    hazard_dict = settings['static_data']['hazard']
    flood_map, levels_sections = FloodHazardMerge(
         section_map=hazard_dict["section_file"]["file_name"],
         section_map_field=hazard_dict["section_file"]["field"],
         return_periods=hazard_dict["return_period"],
         flood_maps_template=hazard_dict["flood_maps"],
         decode_map=hazard_dict["decode_map"],
         outcome_folder=settings['outcome']['flood_map']['folder'],
         outcome_filename=settings['outcome']['flood_map']["file_name"]
    ).run(date_now, rp_file)

    logging.info("Perform impact assessment")
    # Read the shapefile for the whole domain
    domain_shape = gpd.read_file(settings['static_data']['impacts']['admin']['shapefile']['filename'])
    domain_shape.set_index(settings['static_data']['impacts']['admin']['shapefile']['admin_column'], inplace=True)

    # Initialize the impacts table with all categories and sub-categories
    impacts_table = pd.DataFrame(index=domain_shape.index)
    exposed_elements = []
    for exposed_element, subcategories in settings['static_data']['impacts']['MUL'].items():
        impacts_table["flood_tot_" + exposed_element] = 0.0
        if isinstance(subcategories['files'], dict):
            for sub_category in subcategories['files']:
                impacts_table["flood_tot_" + exposed_element + "_" + sub_category] = 0.0
        exposed_elements = exposed_elements + [exposed_element]

    # Loop through the subdomains
    for subdomain in settings['static_data']['impacts']['admin']['subdomains']:
        logging.info(f"Processing subdomain: {subdomain}")
        hydro_to_admin, impact_files = initialize_subdomain_inputs(
            settings['settings']['domain'], subdomain, domain_shape,
            settings['static_data']['impacts']['MUL'],
            settings['static_data']['impacts']['admin']['hydro_to_admin_table']
        )

        # Perform the impact assessment
        impact_assessment = ImpactAssessment(admin_shape=domain_shape)
        subdomain_impacts_table = impact_assessment.run(levels_sections, hydro_to_admin, impact_files)
        impacts_table = impacts_table.add(subdomain_impacts_table, fill_value=0)

    logging.info("Save output shapefiles")
    # Save the final shapefiles with the calculated impacts
    for exposed_element in exposed_elements:
        domain_shape_toedit = domain_shape.copy()
        logging.info(f"Saving shapefiles for {exposed_element}")
        IOHandler.save_impact_shapefiles(
            exposed_element = exposed_element,
            folder_name= format_path_with_time(update_file_paths(settings['outcome']['impact_shapefile']['folder'], {"element": exposed_element}), date_now),
            file_name= format_path_with_time(update_file_paths(settings['outcome']['impact_shapefile']['file_name'], {"element": exposed_element}), date_now),
            domain_shape= domain_shape_toedit,
            impacts_table=impacts_table,
            hazard = "flood",
            rounding = settings['static_data']['impacts']['MUL'].get(exposed_element, {}).get('rounding', False)
        )

    logging.info("Classify impact levels")
    # Perform classification for each element listed in the "classification" key
    for element, classification_info in settings['static_data']['classification'].items():
        logging.info(f"Classifying impact levels for {element}")
        shapefile_path = format_path_with_time(update_file_paths(os.path.join(
            settings['outcome']['impact_shapefile']['folder'], settings['outcome']['impact_shapefile']['file_name']),
            {"element": element}), date_now)
        classifier = ImpactClassifier(shapefile_path, classification_info, "flood")
        classifier.classify()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-settings_file', required=True, help='Path to the settings file')
    parser.add_argument('-time', required=True, help='Algorithm time in "YYYY-MM-DD HH:MM" format')
    parser.add_argument('-domain', required=False, help='Domain to use, overrides settings file')
    args = parser.parse_args()
    main(args.settings_file, args.time, args.domain)