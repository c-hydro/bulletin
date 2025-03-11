import numpy as np
import logging
import os
import pandas as pd
import datetime as dt
from common.io_handler import IOHandler, format_path_with_time
from common.evd import get_distribution
from common.hydro_tools import HydroTools

class CalculateFloodReturnPeriod:
    def __init__(self, forecast_length_h: int, thresholds: dict, distribution: str, static_data: dict, input_data: dict, ancillary_folder: str, outcome_folder: str, outcome_filename: str, clear_ancillary_flag: bool, skip_missing_models: bool, save_return_period_shapefile: bool, shapefile_folder: str or None = None, shapefile_filename: str or None = None):
        """
        Initialize the CalculateFloodReturnPeriod.

        :param forecast_length_h: Forecast length in hours.
        :param thresholds: Dictionary of thresholds.
        :param distribution: Name of the distribution.
        :param static_data: Dictionary of static data.
        :param input_data: Dictionary of input data.
        :param ancillary_folder: Path to the ancillary folder.
        :param outcome_folder: Path to the outcome folder.
        :param outcome_filename: Name of the outcome file.
        :param clear_ancillary_flag: Flag to clear the ancillary folder.
        :param skip_missing_models: Flag to skip missing models.
        :param save_return_period_shapefile: Flag to save the return period shapefile.
        :param shapefile_folder: Path to the shapefile folder.
        :param shapefile_filename: Name of the shapefile file.
        """
        self.forecast_length_h = forecast_length_h
        self.thresholds = thresholds
        self.distribution = distribution
        self.input_data = input_data
        self.static_data = static_data
        self.ancillary_folder = ancillary_folder
        self.outcome_folder = outcome_folder
        self.outcome_filename = outcome_filename
        self.clear_ancillary_flag = clear_ancillary_flag
        self.skip_missing_models = skip_missing_models
        self.save_return_period_shapefile = save_return_period_shapefile
        self.shapefile_folder = shapefile_folder
        self.shapefile_filename = shapefile_filename

    def create_directories(self, date_now: dt.datetime) -> None:
        """
        Create necessary directories.

        :param date_now: Current date.
        """
        logging.info("Creating directories")
        paths = [format_path_with_time(self.outcome_folder, date_now),
                 format_path_with_time(self.ancillary_folder, date_now)]
        IOHandler.create_directories(paths)

    def extract_hmc_results(self, date_now: dt.datetime, forecast_end: dt.datetime) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract HMC results.

        :param date_now: Current date.
        :param forecast_end: Forecast end date.
        :return: Tuple of results, mask, longitude, and latitude.
        """
        logging.info("Extracting HMC results")
        return IOHandler.extract_hmc_results(format_path_with_time(self.input_folder, date_now),
                                             date_now, forecast_end)

    def calculate_max_forecast(self, results: list[np.ndarray]) -> np.ndarray:
        """
        Calculate the maximum forecast.

        :param results: List of forecast results.
        :return: Maximum forecast array.
        """
        logging.info("Calculating max forecast")
        first_step = True
        for map_now in results:
            if first_step:
                dis_max = map_now
                first_step = False
            else:
                dis_max = np.maximum(dis_max, map_now)
        return dis_max

    def read_spatial_maps(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Read spatial maps for distribution parameters and other required data.

        :return: Tuple of theta1, theta2, theta3, average_max, area, and areacell maps.
        """
        logging.info("Reading spatial maps for distribution parameters and other required data")
        theta1_map = IOHandler.read_raster(self.static_data['parameters']['theta1']).values.squeeze()
        theta2_map = IOHandler.read_raster(self.static_data['parameters']['theta2']).values.squeeze()
        theta3_map = IOHandler.read_raster(self.static_data['parameters']['theta3']).values.squeeze()
        average_max_map = IOHandler.read_raster(self.static_data['average_max']).values.squeeze()
        area_map = IOHandler.read_raster(self.static_data['area']).values.squeeze()
        areacell_map = IOHandler.read_raster(self.static_data['areacell']).values.squeeze()
        return theta1_map, theta2_map, theta3_map, average_max_map, area_map, areacell_map

    def calculate_return_period(self, dis_max: np.ndarray, theta1_map: np.ndarray, theta2_map: np.ndarray, theta3_map: np.ndarray, average_max_map: np.ndarray, area_map: np.ndarray, areacell_map: np.ndarray) -> np.ndarray:
        """
        Calculate the return period.

        :param dis_max: Maximum discharge array.
        :param theta1_map: Theta1 map.
        :param theta2_map: Theta2 map.
        :param theta3_map: Theta3 map.
        :param average_max_map: Average maximum map.
        :param area_map: Area map.
        :param areacell_map: Areacell map.
        :return: Return period array.
        """
        logging.info("Calculating return period")
        distribution = get_distribution(self.distribution,
                                        {"theta1": theta1_map, "theta2": theta2_map, "theta3": theta3_map})
        rp = distribution.calculate_return_period(dis_max)

        null_theta1_mask = np.isnan(theta1_map)
        rp[null_theta1_mask] = np.where(dis_max[null_theta1_mask] > average_max_map[null_theta1_mask],
                                        self.thresholds['backup_T'], 1)

        rp[dis_max < self.thresholds['discharge_min']] = 1
        rp[area_map * areacell_map / (10**6) <= self.thresholds['area_km']] = np.nan
        return rp

    def save_results(self, dis_max: np.ndarray, rp: np.ndarray, lon: np.ndarray, lat: np.ndarray, date_now: dt.datetime) -> None:
        """
        Save the results.

        :param dis_max: Maximum discharge array.
        :param rp: Return period array.
        :param lon: Longitude values.
        :param lat: Latitude values.
        :param date_now: Current date.
        """
        logging.info("Saving results")
        max_discharge_path = format_path_with_time(
            os.path.join(self.ancillary_folder, 'max_discharge_%Y%m%d%H%M.tif'), date_now)
        return_period_path = format_path_with_time(
            os.path.join(self.outcome_folder, self.outcome_filename), date_now)
        IOHandler.write_tif(dis_max, lon, lat, max_discharge_path)
        IOHandler.write_tif(rp, lon, lat, return_period_path, nodata=0, dtype='int16')


    def run(self, date_now: dt.datetime, forecast_end: dt.datetime) -> str:
        self.create_directories(date_now)

        dis_max_list = []
        weights = []
        missing_models = []

        for model, data in self.input_data.items():
            try:
                logging.info(f"Processing model {model}")
                results, mask, lon, lat = IOHandler.extract_hmc_results(format_path_with_time(data['folder'], date_now),
                                                                        date_now, forecast_end)
                dis_max = self.calculate_max_forecast(results)
                dis_max_list.append(dis_max)
                weights.append(data['weight'])
            except FileNotFoundError:
                missing_models.append(model)
                if not self.skip_missing_models:
                    raise FileNotFoundError(
                        f"Model results for {model} not found and skip_missing_models is set to False.")

        if missing_models:
            logging.warning(f"Missing models: {', '.join(missing_models)}")
            if self.skip_missing_models:
                total_weight = sum(weights)
                weights = [w / total_weight for w in weights]

        if not dis_max_list:
            raise ValueError("No model results available.")

        if len(dis_max_list) > 1:
            weighted_dis_max = np.average(dis_max_list, axis=0, weights=weights)
        else:
            weighted_dis_max = dis_max_list[0]

        theta1_map, theta2_map, theta3_map, average_max_map, area_map, areacell_map = self.read_spatial_maps()
        rp = self.calculate_return_period(weighted_dis_max, theta1_map, theta2_map, theta3_map, average_max_map,
                                          area_map, areacell_map)

        # Always save the raster
        self.save_results(weighted_dis_max, rp, lon, lat, date_now)

        # Conditionally save the shapefile
        if self.save_return_period_shapefile:
            hydro_tools = HydroTools(self.static_data['area'], self.static_data['areacell'], self.static_data['direction'])
            maps_to_extract = {"rp": rp}
            gdf = hydro_tools.extract_river_geodataframe(area_limit=self.thresholds['area_km'], maps_to_extract=maps_to_extract, include_log_area=True)
            output_shapefile_path = format_path_with_time(os.path.join(self.shapefile_folder, self.shapefile_filename), date_now)
            gdf.to_file(output_shapefile_path, driver="ESRI Shapefile")
            logging.info("Return period shapefile has been created successfully.")

        IOHandler.clear_ancillary_folder(self.ancillary_folder, self.clear_ancillary_flag)
        return format_path_with_time(os.path.join(self.outcome_folder, self.outcome_filename), date_now)