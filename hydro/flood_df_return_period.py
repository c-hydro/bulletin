import numpy as np
import logging
import os
import pandas as pd
import geopandas as gpd
import datetime as dt
from common.io_handler import IOHandler, format_path_with_time
from common.evd import get_distribution, GEVDistribution
from common.hydro_tools import HydroTools

class CalculateFloodReturnPeriod:
    def __init__(self, forecast_length_h: int, distribution: str, static_data: dict, input_data: dict, ancillary_folder: str,  clear_ancillary_flag: bool, shapefile_folder: str or None = None, shapefile_filename: str or None = None):
        """
        Initialize the CalculateFloodReturnPeriod.

        :param forecast_length_h: Forecast length in hours.
        :param distribution: Name of the distribution.
        :param static_data: Dictionary of static data.
        :param input_data: Dictionary of input data.
        :param ancillary_folder: Path to the ancillary folder.
        :param clear_ancillary_flag: Flag to clear the ancillary folder.
        :param shapefile_folder: Path to the shapefile folder.
        :param shapefile_filename: Name of the shapefile file.
        """
        self.forecast_length_h = forecast_length_h
        self.distribution = distribution
        self.input_data = input_data
        self.static_data = static_data
        self.ancillary_folder = ancillary_folder
        self.clear_ancillary_flag = clear_ancillary_flag
        self.shapefile_folder = shapefile_folder
        self.shapefile_filename = shapefile_filename

    def create_directories(self, date_now: dt.datetime) -> None:
        """
        Create necessary directories.

        :param date_now: Current date.
        """
        logging.info("Creating directories")
        paths = [format_path_with_time(self.shapefile_folder, date_now),
                 format_path_with_time(self.ancillary_folder, date_now)]
        IOHandler.create_directories(paths)

    @staticmethod
    def extract_fanfar_results(sections: list, out_fanfar_path: str, date_start: dt.datetime,
                               date_end: dt.datetime) -> pd.DataFrame:
        """
        Extract FANFAR results for a given section list and date range.

        :param sections: List of sections to extract results for.
        :param out_fanfar_path: Path to FANFAR output files.
        :param date_start: Start date.
        :param date_end: End date.
        :return: results dataframe
        """
        logging.info(f"Extracting FANFAR results from {date_start} to {date_end}")
        results = pd.DataFrame(columns=sections, index=pd.date_range(date_start, date_end, freq="D"))

        for section in sections:
            file = os.path.join(format_path_with_time(out_fanfar_path, date_start), f"hydrograph__{section}.txt")
            if os.path.isfile(file):
                logging.debug(f"Reading FANFAR result file {file}")
                frc = IOHandler.read_fanfar_file(file)
                results[section] = frc.reindex(results.index.tz_localize(None), method='nearest').values
            else:
                logging.error(f"Output file {file} not found!")
                raise FileNotFoundError(f"Output file {file} not found!")

        return results

    def run(self, date_now: dt.datetime, forecast_end: dt.datetime) -> pd.DataFrame:
        self.create_directories(date_now)

        logging.info("Reading basin file")
        basins = gpd.read_file(self.static_data['subbasins']["filename"])
        sections = [int(i) for i in basins[self.static_data['subbasins']["column_id"]].values]
        logging.info(f"Found {len(sections)} sections in the basin file")
        if len(sections) == 0:
            raise ValueError("No sections found in the basin file.")

        logging.info("Extracting FANFAR results")
        results = self.extract_fanfar_results(sections, self.input_data['fanfar']['folder'], date_now, forecast_end)
        if results.empty:
            raise ValueError("No FANFAR results found for the specified sections and date range.")

        dis_max = results.max(axis=0)
        parameters = pd.read_csv(self.static_data['parameters'], header=0, index_col=0)
        parameters.columns = [int(i) for i in parameters.columns]
        # drop the columns of parameters not in the sections
        parameters = parameters[parameters.columns.intersection(sections)]

        available_sections = [section for section in sections if section in parameters.columns and section in dis_max.index]
        missing_stations = set(sections) - set(available_sections)
        if missing_stations:
            logging.warning(
                f"Missing parameters or discharge values for stations: {missing_stations}. "
                "These stations will be excluded from the return period calculation."
            )
            dis_max = dis_max[dis_max.index.isin(available_sections)]
            parameters = parameters[available_sections]
        if dis_max.empty:
            raise ValueError("No valid discharge values found after filtering by parameters.")

        logging.info("Calculating return period")
        # Initialize the GEVDistribution
        gev = GEVDistribution(parameters.loc["theta1"].values, parameters.loc["theta2"].values, parameters.loc["theta3"].values)
        # Calculate return periods
        return_periods = gev.calculate_return_period(dis_max.values)
        # Create a DataFrame with the results
        rp_df = pd.DataFrame(index = dis_max.index, data={"T": return_periods})

        # Assign the "T" column to the input section shapefile using the index of rp_df and the self.static_data['subbasins']["column_id"] column of the shapefile
        basins = basins.merge(rp_df, left_on=self.static_data['subbasins']["column_id"], right_index=True)
        output_shapefile_path = format_path_with_time(os.path.join(self.shapefile_folder, self.shapefile_filename),
                                                      date_now)
        os.makedirs(os.path.dirname(output_shapefile_path), exist_ok=True)
        basins.to_file(output_shapefile_path, driver="ESRI Shapefile")
        logging.info("Return period shapefile has been created successfully.")

        rp_df = rp_df.reset_index()
        rp_df = rp_df.rename(columns={"index": "section"})
        # rp_df = rp_df.set_index(np.arange(0,len(rp_df),1))

        return rp_df
