import geopandas as gpd
import numpy as np
from shapely.geometry import LineString
import logging
import os
from common.io_handler import IOHandler
import rasterio as rio

class HydroTools:
    def __init__(self, static_directory:str, domain:str):
        """
        Initialize HydroTools with paths to area, areacell, and direction rasters.

        :param area: Path to the area raster file.
        :param areacell: Path to the areacell raster file.
        :param direction: Path to the direction raster file.
        """
        self.area = os.path.join(static_directory, domain + ".area.txt")
        self.areacell = os.path.join(static_directory, domain + ".areacell.txt")
        self.direction = os.path.join(static_directory, domain + ".pnt.txt")

    def calculate_area_km(self):
        """
        Calculate the area in square kilometers.

        :return: Numpy array of area in square kilometers.
        """
        io_handler = IOHandler()
        ups_mask = io_handler.read_raster(self.area).values.squeeze()
        area_cell = io_handler.read_raster(self.areacell).values.squeeze()
        area_km = ups_mask * area_cell / 1e6
        return area_km

    def extract_river_geodataframe(self, area_limit, maps_to_extract: dict or None, include_log_area: bool = False,
                                   include_area_km: bool = False) -> gpd.GeoDataFrame:
        """
        Extract a GeoDataFrame of the river network based on the area limit.

        :param area_limit: Minimum area limit to include in the river network.
        :param maps_to_extract: Dictionary of additional maps to extract.
        :param include_log_area: Whether to include the log area in the output.
        :param include_area_km: Whether to include the area in square kilometers in the output.
        :return: GeoDataFrame of the river network.
        """
        # Load raster maps
        logging.info("Load drainage direction...")
        io_handler = IOHandler()
        drainage_directions = io_handler.read_raster(self.direction)
        transform = drainage_directions.rio.transform()
        crs = drainage_directions.rio.crs
        if crs is None:
            crs = "EPSG:4326"
            logging.warning("CRS not found in raster file. Defaulting to EPSG:4326")
        drainage_directions = drainage_directions.values.squeeze()

        area_km = self.calculate_area_km()

        # Convert drainage directions to displacement
        direction_offsets = {
            1: (1, -1), 2: (1, 0), 3: (1, 1),
            4: (0, -1), 5: (0, 0), 6: (0, 1),
            7: (-1, -1), 8: (-1, 0), 9: (-1, 1)
        }

        # Extract river network points (cell indices)
        logging.info("Extract river network points...")
        rows, cols = np.where(area_km >= area_limit)
        lines = []

        if maps_to_extract is None:
            maps_to_extract = {}

        if include_area_km:
            maps_to_extract["area_km"] = area_km

        if include_log_area:
                maps_to_extract["log_ups"] = np.floor(np.where(area_km > 0, np.log2(area_km + 1), np.nan)).astype(np.int16)

        tabular_values = {}
        for key in maps_to_extract.keys():
            tabular_values[key] = []

        for row, col in zip(rows, cols):
            direction = drainage_directions[row, col]
            if direction in direction_offsets:
                dr, dc = direction_offsets[direction]
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < drainage_directions.shape[0] and 0 <= new_col < drainage_directions.shape[1]:
                    start_coords = rio.transform.xy(transform, row, col)
                    end_coords = rio.transform.xy(transform, new_row, new_col)
                    lines.append(LineString([start_coords, end_coords]))
                    for key in maps_to_extract.keys():
                        tabular_values[key].append(maps_to_extract[key][row, col])

        # Create GeoDataFrame with matching data lengths
        logging.info("Save output...")
        gdf = gpd.GeoDataFrame(tabular_values, geometry=lines, crs=crs)
        gdf.crs = "EPSG:4326"
        return gdf