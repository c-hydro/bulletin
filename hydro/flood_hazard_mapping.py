import numpy as np
import os
import logging
import geopandas as gpd
import datetime as dt
import xarray as xr
from common.io_handler import IOHandler, format_path_with_time

class FloodHazardMerge:
    def __init__(self, section_map: str, section_map_field: str, return_periods: list[int], flood_maps_template: str, decode_map: str, outcome_folder: str, outcome_filename: str):
        """
        Initialize the FloodHazardMerge.

        :param section_map: Path to the section map shapefile.
        :param section_map_field: Field name in the section map.
        :param return_periods: List of return periods.
        :param flood_maps_template: Template path for flood maps.
        :param decode_map: Path to the decode map.
        :param outcome_folder: Folder to save the outcome.
        :param outcome_filename: Name of the outcome file.
        """
        self.section_map = section_map
        self.section_map_field = section_map_field
        self.return_periods = return_periods
        self.flood_maps_template = flood_maps_template
        self.decode_map = decode_map
        self.outcome_folder = outcome_folder
        self.outcome_filename = outcome_filename

    def create_flood_map(self, rp_raster: 'xr.DataArray') -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Create a flood map.

        :param rp_raster: Raster of return periods.
        :return: Tuple of mosaic flood map, latitude mosaic, longitude mosaic, and levels sections.
        """
        logging.info("Tailoring flood map")
        section_gdf = gpd.read_file(self.section_map)

        logging.info("Assign the return period to the sections (it might take a while)")
        section_gdf = self.assign_return_periods(section_gdf, rp_raster)

        levels_sections = {}
        available_rps = np.array(self.return_periods)

        # Convert the raster of return periods to the available ones
        section_gdf['T'] = section_gdf['T'].apply(
            lambda T: available_rps[available_rps <= T].max() if T > 1 and not np.isnan(T) else T)

        first_map = True
        for T in np.unique(section_gdf["T"]):
            if T <= 1 or np.isnan(T):
                continue

            logging.info(f'Import maps for RP {int(T)} years')
            flood_map_level = self.import_flood_map_level(T)
            if first_map:
                mosaic_flood_map, lat_mosaic, lon_mosaic, decode_map = self.initialize_mosaic(flood_map_level)
                first_map = False
            logging.info(f'Select and mosaic map for RP {int(T)} years')
            codes_in = section_gdf[section_gdf['T'] == T][self.section_map_field].values
            levels_sections[T] = codes_in
            mask = np.isin(decode_map, codes_in)
            flood_map_level.values *= mask
            mosaic_flood_map += flood_map_level
            del flood_map_level, mask  # Free memory

        if first_map:
            mosaic_flood_map, lat_mosaic, lon_mosaic = self.create_empty_mosaic()

        return mosaic_flood_map.astype(np.float32), lat_mosaic, lon_mosaic, levels_sections

    def assign_return_periods(self, section_gdf: gpd.GeoDataFrame, rp_raster: 'xr.DataArray') -> gpd.GeoDataFrame:
        """
        Assign return periods to sections.

        :param section_gdf: GeoDataFrame of the sections.
        :param rp_raster: Raster of return periods.
        :return: Updated GeoDataFrame with return periods.
        """
        def get_raster_value(point):
            return rp_raster.sel(x=point.x, y=point.y, method="nearest").values

        section_gdf['T'] = section_gdf.geometry.apply(get_raster_value)
        return section_gdf

    def import_flood_map_level(self, T: int) -> 'xr.DataArray':
        """
        Import flood map level for a given return period.

        :param T: Return period.
        :return: Flood map level as an xarray DataArray.
        """
        flood_map_path = self.flood_maps_template.format(return_period=int(T))
        flood_map_level = IOHandler.read_raster(flood_map_path, memory_map=True).astype(np.int16)
        return flood_map_level

    def initialize_mosaic(self, flood_map_level: 'xr.DataArray') -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Initialize the mosaic flood map.

        :param flood_map_level: Flood map level as an xarray DataArray.
        :return: Tuple of mosaic flood map, latitude mosaic, longitude mosaic, and decode map.
        """
        mosaic_flood_map = np.zeros_like(flood_map_level.values, dtype=np.int32)
        lat_mosaic = flood_map_level.y.values
        lon_mosaic = flood_map_level.x.values
        decode_map = IOHandler.read_raster(self.decode_map, memory_map=True).astype(np.int32).values
        return mosaic_flood_map, lat_mosaic, lon_mosaic, decode_map

    def create_empty_mosaic(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create an empty mosaic flood map.

        :return: Tuple of empty mosaic flood map, latitude mosaic, and longitude mosaic.
        """
        flood_map_level = IOHandler.read_raster(self.flood_maps_template.format(return_period=int(np.min(self.return_periods))), memory_map=True)
        mosaic_flood_map = np.zeros_like(flood_map_level.values, dtype=np.float32)
        lat_mosaic = flood_map_level.y.values
        lon_mosaic = flood_map_level.x.values
        return mosaic_flood_map, lat_mosaic, lon_mosaic

    def run(self, date_now: dt.datetime, rp_file: str) -> tuple[str, dict]:
        """
        Run the flood hazard mapping.

        :param date_now: Current date.
        :param rp_file: Path to the return period file.
        :return: Tuple of flood map path and levels sections.
        """
        rp_raster = IOHandler.read_raster(rp_file, memory_map=True)
        mosaic_flood_map, lat_mosaic, lon_mosaic, levels_sections = self.create_flood_map(rp_raster)
        flood_map = format_path_with_time(os.path.join(self.outcome_folder, self.outcome_filename), date_now)
        IOHandler.create_directories([os.path.dirname(flood_map)])
        IOHandler.write_tif(mosaic_flood_map, lon_mosaic, lat_mosaic, flood_map, dtype="int16")
        return flood_map, levels_sections