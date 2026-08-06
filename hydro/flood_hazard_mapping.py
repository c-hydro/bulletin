import numpy as np
import os
import pandas as pd
import logging
import geopandas as gpd
import datetime as dt
import warnings
from copy import deepcopy
import xarray as xr
import rioxarray as rx
from typing import Optional

from common.grids_handler import GridsHandler
from common.io_handler import IOHandler, format_path_with_time

class FloodHazardMerge:
    def __init__(self, section_map: str or None, section_map_field: str or None, return_periods: list[int], flood_maps_template: str, decode_map: str, outcome_folder: str, outcome_filename: str, skip_empty_maps: bool = False):
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
        self.return_periods = return_periods
        self.flood_maps_template = flood_maps_template
        self.decode_map = decode_map
        self.outcome_folder = outcome_folder
        self.outcome_filename = outcome_filename
        self.skip_empty_maps = skip_empty_maps

        if section_map_field is None:
            self.section_map_field = 'section'
        else:
            self.section_map_field = section_map_field

    def create_flood_map(self,
                         rp_raster: Optional['xr.DataArray'] = None,
                         section_T_df: Optional['pd.DataFrame'] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Create a flood map.

        :param rp_raster: Raster of return periods.
        :param section_T_df: Optional DataFrame with section–T mapping.
        :return: Tuple of mosaic flood map, latitude mosaic, longitude mosaic, and levels sections.
        """
        if section_T_df is None:
            # Original behaviour: read section map and assign T from raster
            section_gdf = gpd.read_file(self.section_map)
            logging.info("Assign the return period to the sections (it might take a while)")
            section_gdf = self.assign_return_periods(section_gdf, rp_raster)
        else:
            # New behaviour: use provided table section–T
            section_gdf = section_T_df.copy()
            del section_T_df
            logging.info("Using provided section–T table, skipping raster assignment")

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

    def run(self, date_now: dt.datetime, rp_file: str, section_T_df: Optional['pd.DataFrame'] = None) -> tuple[str, dict]:
        """
        Run the flood hazard mapping.

        :param date_now: Current date.
        :param rp_file: Path to the return period file (may be empty if section_DF is already provided).
        :param section_T_df: Optional DataFrame with section–T mapping.
        :return: Tuple of flood map path and levels sections.
        """
        if section_T_df is None:
            rp_raster = IOHandler.read_raster(rp_file, memory_map=True)
            mosaic_flood_map, lat_mosaic, lon_mosaic, levels_sections = self.create_flood_map(rp_raster=rp_raster)
        else:
            mosaic_flood_map, lat_mosaic, lon_mosaic, levels_sections = self.create_flood_map(section_T_df=section_T_df)
        flood_map = format_path_with_time(os.path.join(self.outcome_folder, self.outcome_filename), date_now)
        IOHandler.create_directories([os.path.dirname(flood_map)])
        if self.skip_empty_maps and np.nanmax(mosaic_flood_map) == 0:
            logging.info("Skipping writing empty flood map")
        else:
            IOHandler.write_tif(mosaic_flood_map, lon_mosaic, lat_mosaic, flood_map, dtype="int16")
        return flood_map, levels_sections


def convert_hazard_classes(alert_map: xr.DataArray, conversion_table: dict) -> xr.DataArray:
    """
    Convert raw threshold classes to operational hazard classes.

    This utility is opt-in and does not alter the standard flood-map workflow.
    """
    out = alert_map.copy()
    converted = np.where(out.values == 0, 0, 1).astype(np.int16)
    for class_out, classes_in in conversion_table.items():
        converted[np.isin(out.values, classes_in)] = int(class_out)
    out.values = converted
    return out


def classify_admin_hazard(
    admin_gdf: gpd.GeoDataFrame,
    alert_map: xr.DataArray,
    min_warning_pixel: int,
    output_file: str,
    output_column: str = "flood_level",
) -> gpd.GeoDataFrame:
    """
    Assign the maximum spatially supported hazard class to each admin feature.

    A class is retained only when at least ``min_warning_pixel`` cells inside
    the feature reach that class. It is invoked explicitly by workflows that
    need an administrative hazard layer.
    """
    out_gdf = admin_gdf.copy()
    out_gdf[output_column] = -9999.0
    shapes = [(shape, position) for position, shape in enumerate(out_gdf.geometry)]
    states = GridsHandler.rasterize_shapes(
        shapes,
        {"lon": alert_map.lon.values, "lat": alert_map.lat.values},
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for position, (index, _) in enumerate(out_gdf.iterrows()):
            admin_alert = alert_map.where(states == position)
            val_max = np.nanmax(admin_alert)
            while val_max > 1:
                total_over = np.count_nonzero(admin_alert >= val_max)
                if total_over >= min_warning_pixel:
                    break
                val_max -= 1
            out_gdf.at[index, output_column] = val_max

    IOHandler.create_directories([os.path.dirname(output_file)])
    out_gdf.to_file(output_file)
    return out_gdf


class FloodHazardOverlayMerge:
    """
    Build flood mosaics by overlaying threshold classes, AOI decode maps, and
    precomputed flood maps.

    This is a parallel, opt-in path. ``FloodHazardMerge`` remains unchanged and
    continues to serve the existing section/return-period workflows.
    """

    def __init__(self, overlay_settings: dict, temp_folder: str):
        self.overlay_settings = overlay_settings
        self.temp_folder = temp_folder

    @staticmethod
    def _intersect_aoi(
        mosaic_flood_map,
        flood_map_file: str,
        aoi_map: xr.DataArray,
        level: int,
        codes_in: pd.DataFrame,
    ) -> xr.DataArray:
        flood_map_level = (
            rx.open_rasterio(flood_map_file, cache=False)
            .rio.clip_box(*aoi_map.rio.bounds())
            .squeeze()
        )
        flood_map_level = flood_map_level.reindex_like(aoi_map, method="nearest")
        flood_map_level.values = np.where(
            (flood_map_level.values > 99999)
            | (flood_map_level.values <= 0)
            | np.isnan(flood_map_level.values),
            0,
            1,
        )

        if mosaic_flood_map is None:
            mosaic_flood_map = (flood_map_level.copy() * 0).astype(np.int16)

        aoi_in = deepcopy(aoi_map)
        aoi_in.values = np.where(
            np.isin(aoi_map.values, codes_in["hydro"].values),
            1,
            0,
        )
        return (
            mosaic_flood_map
            + flood_map_level.astype(np.int16) * level * aoi_in.astype(np.int16)
        )

    @staticmethod
    def _split_large_aoi(aoi_map: xr.DataArray) -> dict[int, xr.DataArray]:
        cells = aoi_map.sizes["x"] * aoi_map.sizes["y"]
        x_half = int(aoi_map.sizes["x"] / 2)
        y_half = int(aoi_map.sizes["y"] / 2)

        if cells > 1000000000:
            return {
                0: aoi_map.isel(x=slice(0, x_half), y=slice(0, y_half)),
                1: aoi_map.isel(x=slice(x_half, None), y=slice(0, y_half)),
                2: aoi_map.isel(x=slice(0, x_half), y=slice(y_half, None)),
                3: aoi_map.isel(x=slice(x_half, None), y=slice(y_half, None)),
            }
        if cells > 800000000:
            return {
                0: aoi_map.isel(x=slice(0, x_half), y=slice(0, y_half)),
                1: aoi_map.isel(x=slice(x_half, None), y=slice(0, y_half)),
                2: aoi_map.isel(x=slice(0, x_half), y=slice(y_half, None)),
                3: aoi_map.isel(x=slice(x_half, None), y=slice(y_half, None)),
            }
        if cells > 600000000:
            return {
                0: aoi_map.isel(x=slice(0, x_half), y=slice(0, None)),
                1: aoi_map.isel(x=slice(x_half, None), y=slice(0, None)),
            }
        return {0: aoi_map}

    @staticmethod
    def _merge_rasters(src_glob: str, out_file: str, dtype: str) -> None:
        import glob
        import rasterio
        from rasterio.merge import merge

        out_file_abs = os.path.abspath(out_file)
        matches = [
            path
            for path in sorted(glob.glob(src_glob))
            if os.path.abspath(path) != out_file_abs
        ]
        if not matches:
            logging.warning(f"No rasters found to merge: {src_glob}")
            return

        dtype_name = dtype.lower()
        src_files = [rasterio.open(path) for path in matches]
        try:
            mosaic, transform = merge(src_files, nodata=0)
            profile = src_files[0].profile.copy()
            profile.update(
                driver="GTiff",
                height=mosaic.shape[1],
                width=mosaic.shape[2],
                transform=transform,
                count=mosaic.shape[0],
                dtype=dtype_name,
                nodata=0,
                compress="DEFLATE",
                BIGTIFF="YES",
            )
            IOHandler.create_directories([os.path.dirname(out_file)])
            with rasterio.open(out_file, "w", **profile) as destination:
                destination.write(mosaic.astype(dtype_name))
        finally:
            for src_file in src_files:
                src_file.close()

    def run(self, alert_levels: xr.DataArray) -> tuple[str, str]:
        """Create the classified and hazard-weighted flood-map mosaics."""
        logging.info("Creating flood-map mosaics from AOI overlays")
        IOHandler.create_directories([self.temp_folder])

        decode_map = IOHandler.read_raster(
            self.overlay_settings["decode_map"],
            memory_map=True,
        )
        decode_map.values = np.where(decode_map.values < 0, -1, decode_map.values)
        alert_reindexed = alert_levels.reindex(
            {"lon": decode_map.x.values, "lat": decode_map.y.values},
            method="nearest",
        )

        decode_cfg = self.overlay_settings["aoi"]["decode_table"]
        conversion_table = pd.read_csv(
            decode_cfg["filename"],
            sep=",",
            usecols=[
                decode_cfg["col_flood"],
                decode_cfg["col_hydro"],
                decode_cfg["col_domain"],
            ],
            names=["flood", "hydro", "domain"],
            header=0,
        )
        domains = self.overlay_settings["aoi"]["domains"]
        conversion_table = conversion_table[
            conversion_table["domain"].isin(np.unique(domains))
        ]
        valid_hydro_codes = conversion_table["hydro"].values

        codes_per_level = {}
        unique_levels = np.unique(alert_reindexed.values)
        for level, _ in enumerate(
            self.overlay_settings["flood_maps"]["associated_rp"],
            start=2,
        ):
            if level in unique_levels:
                level_codes = decode_map.values[alert_reindexed.values == level]
                codes_per_level[level] = level_codes[
                    (level_codes > -1) & np.isin(level_codes, valid_hydro_codes)
                ]
            else:
                codes_per_level[level] = np.array([], dtype=decode_map.values.dtype)

        domain_map_template = self.overlay_settings["aoi"]["domain_map"]
        for aoi in domains:
            logging.info(f"Processing AOI domain {aoi}")
            aoi_map_file = domain_map_template.format(aoi=aoi, domain=aoi)
            aoi_map = IOHandler.read_raster(aoi_map_file, memory_map=True)
            aoi_map.rio.write_nodata(-9999, inplace=True)
            aoi_map.values = np.where(
                (aoi_map.values < 0) | np.isnan(aoi_map.values),
                0,
                aoi_map.values,
            )
            aoi_maps = self._split_large_aoi(aoi_map)

            for group, aoi_part in aoi_maps.items():
                logging.info(
                    f"Processing AOI domain {aoi}, part {group + 1} of {len(aoi_maps)}"
                )
                mosaic_flood_map = None
                for level, associated_rp in enumerate(
                    self.overlay_settings["flood_maps"]["associated_rp"],
                    start=2,
                ):
                    codes_in = conversion_table.loc[
                        conversion_table["hydro"].isin(codes_per_level[level])
                        & (conversion_table["domain"] == aoi)
                    ]
                    if level == 2 or not codes_in.empty:
                        flood_map_file = self.overlay_settings["flood_maps"][
                            "file_name"
                        ].format(
                            return_period=str(associated_rp),
                            aoi=aoi,
                            domain=aoi,
                        )
                        mosaic_flood_map = self._intersect_aoi(
                            mosaic_flood_map,
                            flood_map_file,
                            aoi_part,
                            level,
                            codes_in,
                        )
                    else:
                        logging.info(
                            f"No flood-map codes in domain {aoi} for level {level}"
                        )

                if mosaic_flood_map is None:
                    continue

                mosaic_flood_map.rio.write_crs("epsg:4326", inplace=True)
                mosaic_flood_map.rio.write_nodata(0, inplace=True)
                weighted_map = mosaic_flood_map.copy()
                weighted_map.values = np.zeros_like(
                    mosaic_flood_map.values,
                    dtype=np.int16,
                )
                for level, level_weight in enumerate(
                    self.overlay_settings["weight_hazard_levels"],
                    start=2,
                ):
                    weighted_map.values = np.where(
                        mosaic_flood_map.values == level,
                        int(level_weight * 100),
                        weighted_map.values,
                    )

                if np.nanmax(mosaic_flood_map.values) > 0:
                    flood_out = os.path.join(
                        self.temp_folder,
                        f"flood_map_{aoi}_{group}.tif",
                    )
                    weight_out = os.path.join(
                        self.temp_folder,
                        f"weight_map_{aoi}_{group}.tif",
                    )
                    mosaic_flood_map.astype(np.int16).rio.to_raster(
                        flood_out,
                        compress="DEFLATE",
                        dtype="int16",
                    )
                    weighted_map.astype(np.int16).rio.to_raster(
                        weight_out,
                        compress="DEFLATE",
                        dtype="int16",
                    )

        flood_map = os.path.join(self.temp_folder, "flood_map_merged.tif")
        weight_map = os.path.join(self.temp_folder, "weight_map_merged.tif")
        self._merge_rasters(
            os.path.join(self.temp_folder, "flood_map_*.tif"),
            flood_map,
            "Int16",
        )
        self._merge_rasters(
            os.path.join(self.temp_folder, "weight_map_*.tif"),
            weight_map,
            "Int16",
        )
        return flood_map, weight_map


def write_binary_flood_map(input_file: str, output_file: str) -> None:
    """Save a binary flood-presence raster from a classified flood mosaic."""
    if not os.path.isfile(input_file):
        logging.warning(
            f"Flood-map mosaic not found, binary map will not be written: {input_file}"
        )
        return

    flood_map = rx.open_rasterio(input_file, cache=False).squeeze()
    flood_map.values = np.where(flood_map.values > 0, 1, 0).astype(np.uint8)
    flood_map.rio.write_nodata(0, inplace=True)
    IOHandler.create_directories([os.path.dirname(output_file)])
    flood_map.rio.to_raster(output_file, compress="DEFLATE", dtype="uint8")

