import xarray as xr
import os
import pandas as pd
import numpy as np
import datetime as dt
import gzip
import shutil
import logging
import rioxarray as rxr
import geopandas as gpd

class IOHandler:
    @staticmethod
    def create_directories(paths: list[str]) -> None:
        """
        Create directories if they do not exist.

        :param paths: List of directory paths to create.
        """
        for path in paths:
            logging.debug(f"Creating directory {path}")
            os.makedirs(path, exist_ok=True)

    @staticmethod
    def read_vector(file_path: str) -> gpd.GeoDataFrame:
        """
        Read a vector file (shapefile, GeoJSON, GeoPackage...) using GeoPandas.

        :param file_path: Path to the vector file.
        :return: GeoDataFrame.
        """
        logging.debug(f"Reading vector from {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)
        gdf = gpd.read_file(file_path)
        if gdf.empty:
            raise ValueError(f"Vector file is empty: {file_path}")
        return gdf

    @staticmethod
    def open_netcdf_dataset(nc_path: str) -> xr.Dataset:
        """
        Open a NetCDF dataset.

        :param nc_path: Path to the NetCDF file.
        :return: xarray Dataset.
        """
        logging.debug(f"Opening NetCDF dataset: {nc_path}")
        if not os.path.exists(nc_path):
            raise FileNotFoundError(nc_path)
        return xr.open_dataset(nc_path)






    @staticmethod
    def read_raster(file_path: str, memory_map: bool = False) -> xr.DataArray:
        """
        Read a raster file.

        :param file_path: Path to the raster file.
        :param memory_map: Boolean flag to use memory mapping.
        :return: Raster data as an xarray DataArray.
        """
        logging.debug(f"Reading raster from {file_path}")
        if memory_map:
            raster = rxr.open_rasterio(file_path, chunks={'band': 1}, cache=False).squeeze()
        else:
            raster = rxr.open_rasterio(file_path, cache=False).squeeze()
        if raster.y[0] < raster.y[-1]:
            logging.debug(f"Flipping y-axis for {file_path}")
            raster = raster.reindex(y=raster.y[::-1])
        return raster

    @staticmethod
    def write_raster(raster: xr.DataArray, out_path: str, nodata: float = None, compress: str = "DEFLATE") -> None:
        """
        Write an xarray DataArray (with rioxarray spatial metadata) to GeoTIFF.

        :param raster: DataArray with .rio accessor
        :param out_path: Output GeoTIFF path
        :param nodata: Optional nodata value
        :param compress: Compression (default DEFLATE)
        """
        logging.info(f"Writing raster to {out_path}")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if nodata is not None:
            raster = raster.rio.write_nodata(nodata, inplace=False)
        raster.rio.to_raster(out_path, compress=compress)

    @staticmethod
    def write_tif(data: np.ndarray, lon: np.ndarray, lat: np.ndarray, out_filename: str, crs: str = 'epsg:4326', nodata: int = -9999, dtype: str = 'float32') -> None:
        """
        Write data to a GeoTIFF file.

        :param data: Data to write.
        :param lon: Longitude values.
        :param lat: Latitude values.
        :param out_filename: Output file name.
        :param crs: Coordinate reference system.
        :param nodata: No data value.
        :param dtype: Data type.
        """
        logging.info(f"Writing TIF to {out_filename}")
        out_ds = xr.DataArray(data, dims=["y", "x"], coords={"y": lat, "x": lon})
        out_ds.values = np.where(out_ds.values == nodata, nodata, out_ds.values.astype(dtype))
        out_ds = out_ds.rio.write_crs(crs, inplace=True).rio.write_nodata(nodata, inplace=True)
        out_ds.rio.to_raster(out_filename, driver="GTiff", crs=crs, height=len(lat), width=len(lon), dtype=out_ds.dtype,
                         compress="DEFLATE", nodata=nodata)

    @staticmethod
    def save_impact_shapefiles(exposed_element: str, folder_name: str, file_name: str, domain_shape: gpd.GeoDataFrame, impacts_table: pd.DataFrame, hazard: str, rounding: bool = False) -> None:
        """
        Save impact data to shapefiles.

        :param exposed_element: Exposed element name.
        :param folder_name: Folder to save the shapefiles.
        :param file_name: Name of the shapefile.
        :param domain_shape: GeoDataFrame of the domain shape.
        :param impacts_table: DataFrame of impacts.
        :param hazard: Hazard name.
        :param rounding: The number of decimal places to round to.
        """
        if rounding:
            impacts_table = impacts_table.round(0).astype(int)

        IOHandler.create_directories([folder_name])
        output_file = os.path.join(folder_name, file_name)

        if domain_shape.crs is None:
            domain_shape.crs = "epsg:4326"

        # Shorten the hazard name to ensure field names do not exceed 10 characters
        short_hazard = hazard[:6]

        # Save the total impacts
        domain_shape[short_hazard + "_tot"] = impacts_table[hazard + "_tot_" + exposed_element]

        # Save sub-categories if they exist
        for col in impacts_table.columns:
            if col.startswith(hazard + "_tot_" + exposed_element + "_"):
                sub_category = col.replace(hazard + "_tot_" + exposed_element + "_", "")
                short_sub_category = sub_category[:6]  # Shorten sub_category if necessary
                domain_shape["tot_" + short_sub_category] = impacts_table[col]

        domain_shape.to_file(output_file)

    @staticmethod
    def extract_hmc_results(out_hmc_path: str, date_start: dt.datetime, date_end: dt.datetime, mask: np.ndarray = None, lat: np.ndarray = None, lon: np.ndarray = None) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract HMC results for a given date range.

        :param out_hmc_path: Path to HMC output files.
        :param date_start: Start date.
        :param date_end: End date.
        :param mask: Optional mask array.
        :param lat: Optional latitude values.
        :param lon: Optional longitude values.
        :return: Tuple of results, mask, longitude, and latitude.
        """
        logging.info(f"Extracting HMC results from {date_start} to {date_end}")
        results = []
        for time_now in pd.date_range(date_start, date_end, freq="h"):
            file_path = os.path.join(out_hmc_path, time_now.strftime("%Y%m%d%H%M") + "_hmc.out.nc")
            if os.path.exists(file_path):
                ds = xr.open_dataset(file_path)
                if lat is None or lon is None:
                    lat = ds["lat"].values
                    lon = ds["lon"].values
                if mask is None:
                    mask = ds["mask"].values
                result = ds["Qout"].values
                if mask is not None:
                    result = np.where(mask == 1, result, np.nan)
                results.append(result)
            else:
                logging.warning(f"File {file_path} does not exist")
        return results, mask, lon, lat


def format_path_with_time(path_template: str, date_time: dt.datetime) -> str:
    """
    Format a path template with a given date and time.

    :param path_template: Path template string.
    :param date_time: Date and time to format the path with.
    :return: Formatted path string.
    """
    logging.debug(f"Formatting path with time: {date_time}")
    return date_time.strftime(path_template)


def replace_keys(value, replacements: dict[str, str]):
    """
    Replace keys in a string with values from a replacements dictionary.

    :param value: The string to perform replacements on.
    :param replacements: A dictionary with keys to replace and their replacement values.
    :return: The updated string with replacements applied.
    """
    if isinstance(value, str):
        for key, replacement in replacements.items():
            value = value.replace(key, replacement)
    return value


def update_file_paths(settings: dict, time_now: dt.datetime, keys_to_update: list[str], logger: logging.Logger) -> dict:
    """
    Update file paths in a settings dictionary by formatting path templates with a given date and time.

    :param settings: Settings dictionary.
    :param time_now: Current date and time.
    :param keys_to_update: List of keys to update in the settings dictionary.
    :param logger: Logger instance.
    :return: Updated settings dictionary.
    """
    for key in keys_to_update:
        if key in settings:
            logger.debug(f"Updating file paths for {key}")
            settings[key] = format_path_with_time(settings[key], time_now)
    return settings
