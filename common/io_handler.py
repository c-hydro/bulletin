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
    def save_impact_shapefiles(exposed_element: str, folder_name: str, file_name: str, domain_shape: gpd.GeoDataFrame, impacts_table: pd.DataFrame, hazard: str) -> None:
        """
        Save impact data to shapefiles.

        :param exposed_element: Exposed element name.
        :param folder_name: Folder to save the shapefiles.
        :param file_name: Name of the shapefile.
        :param domain_shape: GeoDataFrame of the domain shape.
        :param impacts_table: DataFrame of impacts.
        :param hazard: Hazard name.
        """
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
            file = os.path.join(out_hmc_path, f"hmc.output-grid.{time_now.strftime('%Y%m%d%H%M')}.nc")
            if os.path.isfile(file + ".gz"):
                logging.debug(f"Unzipping file {file}.gz")
                gunzip_file(file + ".gz", file)
            if os.path.isfile(file):
                logging.debug(f"Reading HMC result file {file}")
                file_now = xr.open_dataset(file)
                results.append(file_now["Discharge"].values)
                if lat is None:
                    lon = file_now['Longitude'].values[0, :]
                    lat = file_now['Latitude'].values[:, 0]
                if mask is None:
                    mask = np.where(file_now["SM"].values < 0, 0, 1)
            else:
                logging.error(f"Output file {file} not found!")
                raise FileNotFoundError(f"Output file {file} not found!")
            os.remove(file)

        # Check and flip y-axis if necessary
        if lat[0] < lat[-1]:
            logging.debug("Flipping y-axis for extracted HMC results")
            results = [np.flipud(result) for result in results]
            lat = lat[::-1]

        return results, mask, lon, lat

    @staticmethod
    def clear_ancillary_folder(folder_path: str, clear_flag: bool) -> None:
        """
        Clear the ancillary folder if the clear flag is set.

        :param folder_path: Path to the ancillary folder.
        :param clear_flag: Boolean flag to clear the folder.
        """
        if clear_flag:
            logging.info("Clearing ancillary folder")
            shutil.rmtree(folder_path)


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
    Replace keys in a string with their corresponding values.

    :param value: String with keys to replace.
    :param replacements: Dictionary of replacements.
    :return: String with replaced keys or the original value if it's not a string.
    """
    if isinstance(value, str):
        for key, replacement in replacements.items():
            value = value.replace(f"{{{key}}}", replacement)
    return value

def update_file_paths(file_paths, replacements: dict[str, str]):
    """
    Update file paths with replacements.

    :param file_paths: File paths to update.
    :param replacements: Dictionary of replacements.
    :return: Updated file paths.
    """
    if isinstance(file_paths, dict):
        updated_files = {}
        for key, value in file_paths.items():
            if isinstance(value, dict):
                updated_files[key] = update_file_paths(value, replacements)
            elif isinstance(value, list):
                updated_files[key] = [replace_keys(f, replacements) for f in value]
            else:
                updated_files[key] = replace_keys(value, replacements)
        return updated_files
    elif isinstance(file_paths, list):
        return [replace_keys(f, replacements) for f in file_paths]
    else:
        return replace_keys(file_paths, replacements)

def gunzip_file(gz_file_path: str, output_file_path: str) -> None:
    """
    Unzip a gzip file.

    :param gz_file_path: Path to the gzip file.
    :param output_file_path: Path to the output file.
    """
    logging.debug(f"Gunzipping file {gz_file_path} to {output_file_path}")
    with gzip.open(gz_file_path, 'rb') as f_in:
        with open(output_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)