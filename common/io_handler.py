import datetime as dt
import gzip
import logging
import os
import shutil

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr


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
            raster = rxr.open_rasterio(file_path, chunks={"band": 1}, cache=False).squeeze()
        else:
            raster = rxr.open_rasterio(file_path, cache=False).squeeze()
        if raster.y[0] < raster.y[-1]:
            logging.debug(f"Flipping y-axis for {file_path}")
            raster = raster.reindex(y=raster.y[::-1])
        return raster

    @staticmethod
    def write_raster(
        raster: xr.DataArray,
        out_path: str,
        nodata: float = None,
        compress: str = "DEFLATE",
    ) -> None:
        """
        Write an xarray DataArray with rioxarray spatial metadata to GeoTIFF.

        :param raster: DataArray with .rio accessor.
        :param out_path: Output GeoTIFF path.
        :param nodata: Optional nodata value.
        :param compress: Compression method.
        """
        logging.info(f"Writing raster to {out_path}")
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        if nodata is not None:
            raster = raster.rio.write_nodata(nodata, inplace=False)
        raster.rio.to_raster(out_path, compress=compress)

    @staticmethod
    def write_tif(
        data: np.ndarray,
        lon: np.ndarray,
        lat: np.ndarray,
        out_filename: str,
        crs: str = "epsg:4326",
        nodata: int = -9999,
        dtype: str = "float32",
    ) -> None:
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
        out_dir = os.path.dirname(out_filename)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        out_ds = xr.DataArray(data, dims=["y", "x"], coords={"y": lat, "x": lon})
        out_ds.values = np.where(out_ds.values == nodata, nodata, out_ds.values.astype(dtype))
        out_ds = out_ds.rio.write_crs(crs, inplace=True).rio.write_nodata(nodata, inplace=True)
        out_ds.rio.to_raster(
            out_filename,
            driver="GTiff",
            crs=crs,
            height=len(lat),
            width=len(lon),
            dtype=out_ds.dtype,
            compress="DEFLATE",
            nodata=nodata,
        )

    @staticmethod
    def save_impact_shapefiles(
        exposed_element: str,
        folder_name: str,
        file_name: str,
        domain_shape: gpd.GeoDataFrame,
        impacts_table: pd.DataFrame,
        hazard: str,
        rounding: bool = False,
    ) -> None:
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

        short_hazard = hazard[:6]
        domain_shape[short_hazard + "_tot"] = impacts_table[hazard + "_tot_" + exposed_element]

        for col in impacts_table.columns:
            if col.startswith(hazard + "_tot_" + exposed_element + "_"):
                sub_category = col.replace(hazard + "_tot_" + exposed_element + "_", "")
                short_sub_category = sub_category[:6]
                domain_shape["tot_" + short_sub_category] = impacts_table[col]

        domain_shape.to_file(output_file)

    @staticmethod
    def extract_hmc_results(
        out_hmc_path: str,
        date_start: dt.datetime,
        date_end: dt.datetime,
        mask: np.ndarray = None,
        lat: np.ndarray = None,
        lon: np.ndarray = None,
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract HMC results for a given date range.

        Supports both:
        - generic files: YYYYMMDDHHMM_hmc.out.nc with Qout/lat/lon/mask
        - Continuum/HMC files: hmc.output-grid.YYYYMMDDHHMM.nc(.gz) with
          Discharge/Longitude/Latitude/SM
        """
        logging.info(f"Extracting HMC results from {date_start} to {date_end}")
        results = []
        for time_now in pd.date_range(date_start, date_end, freq="h"):
            timestamp = time_now.strftime("%Y%m%d%H%M")
            generic_file = os.path.join(out_hmc_path, f"{timestamp}_hmc.out.nc")
            continuum_file = os.path.join(out_hmc_path, f"hmc.output-grid.{timestamp}.nc")

            if os.path.exists(generic_file):
                with xr.open_dataset(generic_file) as ds:
                    if lat is None or lon is None:
                        lat = ds["lat"].values
                        lon = ds["lon"].values
                    if mask is None and "mask" in ds:
                        mask = ds["mask"].values
                    result = ds["Qout"].values
                    if mask is not None:
                        result = np.where(mask == 1, result, np.nan)
                    results.append(result)
                continue

            created_tmp_file = False
            if os.path.isfile(continuum_file + ".gz"):
                logging.debug(f"Unzipping file {continuum_file}.gz")
                gunzip_file(continuum_file + ".gz", continuum_file)
                created_tmp_file = True

            if os.path.isfile(continuum_file):
                with xr.open_dataset(continuum_file) as ds:
                    results.append(ds["Discharge"].values)
                    if lat is None or lon is None:
                        lon = ds["Longitude"].values[0, :]
                        lat = ds["Latitude"].values[:, 0]
                    if mask is None and "SM" in ds:
                        mask = np.where(ds["SM"].values < 0, 0, 1)
                if created_tmp_file and os.path.isfile(continuum_file):
                    os.remove(continuum_file)
                continue

            logging.warning(f"No HMC output found for {timestamp} in {out_hmc_path}")

        if len(results) == 0:
            raise FileNotFoundError(f"No HMC output found in {out_hmc_path}")

        if lat is not None and len(lat) > 1 and lat[0] < lat[-1]:
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
        if clear_flag and os.path.isdir(folder_path):
            logging.info(f"Clearing ancillary folder: {folder_path}")
            shutil.rmtree(folder_path)

    @staticmethod
    def read_fanfar_file(file: str) -> pd.Series:
        """
        Read a FANFAR hydrograph text file.

        Expected metadata includes DateStart (YYYYMMDDHHMM) and Temp.Resolution
        in minutes. The discharge values are read from the first numeric data row.
        """
        logging.debug(f"Reading FANFAR file {file}")
        with open(file) as f:
            lines = f.readlines()

        meta: dict[str, str] = {}
        values = None
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if "=" in stripped:
                key, value = stripped.split("=", 1)
                meta[key] = value
                continue
            parsed = np.fromstring(stripped, sep=" ")
            if parsed.size > 0 and values is None:
                values = parsed

        if values is None:
            raise ValueError(f"No data values found in FANFAR file: {file}")

        start = dt.datetime.strptime(meta["DateStart"], "%Y%m%d%H%M")
        step_min = int(meta["Temp.Resolution"])
        time_index = [start + dt.timedelta(minutes=step_min * i) for i in range(len(values))]
        return pd.Series(values, index=pd.DatetimeIndex(time_index), name="value")


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

    Supports the placeholder style {key}. As a fallback, also supports raw key
    replacement for older utilities that may pass unbraced tokens.
    """
    if isinstance(value, str):
        for key, replacement in replacements.items():
            value = value.replace(f"{{{key}}}", str(replacement))
            value = value.replace(key, str(replacement))
    return value


def update_file_paths(
    file_paths,
    replacements: dict[str, str] = None,
    keys_to_update: list[str] = None,
    logger: logging.Logger = None,
):
    """
    Update file paths with replacements.

    Primary usage:
        update_file_paths(file_paths, {"element": "population"})

    Backward-compatible usage:
        update_file_paths(settings, time_now, keys_to_update, logger)
    where the second argument is a datetime and selected settings keys are
    formatted with strftime.
    """
    if isinstance(replacements, dt.datetime):
        settings = file_paths
        time_now = replacements
        updated_keys = keys_to_update or []
        for key in updated_keys:
            if key in settings:
                if logger is not None:
                    logger.debug(f"Updating file paths for {key}")
                settings[key] = format_path_with_time(settings[key], time_now)
        return settings

    replacements = replacements or {}
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
    if isinstance(file_paths, list):
        return [replace_keys(f, replacements) for f in file_paths]
    return replace_keys(file_paths, replacements)


def gunzip_file(gz_file_path: str, output_file_path: str) -> None:
    """
    Unzip a gzip file.

    :param gz_file_path: Path to the gzip file.
    :param output_file_path: Path to the output file.
    """
    logging.debug(f"Gunzipping file {gz_file_path} to {output_file_path}")
    with gzip.open(gz_file_path, "rb") as f_in:
        with open(output_file_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
