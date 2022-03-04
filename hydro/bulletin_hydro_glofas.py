"""
bulletin - hydro - GLOFAS
__date__ = '20211111'
__version__ = '1.0.0'
__author__ =
        'Andrea Libertino (andrea.libertino@cimafoundation.org',
__library__ = 'bulletin'
General command line:
### python fp_bulletin_hydro_glofas.py -settings_file "settings.json" -time "YYYY-MM-DD HH:MM"
Version(s):
20211111 (1.0.0) --> Beta release for Africa Continental Watch
"""
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Complete library
import cdsapi
import os
import numpy as np
import datetime as dt
import xarray as xr
import geopandas as gpd
from rasterio import features
from affine import Affine
import warnings
import logging
import json
from argparse import ArgumentParser
import pytz
import time
import subprocess
import random
import string

# -------------------------------------------------------------------------------------
# Script Main
def main():
    # -------------------------------------------------------------------------------------
    # Version and algorithm information
    alg_name = 'bulletin - Hydrological warning with GLOFAS '
    alg_version = '1.0.0'
    alg_release = '2021-11-11'
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Get algorithm settings
    alg_settings, alg_time = get_args()

    # Set algorithm settings
    data_settings = read_file_json(alg_settings)
    paths = {}
    paths["cdo"] = data_settings["algorithm"]["system"]["cdo_path"]
    paths["grib_copy"] = data_settings["algorithm"]["system"]["grib_copy_path"]

    # Check eccodes version
    try:
        output = subprocess.check_output(paths["grib_copy"] + "grib_copy -V", shell=True)
        eccodes_version = output.decode("utf-8").split(" ")[2].replace("\n", "")
        eccodes_version_major = float(output.decode("utf-8").split(" ")[2].replace("\n", "").split(".")[0])
        eccodes_version_minor = float(output.decode("utf-8").split(" ")[2].replace("\n", "").split(".")[1])
        if eccodes_version_major < 2:
            logging.error(" --> grib_copy version found is " + str(eccodes_version) + ". Eccodes > 2.20.0 is required")
            raise OSError
        if eccodes_version_major == 2 and eccodes_version_minor < 20:
            logging.error(" --> grib_copy version found is " + str(eccodes_version) + ". Eccodes > 2.20.0 is required")
            raise OSError
    except:
        logging.error(" --> grib_copy not found at the provided path! Check settings file!")
        raise OSError

    # Set algorithm logging
    os.makedirs(data_settings['data']['log']['folder'], exist_ok=True)
    set_logging(
        logger_file=os.path.join(data_settings['data']['log']['folder'], data_settings['data']['log']['filename']))
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Get time settings
    date_now = pytz.utc.localize(dt.datetime.strptime(alg_time, "%Y-%m-%d %H:%M"))
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Create directories
    output_path_empty = data_settings['data']['dynamic']['outcome']['folder']
    output_resume_path_empty = data_settings['data']['dynamic']['outcome']['folder_resume']
    ancillary_path_empty = data_settings['data']['dynamic']['ancillary']['folder']

    dict_empty = data_settings['algorithm']['template']
    dict_filled = dict_empty.copy()

    for key in dict_empty.keys():
        dict_filled[key] = date_now.strftime(dict_empty[key])

    paths["output"] = output_path_empty.format(**dict_filled)
    paths["output_resume"] = output_resume_path_empty.format(**dict_filled)
    paths["ancillary"] = ancillary_path_empty.format(**dict_filled)
    paths["ancillary_raw"] = os.path.join(paths["ancillary"],"raw","")

    os.makedirs(paths["output"], exist_ok=True)
    os.makedirs(paths["output_resume"], exist_ok=True)
    os.makedirs(paths["ancillary_raw"], exist_ok =True)
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Info algorithm
    logging.info(' ============================================================================ ')
    logging.info(' ==> START ... ')
    logging.info(' ')

    logging.info(" --> Time now : " + alg_time)

    # Time algorithm information
    start_time = time.time()

    try:
        debug_mode = data_settings["algorithm"]["flags"]["debug_mode"]
    except:
        debug_mode = False

    if debug_mode:
        logging.info(" --> Debug mode : ACTIVE")
    else:
        logging.info(" --> Debug mode : NOT ACTIVE")
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Download GLOFAS forecast
    logging.info(" --> Compute glofas forecast with date: " + date_now.strftime("%Y-%m-%d %H:%M"))
    nc_avg_name = download_from_cds(date_now, dict_filled, data_settings, paths)
    logging.info(" --> Compute glofas forecast with date: " + date_now.strftime("%Y-%m-%d %H:%M") + "...DONE")
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Classify discharge maps
    logging.info(" --> Calculate hydro warning levels...")

    logging.info(" ---> Classify discharge maps...")
    first_step = True
    alert_level_days = {}

    time_steps = data_settings["data"]["dynamic"]["glofas"]["time_steps"]

    for step in time_steps:
        logging.info(" ----> Analyse time step " + step + "...")
        dis_max = xr.open_dataset(nc_avg_name.format(perturbationNumber="*", step=step))["dis24"].squeeze()

        if first_step == True:
            area = xr.open_rasterio(data_settings['data']['static']['area']).reindex(
                {"x": dis_max.lon.values, "y": dis_max.lat.values}, method='nearest').squeeze()
            mask = np.where(area>=data_settings["data"]["dynamic"]["thresholds"]["area_km2"],1,0)
            alert_map = np.ones(area.shape)
            alert_max = np.ones(area.shape)
            first_step = False

        for val, rp in enumerate(data_settings["data"]["static"]["discharge_thresholds"]["return_periods"], start=2):
            th_map = xr.open_rasterio(os.path.join(data_settings["data"]["static"]["discharge_thresholds"]["folder"],
                                                   data_settings["data"]["static"]["discharge_thresholds"][
                                                       "file_name"]).format(domain=None, return_period=rp)).reindex(
                {"x": dis_max.lon.values, "y": dis_max.lat.values}, method='nearest').squeeze()
            th_map.values[th_map.values <= 0] = np.Inf
            alert_map = np.where((dis_max.values >= th_map.values) & (dis_max.values >= data_settings["data"]["dynamic"]["thresholds"]["discharge_min"]), val, alert_map)
        alert_level_days[step] = np.where(mask == 1, alert_map, 0)

        alert_max = np.maximum(alert_max,alert_level_days[step])

    logging.info(" ----> Save discharge maps...")
    out_map = xr.DataArray(alert_max, dims=["lat","lon"], coords={"lon": dis_max.lon.values, "lat": dis_max.lat.values})
    out_file_tif = os.path.join(paths["output"], data_settings['data']['dynamic']['outcome']['file_name']).format(
        **dict_filled)
    out_map.to_netcdf(os.path.join(paths["ancillary"], "temp.nc"))
    os.system(
        "gdal_translate " + os.path.join(paths["ancillary"], "temp.nc") + " -a_srs EPSG:4326 " + out_file_tif)
    logging.info(" ----> Save discharge maps...DONE")

    logging.info(" ---> Classify discharge maps...DONE")

    logging.info(" ---> Assign flood country warning levels...")

    # Assign flood country warning level
    shp_fo = data_settings["data"]["static"]["warning_regions"]
    shp_df = gpd.read_file(shp_fo)
    shp_df_hydro_model = shp_df.copy()
    shp_df_hydro_model["level_GLOFAS"] = -9999.0

    th_levels = xr.open_rasterio(out_file_tif).squeeze().rename({"x": "lon", "y": "lat"})

    shapes = [(shape, n) for n, shape in enumerate(shp_df.geometry)]
    ds = xr.Dataset(coords={'lon': th_levels['lon'].values, 'lat': th_levels['lat'].values})
    ds['states'] = rasterize(shapes, ds.coords)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        logging.info(" ----> Loop through the alert zones...")
        for index, row in shp_df_hydro_model.iterrows():
            val_max = np.nanmax(th_levels.where(ds['states'] == index))
            while val_max > 1:
                tot_over = np.count_nonzero(th_levels.where(ds['states'] == index) >= val_max)
                if tot_over >= data_settings["data"]["dynamic"]["thresholds"]["min_warning_pixel"]:
                    break
                else:
                    val_max = val_max - 1
            shp_df_hydro_model["level_GLOFAS"].at[index] = val_max
        logging.info(" ----> Loop through the alert zones...DONE")

    logging.info(" --> Classify flood country warning levels...DONE")
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Write outputs
    logging.info(" --> Write outputs and clean system...")
    logging.info(" ---> Write output shapefile...")

    out_file_shp = os.path.join(paths["output_resume"], data_settings['data']['dynamic']['outcome']['file_name_resume']).format(
        **dict_filled)
    shp_df_hydro_model.to_file(os.path.join(paths["output_resume"], out_file_shp))
    logging.info(" ---> Write outputs...DONE")

    if data_settings["algorithm"]["flags"]["clear_ancillary_data"] and not debug_mode:
        logging.info(" ---> Clean ancillary data...")
        os.system("rm -r " + paths["ancillary"])
        logging.info(" ---> Clean ancillary data...DONE")
    else:
        logging.info(" ---> Clean ancillary data deactivated or debug mode active...SKIP")
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Info algorithm
    time_elapsed = round(time.time() - start_time, 1)

    logging.info(' ')
    logging.info(' ==> ' + alg_name + ' (Version: ' + alg_version + ' Release_Date: ' + alg_release + ')')
    logging.info(' ==> TIME ELAPSED: ' + str(time_elapsed) + ' seconds')
    logging.info(' ==> ... END')
    logging.info(' ==> Bye, Bye')
    logging.info(' ============================================================================ ')
    # -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Method to read file json
def read_file_json(file_name):

    env_ws = {}
    for env_item, env_value in os.environ.items():
        env_ws[env_item] = env_value

    with open(file_name, "r") as file_handle:
        json_block = []
        for file_row in file_handle:

            for env_key, env_value in env_ws.items():
                env_tag = '$' + env_key
                if env_tag in file_row:
                    env_value = env_value.strip("'\\'")
                    file_row = file_row.replace(env_tag, env_value)
                    file_row = file_row.replace('//', '/')

            # Add the line to our JSON block
            json_block.append(file_row)

            # Check whether we closed our JSON block
            if file_row.startswith('}'):
                # Do something with the JSON dictionary
                json_dict = json.loads(''.join(json_block))
                # Start a new block
                json_block = []

    return json_dict
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Method to get script argument(s)
def get_args():
    parser_handle = ArgumentParser()
    parser_handle.add_argument('-settings_file', action="store", dest="alg_settings")
    parser_handle.add_argument('-time', action="store", dest="alg_time")
    parser_values = parser_handle.parse_args()

    if parser_values.alg_settings:
        alg_settings = parser_values.alg_settings
    else:
        alg_settings = 'configuration.json'

    if parser_values.alg_time:
        alg_time = parser_values.alg_time
    else:
        alg_time = None

    return alg_settings, alg_time
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Method to set logging information
def set_logging(logger_file='log.txt', logger_format=None):
    if logger_format is None:
        logger_format = '%(asctime)s %(name)-12s %(levelname)-8s ' \
                        '%(filename)s:[%(lineno)-6s - %(funcName)20s()] %(message)s'

    # Remove old logging file
    if os.path.exists(logger_file):
        os.remove(logger_file)

    # Set level of root debugger
    logging.root.setLevel(logging.INFO)

    # Open logging basic configuration
    logging.basicConfig(level=logging.INFO, format=logger_format, filename=logger_file, filemode='w')

    # Set logger handle
    logger_handle_1 = logging.FileHandler(logger_file, 'w')
    logger_handle_2 = logging.StreamHandler()
    # Set logger level
    logger_handle_1.setLevel(logging.INFO)
    logger_handle_2.setLevel(logging.INFO)
    # Set logger formatter
    logger_formatter = logging.Formatter(logger_format)
    logger_handle_1.setFormatter(logger_formatter)
    logger_handle_2.setFormatter(logger_formatter)
    # Add handle to logging
    logging.getLogger('').addHandler(logger_handle_1)
    logging.getLogger('').addHandler(logger_handle_2)
# -------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def transform_from_latlon(lat, lon):
        lat = np.asarray(lat)
        lon = np.asarray(lon)
        trans = Affine.translation(lon[0], lat[0])
        scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
        return trans * scale

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def rasterize(shapes, coords, fill=np.nan, **kwargs):
        """Rasterize a list of (geometry, fill_value) tuples onto the given
        xray coordinates. This only works for 1d latitude and longitude
        arrays.
        """
        transform = transform_from_latlon(coords['lat'], coords['lon'])
        out_shape = (len(coords['lat']), len(coords['lon']))
        raster = features.rasterize(shapes, out_shape=out_shape,
                                    fill=fill, transform=transform,
                                    dtype=float, **kwargs)
        return xr.DataArray(raster, coords=coords, dims=('lat', 'lon'))
# ----------------------------------------------------------------------------

def download_from_cds(date_now, dict_filled, data_settings, paths):
    logging.info(" ---> Download glofas forecast from CDS...")

    forecast_file = os.path.join(paths["ancillary_raw"], data_settings['data']['dynamic']['ancillary']['file_name']).format(**dict_filled)
    bounding_box = [data_settings["data"]["dynamic"]["bbox"][i] for i in data_settings["data"]["dynamic"]["bbox"]]
    time_steps = data_settings["data"]["dynamic"]["glofas"]["time_steps"]
    ens_members = np.arange(0, data_settings["data"]["dynamic"]["glofas"]["ens_max"], 1)

    digits = string.digits
    rand_str = ''.join(random.choice(digits) for i in range(3))

    c = cdsapi.Client()
    c.retrieve(
    'cems-glofas-forecast',
    {
         'system_version': 'operational',
         'variable': 'river_discharge_in_the_last_24_hours',
         'hydrological_model': 'lisflood',
         'format': 'grib',
         'product_type': ['control_forecast', 'ensemble_perturbed_forecasts'],
         'year': str(date_now.year),
         'month': str(date_now.month).zfill(2),
         'day': str(date_now.day).zfill(2),
         'leadtime_hour': time_steps,
         'area': bounding_box,
         'nocache':rand_str
    },
    forecast_file)
    logging.info(" ---> Download glofas forecast from CDS...DONE")

    logging.info(" ---> Convert GRIB fieldset to individual NC files...")
    # Split the GRIB fieldset into individual GRIB fields
    grib_field_template = os.path.join(paths["ancillary_raw"],'glofas_fc_[perturbationNumber]_[step].grb')
    os.system(paths["grib_copy"] + "grib_copy " + forecast_file + " " + grib_field_template)
    grib_field_template = grib_field_template.replace("[","{").replace("]","}")

    ancillary_avg_path = os.path.join(paths["ancillary"], "average")
    os.makedirs(ancillary_avg_path, exist_ok=True)
    nc_avg_name = os.path.join(ancillary_avg_path, "glofas_fc_avg_time_{step}.nc")

    for step in time_steps:
        logging.info(" ---> Compute time step " + step + "...")
        for ens in ens_members:
            ancillary_ens_path = os.path.join(paths["ancillary"], "ens_" + str(ens), "")
            os.makedirs(ancillary_ens_path, exist_ok=True)
            nc_ens_name = os.path.join(ancillary_ens_path,"glofas_fc_{perturbationNumber}_time_{step}.nc")
            os.system((os.path.join(paths["cdo"], "cdo") + " -s -f nc copy " + grib_field_template + " " + nc_ens_name).format(perturbationNumber= str(ens), step=step))
        logging.info(" ---> Average ensemble members...")
        nc_ens_name = os.path.join(paths["ancillary"], "*", "glofas_fc_{perturbationNumber}_time_{step}.nc")
        os.system((os.path.join(paths["cdo"], "cdo") + " -O -s ensmean " + nc_ens_name + " " + nc_avg_name).format(perturbationNumber="*", step=step))

    ancillary_avg_path = os.path.join(paths["ancillary"], "average")
    nc_avg_name = os.path.join(ancillary_avg_path, "glofas_fc_avg_time_{step}.nc")

    logging.info(" ---> Convert GRIB fieldset to individual NC files...DONE")
    return nc_avg_name

# ----------------------------------------------------------------------------
# Call script from external library
if __name__ == "__main__":
    main()
# ----------------------------------------------------------------------------
