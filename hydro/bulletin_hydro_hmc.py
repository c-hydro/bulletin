"""
bulletin - hydro - HMC
__date__ = '20211111'
__version__ = '1.0.0'
__author__ =
        'Andrea Libertino (andrea.libertino@cimafoundation.org',
__library__ = 'bulletin'
General command line:
### python fp_bulletin_hydro_glofas.py -settings_file "settings.json" -time "YYYY-MM-DD HH:MM"
Version(s):
20211111 (1.0.0) --> Beta release for FloodProofs Africa
"""
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Complete library
import os, logging, json
import pytz
from argparse import ArgumentParser
import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import time
from copy import deepcopy
import geopandas as gpd
from rasterio import features
from affine import Affine
import warnings
import rasterio as rio
import sys


# -------------------------------------------------------------------------------------
# Script Main
def main():
    # -------------------------------------------------------------------------------------
    # Version and algorithm information
    alg_name = 'bulletin - Hydrological warning with Continuum '
    alg_version = '1.0.0'
    alg_release = '2021-11-11'
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Get algorithm settings
    alg_settings, alg_time = get_args()

    # Set algorithm settings
    data_settings = read_file_json(alg_settings)

    # Set algorithm logging
    os.makedirs(data_settings['data']['log']['folder'], exist_ok=True)
    set_logging(
        logger_file=os.path.join(data_settings['data']['log']['folder'], data_settings['data']['log']['filename']))
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Get time settings
    date_now = pytz.utc.localize(datetime.strptime(alg_time, "%Y-%m-%d %H:%M"))
    forecast_end = date_now + timedelta(hours=data_settings['data']['dynamic']['HMC']['forecast_length_h'] - 1)
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

    output_path = output_path_empty.format(**dict_filled)
    output_resume_path = output_resume_path_empty.format(**dict_filled)
    ancillary_path = ancillary_path_empty.format(**dict_filled)

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_resume_path, exist_ok=True)
    os.makedirs(ancillary_path, exist_ok =True)
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
    # Read and merge hydrological outputs
    logging.info(" ---> Analyse HMC results...")

    flood_succesful = True

    for group in data_settings["data"]["dynamic"]["HMC"]["domains"].keys():
        logging.info(" ----> GROUP " + group)

        for domain in data_settings["data"]["dynamic"]["HMC"]["domains"][group]:
            logging.info(" ----> Domain " + domain)

            if os.path.isfile(os.path.join(ancillary_path, group + '_alert_level_' + domain + '.tif')) and \
                    os.path.isfile(os.path.join(ancillary_path, 'max_discharge', group + '_discharge_max_' + domain + '.tif')) and \
                    data_settings["algorithm"]["flags"]["use_partial_computed_map"]:
                logging.info(" ----> Results previously computed...SKIP")
                pass
            else:
                dict_filled["domain"] = domain
                out_hmc_path = data_settings["data"]["dynamic"]["HMC"]["folder"].format(**dict_filled)

                logging.info(" ----> Extract results...")
                first_step = True
                for time_now in pd.date_range(date_now, forecast_end, freq="H"):
                    file = os.path.join(out_hmc_path, "hmc.output-grid.{dateout}.nc".format(
                        dateout=time_now.strftime("%Y%m%d%H%M")))
                    if os.path.isfile(file + ".gz"):
                        os.system("gunzip -f " + file + ".gz")

                    try:
                        map_now = xr.open_dataset(file)["Discharge"].values
                    except:
                        logging.error(" ERROR! Output file " + file + " not found!")
                        raise FileNotFoundError

                    if first_step:
                        dis_max = map_now
                        mask = np.where(xr.open_dataset(file)["SM"].values < 0, 0, 1)
                    else:
                        dis_max = np.maximum(dis_max, map_now)
                logging.info(" ----> Extract results...DONE")

                alert_maps = np.ones(dis_max.shape)

                logging.info(" ----> Assign flood hazard level...")
                rps = [str(i).zfill(3) for i in data_settings["data"]["static"]["discharge_thresholds"]["return_periods"]]
                for val, rp in enumerate(rps, start=2):
                    th_map_file = rio.open(os.path.join(
                        data_settings["data"]["static"]["discharge_thresholds"]["folder"],
                        data_settings["data"]["static"]["discharge_thresholds"]["file_name"]).format(
                        domain=domain, return_period=rp))
                    th_map = np.flipud(th_map_file.read(1).squeeze())
                    th_map[th_map <= 0] = np.Inf
                    alert_maps = np.where(dis_max >= th_map, val, alert_maps)

                alert_maps = np.where(mask == 1, alert_maps, 0)
                transform = th_map_file.profile['transform']

                area = np.flipud(rio.open(os.path.join(
                    data_settings["data"]["static"]["gridded_HMC"],
                    "{domain}.area.txt").format(domain=domain)).read(1).squeeze())
                areacell = np.flipud(rio.open(os.path.join(
                    data_settings["data"]["static"]["gridded_HMC"],
                    "{domain}.areacell.txt").format(domain=domain)).read(1).squeeze())
                area_km = area * areacell / (10 ** 6)

                discharge_mask = np.where((th_map >= data_settings["data"]["dynamic"]["thresholds"]["discharge_min"]) &
                                          (np.isfinite(th_map)) &
                                          (area_km >= data_settings["data"]["dynamic"]["thresholds"]["area_km2"]), 1, 0)
                write_tif(os.path.join(ancillary_path, group + '_discharge_mask_' + domain + '.tif'),
                    discharge_mask.astype(np.int16), transform, no_data=0)

                if debug_mode:
                    os.makedirs(os.path.join(ancillary_path, 'max_discharge'), exist_ok=True)
                    write_tif(os.path.join(ancillary_path, 'max_discharge',
                                           group + '_discharge_max_' + domain + '.tif'), dis_max, transform)

                write_tif(os.path.join(ancillary_path, group + '_alert_level_' + domain + '.tif'),
                          alert_maps, transform, no_data=0)
                os.system("gzip -f " + os.path.join(out_hmc_path, "*.nc"))
                logging.info(" ----> Assign flood hazard level...DONE")

        logging.info(logging.info(" ----> Merge results for GROUP " + group))
        os.system("gdal_merge.py -o " + os.path.join(ancillary_path,
                                                     group + "_mosaic_alert_level.tif") + " " + os.path.join(
            ancillary_path, group + '_alert_level_*.tif'))

        os.system("gdal_merge.py -o " + os.path.join(ancillary_path,
                                                     group + "_discharge_mask.tif") + " " + os.path.join(
            ancillary_path, group + '_discharge_mask_*.tif'))

        if not debug_mode:
            os.system("rm " + os.path.join(ancillary_path, group + '_alert_level_*.tif'))
            os.system("rm " + os.path.join(ancillary_path, group + '_discharge_mask_*.tif'))

    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    logging.info(" --> Classify flood country warning level...")

    shp_fo = data_settings["data"]["static"]["warning_regions"]
    shp_df = gpd.read_file(shp_fo)
    shp_df_hydro_model = deepcopy(shp_df)
    shp_df_hydro_model["level_HMC"] = -9999.0

    for group in data_settings["data"]["dynamic"]["HMC"]["domains"].keys():
        logging.info(" ----> GROUP " + group)

        th_levels = xr.open_rasterio(os.path.join(ancillary_path,
                                                  group + "_mosaic_alert_level.tif")).squeeze().rename(
            {"x": "lon", "y": "lat"})


        mask_discharge = xr.open_rasterio(os.path.join(ancillary_path,
                                                       group + "_discharge_mask.tif")).squeeze().rename(
            {"x": "lon", "y": "lat"})
        th_levels_masked = xr.where(mask_discharge == 1, th_levels, 0)

        shapes = [(shape, n) for n, shape in enumerate(shp_df.geometry)]
        ds = xr.Dataset(coords={'lon': th_levels['lon'].values, 'lat': th_levels['lat'].values})
        ds['states'] = rasterize(shapes, ds.coords)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            logging.info(" ---> Loop through the alert zones...")
            for index, row in shp_df_hydro_model.iterrows():
                shp_df_hydro_model.at[index, group] = np.nanmax(
                    th_levels_masked.where(ds['states'] == index))
            logging.info(" ---> Loop through the alert zones...DONE")

        shp_df_hydro_model[group] = shp_df_hydro_model[group].fillna(-9999)
        shp_df_hydro_model["level_HMC"] = np.where(
            shp_df_hydro_model[group] > shp_df_hydro_model["level_HMC"],
            shp_df_hydro_model[group], shp_df_hydro_model["level_HMC"])
        shp_df_hydro_model = shp_df_hydro_model.drop(columns=[group])

        th_levels_masked.to_netcdf(os.path.join(ancillary_path, "temp_output.nc"))
        dict_filled["group"]=group
        file_out_name_tif = data_settings["data"]["dynamic"]["outcome"]["file_name"].format(**dict_filled)

        os.system(
            "gdal_translate -a_srs epsg:4326 -of GTiff " + os.path.join(ancillary_path,
                                                                        "temp_output.nc") + " " + os.path.join(
                output_path, file_out_name_tif))

    logging.info(" ---> Classify HMC results...DONE")

    logging.info(" --> Classify flood country warning level...DONE")
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Write output and clean
    logging.info(" --> Write output...")

    file_name_shp = data_settings["data"]["dynamic"]["outcome"]["file_name_resume"].format(**dict_filled)

    shp_df_hydro_model.to_file(os.path.join(output_resume_path, file_name_shp))

    if data_settings["algorithm"]["flags"]["clear_ancillary_data"]:
        os.system("rm -r " + ancillary_path)

    logging.info(" --> Write gridded output...DONE")
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

# -------------------------------------------------------------------------------------
# Method to fill path names
def fill_template_time_step(template_dict, timeNow):
    template_compiled = {}
    for i in template_dict.keys():
        if '%' in template_dict[i]:
            template_compiled[i] = timeNow.strftime(template_dict[i])
        else:
            template_compiled[i] = template_dict[i]

    return template_compiled


# ----------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------
# Write tif file
def write_tif(out_name, Z, transform, flip_lat=True, proj='epsg:4326', no_data=-9990.0):
    if flip_lat:
        Z = np.flipud(Z)

    with rio.open(
            out_name,
            'w',
            driver='GTiff',
            height=Z.shape[0],
            width=Z.shape[1],
            count=1,
            dtype=Z.dtype,
            crs=proj,
            transform=transform,
            nodata=no_data
    ) as dst:
        dst.write(Z, 1)


# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Call script from external library
if __name__ == "__main__":
    main()
# ----------------------------------------------------------------------------


