"""
bulletin - meteo multimodel merger
__date__ = '20230901'
__version__ = '1.0.0'
__author__ =
        'Andrea Libertino (andrea.libertino@cimafoundation.org',
__library__ = 'bulletin'
General command line:
### python fp_bulletin_meteo_gfs.py -settings_file "settings.json" -time "YYYY-MM-DD HH:MM"
Version(s):
20230901 (1.0.0) --> Beta release
"""
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Complete library
import os, logging, netrc, json
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
import glob
import rasterio as rio
import rioxarray as rx
import sys

# -------------------------------------------------------------------------------------
# Script Main
def main():
    # -------------------------------------------------------------------------------------
    # Version and algorithm information
    alg_name = 'bulletin - Multimodel merger'
    alg_version = '1.0.0'
    alg_release = '2023-09-01'
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
    date_run = datetime.strptime(alg_time, "%Y-%m-%d %H:%M")
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Create directories
    template_time_step = fill_template_time_step(data_settings['algorithm']['template'], date_run)
    template_time_step["model"] = "{model}"
    template_time_step["hazard"] = "{hazard}"
    template_time_step["HAZARD"] = "{HAZARD}"

    output_folder_shape_impacts = data_settings['data']['outcome']['folder_shape_impacts'].format(**template_time_step)
    output_file_shape_impacts = os.path.join(output_folder_shape_impacts,
                                            data_settings['data']['outcome'][
                                                'file_name_shape_impacts']).format(**template_time_step)
    os.makedirs(output_folder_shape_impacts, exist_ok=True)
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Info algorithm
    logging.info(' ============================================================================ ')
    logging.info(' ==> START ... ')
    logging.info(' ')

    logging.info(" --> Time now : " + alg_time)

    # Time algorithm information
    start_time = time.time()
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Collect data
    logging.info(" --> Manage input forecast data...")
    hazards = data_settings["algorithm"]["settings"]["hazards"]
    hazards_short = data_settings["algorithm"]["settings"]["hazards_short"]
    # -------------------------------------------------------------------------------------
    for hazard, haz in zip(hazards, hazards_short):
        first_step = True
        total = 0
        logging.info(" --> Read input data for hazard " + hazard + "...")
        for model in data_settings['data']['models'].keys():
            logging.info(" ---> Model " + model + "...")
            file_in = data_settings['data']['models'][model]['file'].format(**template_time_step).format(model=model, hazard=hazard, HAZARD=hazard.upper())
            if not os.path.isfile(file_in):
                if data_settings["algorithm"]["flags"]["raise_error_if_missing"]:
                    raise FileNotFoundError(" ---> ERROR! File " + file_in + " not found!")
                else:
                    logging.warning(" ---> WARNING! File " + file_in + " not found!")
                    continue
            model_data = gpd.read_file(file_in)
            if first_step is True:
                output_shape = deepcopy(model_data)
                output_shape[haz + "_tot"] = output_shape[haz + "_tot"] * data_settings['data']['models'][model]['weight']
                first_step = False
                total = total + data_settings['data']['models'][model]['weight']
            else:
                output_shape[haz + "_tot"] = output_shape[haz + "_tot"] + model_data[haz + "_tot"] * data_settings['data']['models'][model]['weight']
                total = total + data_settings['data']['models'][model]['weight']
            logging.info(" ---> Model " + model + "...DONE")
        output_shape[haz + "_tot"] = output_shape[haz + "_tot"] / total
        output_shape[haz + "_perc"] = output_shape[haz + "_tot"] / output_shape["stock"]
        logging.info(" --> Read input data for hazard " + hazard + "...DONE")

        logging.info(" --> Assign risk levels for hazard " + hazard + "...")
        risk_treshold = data_settings["data"]["risk_thresholds"]
        output_shape[haz + "_level"] = output_shape.apply(lambda x: assign_risk(x[haz + "_perc"], x[haz + "_tot"], risk_treshold), axis=1)
        logging.info(" ---> Assign risk levels for hazard " + hazard + "...DONE")

        logging.info(" --> Write output...")
        output_shape.to_file(output_file_shape_impacts.format(HAZARD=hazard.upper(), hazard=hazard))
        logging.info(" --> Write output...DONE")
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
def assign_risk(value_rel, value_abs, risk_treshold):
    risk = 0
    for risk_lev, (risk_th_abs, risk_th_rel) in enumerate(zip(risk_treshold["absolute"],
                                                              risk_treshold["relative"]), start=1):
        if risk_th_abs is None: risk_th_abs = 0
        if risk_th_rel is None: risk_th_rel = 0

        if risk_th_abs == 0 and risk_th_rel == 0:
            logging.error(
                " ERROR! Either a relative or an absolute threshold value should be provided for each risk class!")
            raise ValueError("Both absolute and relative trhesholds are none for class " + str(risk_lev))
        elif value_rel >= risk_th_rel and value_abs >= risk_th_abs:
            risk = risk_lev
        else:
            break
    return risk
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Call script from external library
if __name__ == "__main__":
    main()
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------