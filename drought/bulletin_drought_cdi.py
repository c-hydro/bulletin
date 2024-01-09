"""
bulletin - drought - CDI
__date__ = '20230623'
__version__ = '1.0.0'
__author__ =
        'Andrea Libertino (andrea.libertino@cimafoundation.org',
__library__ = 'bulletin'
General command line:
### python bulletin_drougth_cdi.py -settings_file "settings.json" -time "YYYY-MM-DD HH:MM"
Version(s):
20200326 (1.0.0) --> Beta prototipe
"""
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Complete library
import os, logging, netrc, json
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
import glob
import rasterio as rio
import rioxarray as rx
import sys

# -------------------------------------------------------------------------------------
# Script Main
def main():
    # -------------------------------------------------------------------------------------
    # Version and algorithm information
    alg_name = 'bulletin - Drought with CDI '
    alg_version = '1.0.0'
    alg_release = '2023-06-23'
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

    if date_run.day == 1:
        template_time_step["map_num"] = "1st"
    elif date_run.day == 11:
        template_time_step["map_num"] = "2nd"
    elif date_run.day == 21:
        template_time_step["map_num"] = "3rd"
    else:
        raise NotImplementedError("Only the first 3 maps per month are supported")

    ancillary_folder = data_settings['data']['dynamic']['ancillary']['folder'].format(**template_time_step)
    ancillary_file_name = os.path.join(ancillary_folder,
                                       data_settings['data']['dynamic']['ancillary']['file_name']).format(**template_time_step)

    output_folder_table_impacts = data_settings['data']['dynamic']['outcome']['folder_table_impacts'].format(**template_time_step)
    output_file_table_impacts = os.path.join(output_folder_table_impacts,
                                             data_settings['data']['dynamic']['outcome'][
                                                 'file_name_table_impacts']).format(**template_time_step)

    output_folder_shape_hazard = data_settings['data']['dynamic']['outcome']['folder_shape_hazard'].format(**template_time_step)
    output_file_shape_hazard = os.path.join(output_folder_shape_hazard,
                                            data_settings['data']['dynamic']['outcome'][
                                                'file_name_shape_hazard']).format(**template_time_step)

    output_folder_shape_impacts = data_settings['data']['dynamic']['outcome']['folder_shape_impacts'].format(**template_time_step)
    output_file_shape_impacts = os.path.join(output_folder_shape_impacts,
                                            data_settings['data']['dynamic']['outcome'][
                                                'file_name_shape_impacts']).format(**template_time_step)

    os.makedirs(output_folder_table_impacts, exist_ok=True)
    os.makedirs(output_folder_shape_impacts, exist_ok=True)
    os.makedirs(output_folder_shape_hazard, exist_ok=True)
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

    # Flag to verify if the local data derive from a previous run of this procedure (not overwrite temp file)
    check_previous_dload = False
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Invalid settings
    if (data_settings["algorithm"]["settings"]["download_input"] is True and data_settings["algorithm"]["settings"]["use_local_input_file"] is True) or \
        (data_settings["algorithm"]["settings"]["download_input"] is False and data_settings["algorithm"]["settings"]["use_local_input_file"] is False):
        logging.error("ERROR! Select if download forecast or use local forecast!")
        raise ValueError

    # Download with drops
    elif data_settings["algorithm"]["settings"]["download_input"]:
        logging.warning("ERROR! Download of input data is not implemented yet!")

    # Use local file
    elif data_settings["algorithm"]["settings"]["use_local_input_file"]:
        logging.error(" --> Read local forecast file...")
        input_folder = data_settings['data']['dynamic']['input']["local_file"]['folder'].format(**template_time_step)
        input_file = os.path.join(input_folder, data_settings['data']['dynamic']['input']['local_file'][
                                                     'file_name']).format(**template_time_step)
        if not os.path.isfile(input_file):
            logging.error("ERROR! File " + input_file + " is not found!")
            raise FileNotFoundError
        else:
            logging.info(" --> File " + input_file + " is found!")
        data = xr.open_rasterio(input_file)
        logging.info(" --> Use of local available forecast...DONE")
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Postprocess data
    logging.info(" ---> Crop forecast over domain...")
    min_lon = data_settings["data"]["dynamic"]["bbox"]['lon_left']
    max_lon = data_settings["data"]["dynamic"]["bbox"]['lon_right']
    min_lat = data_settings["data"]["dynamic"]["bbox"]['lat_bottom']
    max_lat = data_settings["data"]["dynamic"]["bbox"]['lat_top']

    mask_lon = (data.x >= min_lon) & (data.x <= max_lon)
    mask_lat = (data.y >= min_lat) & (data.y <= max_lat)
    data = data.where(mask_lon & mask_lat, drop=True)
    logging.info(" ---> Crop forecast over domain...DONE")
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Classify rainfall alert level
    logging.info(" --> Classify drought alert level...")
    thresholds = data_settings['data']['static']['drought_tresholds']
    alert_maps = np.where(data.values >= 0, 1, np.nan)

    for val, th in enumerate(thresholds, start=2):
        alert_maps = np.where(data.values >= th, val, alert_maps)

    alert_map_temp = deepcopy(alert_maps)

    if data_settings["algorithm"]["flags"]["convert_hazard_classes"]:
        logging.info(" ----> Convert hazard classes")
        haz_class_out = np.ones(alert_map_temp.shape) * 0
        dict_conversion = data_settings["data"]["static"]["conversion_table"]
        for class_out in dict_conversion.keys():
            classes_in = dict_conversion[class_out]
            haz_class_out[np.isin(alert_map_temp, classes_in)] = int(class_out)
        alert_maps = haz_class_out
        logging.info(" ----> Convert hazard classes...DONE")

    logging.info(" ----> Assign flood hazard level...DONE")

    logging.info(" --> Classify drought alert level...DONE")
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Assign hazard and impact levels
    alert_daily = xr.Dataset({"dry":xr.DataArray(alert_maps, dims=["time","lat","lon"], coords={"time":[date_run], "lon":data.x.values, "lat":data.y.values})})
    hazards = ["dry"]

    if data_settings["algorithm"]["flags"]["hazard_assessment"]:
        logging.info(" --> Classify meteo country warning level based on hazard...")
        shp_fo = data_settings["data"]["static"]["warning_regions"]
        shp_df = gpd.read_file(shp_fo)
        for hazard in hazards:
            classify_warning_levels_pure_hazard(hazard,
                                                output_file_shape_hazard.format(HAZARD=hazard.upper(), hazard=hazard),
                                                shp_df, alert_daily,
                                                data_settings["algorithm"]["settings"]["min_warning_pixel"])
        logging.info(" --> Classify meteo country warning level based on hazard...DONE")

    if data_settings["algorithm"]["flags"]["impact_assessment"]:
        logging.info(" --> Classify meteo country warning level based on impacts...")
        shp_fo = data_settings["data"]["static"]["warning_regions"]
        shp_df = gpd.read_file(shp_fo)
        for hazard in hazards:
            classify_warning_levels_impact_based(hazard,
                                                output_file_shape_impacts.format(HAZARD=hazard.upper(), hazard=hazard),
                                                shp_df, alert_daily,
                                                data_settings["data"]["impacts"])

            # Summarize results in table if required
            if data_settings["data"]["impacts"]["impact_summary"]["save_impact_table"]:
                logging.info(" --> Summarize results in a table...")
                impact_results = gpd.read_file(output_file_shape_impacts.format(HAZARD=hazard.upper(), hazard=hazard))
                grouped_table = impact_results.groupby(data_settings["data"]["impacts"]["impact_summary"]["impact_table_aggregation_column"]).sum().sort_values(hazard + "AffPpl", ascending=False)
                grouped_table[hazard + "AffPpl"].to_csv(os.path.join(output_file_table_impacts, output_file_table_impacts.format(HAZARD=hazard.upper(), hazard=hazard)), header=False, float_format='%.0f')
                logging.info(" --> Summarize results in a table...DONE")
        logging.info(" --> Classify meteo country warning level based on impacts...DONE")
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Write output and clean
    logging.info(" --> Write output...")

    if not data_settings["algorithm"]["flags"]["clear_ancillary_data"]:
        os.makedirs(ancillary_folder, exist_ok=True)
        make_tif(alert_maps.squeeze(), data.x.values, data.y.values, ancillary_file_name, dtype='int16')

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
# Classify a shapefile of countries with an alert level map with a pure-hazard approach
def classify_warning_levels_pure_hazard(hazard, out_name, shp_df, alert_daily, min_warning_treshold=1):
    shapes = [(shape, n) for n, shape in enumerate(shp_df.geometry)]
    ds = xr.Dataset(coords={'lon': alert_daily['lon'].values,
                            'lat': alert_daily['lat'].values})
    ds['states'] = rasterize(shapes, ds.coords)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        shp_df_step = deepcopy(shp_df)
        shp_df_step[hazard + "_level"] = -9999.0

        alert_step = alert_daily[hazard]
        logging.info(" ---> Loop through the alert zones for " + hazard + " risk...")

        for index, row in shp_df_step.iterrows():
            # Rain
            val_max = np.nanmax(alert_step.where(ds['states'] == index))
            while val_max > 1:
                tot_over = np.count_nonzero(alert_step.where(ds['states'] == index) >= val_max)
                if tot_over >= min_warning_treshold:
                    break
                else:
                    val_max = val_max - 1
            shp_df_step.at[index, hazard + "_level"] = val_max

        logging.info(" ---> Loop through the alert zones...DONE")

        logging.info(" ---> Save shapefile for " + hazard + " risk...")
        shp_df_step.to_file(out_name)
        logging.info(" ---> Save shapefile for " + hazard + " risk...DONE")

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Classify a shapefile of countries with an alert level map with an impact-based approach
def classify_warning_levels_impact_based(hazard, out_name, shp_df, alert_daily, impact_dict):
    alert_daily_max = alert_daily.max(dim='time')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        shp_df_step = deepcopy(shp_df)

        shp_df_step["pop_total"] = -9999.0
        shp_df_step[hazard + "AffPpl"] = -9999.0
        shp_df_step[hazard + "AffPrc"] = -9999.0
        shp_df_step[hazard + "_level"] = -9999.0

        logging.info(" ---> Loop through the alert zones for " + hazard + " risk...")

        for index, row in shp_df_step.iterrows():
            logging.info(" ----> Computing zone " + str(index +1) + " of " + str(len(shp_df_step)))
            bbox = row["geometry"].bounds
            clipped_pop = rx.open_rasterio(impact_dict["exposed_population_map"]).rio.clip_box(minx=bbox[0],miny=bbox[1],maxx=bbox[2],maxy=bbox[3])
            clipped_pop.values[clipped_pop.values<0] = 0
            lon_bbox = clipped_pop.x.values
            lat_bbox = clipped_pop.y.values
            alert_bbox = alert_daily_max.reindex({"lon":lon_bbox, "lat":lat_bbox}, method="nearest")
            country_bbox = rasterize([(row['geometry'], index+1)], {"lon":lon_bbox, "lat":lat_bbox})

            weigth_map = np.where(country_bbox==index+1,0,np.nan)
            if hazard == "dry":
                haz_full = "drought"
            else:
                haz_full = hazard
            for lev, weigth in enumerate(impact_dict["weight_hazard_levels"][haz_full], start=2):
                weigth_map = np.where(alert_bbox[hazard].values==lev,weigth_map+weigth,weigth_map)
            aff_people = np.nansum(
                weigth_map * np.squeeze(clipped_pop.values) * (row[impact_dict["lack_coping_capacity_col"]] / 10))
            tot_people = np.nansum(
                np.where(country_bbox == index + 1, np.squeeze(clipped_pop.values), np.nan))
            if tot_people == 0:
                impact_rate = 0
            else:
                impact_rate =  aff_people / tot_people
            risk = 0

            for risk_lev, (risk_th_abs, risk_th_rel) in enumerate(zip(impact_dict["risk_thresholds"]["absolute"],
                                                                      impact_dict["risk_thresholds"]["relative"]), start=1):
                if risk_th_abs is None: risk_th_abs = 0
                if risk_th_rel is None: risk_th_rel = 0

                if risk_th_abs == 0 and risk_th_rel == 0:
                    logging.error(" ERROR! Either a relative or an absolute threshold value should be provided for each risk class!")
                    raise ValueError("Both absolute and relative trhesholds are none for class " + str(risk_lev))
                elif impact_rate >= risk_th_rel and aff_people >= risk_th_abs:
                    risk = risk_lev
                else:
                    break

            shp_df_step.at[index, hazard + "_level"] = risk
            shp_df_step.at[index, hazard + "AffPpl"] = aff_people
            shp_df_step.at[index, hazard + "AffPrc"] = impact_rate
            shp_df_step.at[index, "pop_total"] = tot_people

        logging.info(" ---> Loop through the alert zones for " + hazard + " risk...DONE")

        logging.info(" ---> Save shapefile for " + hazard + " risk...")
        shp_df_step.to_file(out_name)
        logging.info(" ---> Save shapefile for " + hazard + " risk...DONE")

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def make_tif(val, lon, lat, out_filename, crs='epsg:4326', nodata=-9999, dtype='float32'):
    out_ds = xr.DataArray(val, dims=["y", "x"], coords={"y": lat, "x": lon})
    out_ds = out_ds.where(out_ds != nodata).rio.write_crs(crs, inplace=True).rio.write_nodata(nodata, inplace=True)
    out_ds.values = out_ds.values.astype(dtype)
    out_ds.rio.to_raster(out_filename, driver="GTiff", crs='EPSG:4326', height=len(lat), width=len(lon), dtype=out_ds.dtype, compress="DEFLATE")

# -------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Call script from external library
if __name__ == "__main__":
    main()
# ----------------------------------------------------------------------------


