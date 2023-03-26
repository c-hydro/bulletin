"""
bulletin - hydro - floodPROOFS
__date__ = '20221121'
__version__ = '1.0.0'
__author__ =
        'Andrea Libertino (andrea.libertino@cimafoundation.org',
__library__ = 'bulletin'
General command line:
### python bulletin_hydro_fp.py -settings_file "settings.json" -time "YYYY-MM-DD HH:MM"
Version(s):
20221121 (1.0.0) --> Beta release
"""
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Complete library
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
import rioxarray as rx
import pandas as pd
from copy import deepcopy
from rioxarray import merge

# -------------------------------------------------------------------------------------
# Script Main
def main():
    # -------------------------------------------------------------------------------------
    # Version and algorithm information
    alg_name = 'bulletin - Hydrological warning with FloodProofs '
    alg_version = '1.0.0'
    alg_release = '2022-11-22'
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Get algorithm settings
    alg_settings, alg_time = get_args()

    # Set algorithm settings
    data_settings = read_file_json(alg_settings)
    paths = {}

    # Set algorithm logging
    os.makedirs(data_settings['data']['log']['folder'], exist_ok=True)
    set_logging(
        logger_file=os.path.join(data_settings['data']['log']['folder'], data_settings['data']['log']['filename']))
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Get time settings
    date_now = pytz.utc.localize(dt.datetime.strptime(alg_time, "%Y-%m-%d %H:%M"))
    forecast_end = date_now + dt.timedelta(hours=data_settings['data']['dynamic']['fp']['forecast_length_h'] - 1)
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Create directories
    output_maps_empty = data_settings['data']['dynamic']['outcome']['folder_maps']
    output_shape_empty = data_settings['data']['dynamic']['outcome']['folder_shape']
    ancillary_path_empty = data_settings['data']['dynamic']['ancillary']['folder']

    dict_empty = data_settings['algorithm']['template']
    dict_filled = dict_empty.copy()

    for key in dict_empty.keys():
        dict_filled[key] = date_now.strftime(dict_empty[key])

    paths["output_maps"] = output_maps_empty.format(**dict_filled)
    paths["output_shape"] = output_shape_empty.format(**dict_filled)
    paths["ancillary"] = ancillary_path_empty.format(**dict_filled)

    os.makedirs(paths["output_maps"], exist_ok=True)
    os.makedirs(paths["output_shape"], exist_ok=True)
    os.makedirs(paths["ancillary"], exist_ok=True)
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
    # Read and classify results
    flood_maps_ready = []
    missing_domains = []

    impact_dict = data_settings["data"]["impacts"]
    impact_dict["return_periods"] = data_settings["data"]["static"]["discharge_thresholds"]["return_periods"]

    for domain in data_settings["data"]["dynamic"]["fp"]["domains"]:
        logging.info(" --> Compute fp forecast with date: " + date_now.strftime("%Y-%m-%d %H:%M") + " for domain " + domain)
        dict_filled["domain"] = domain

        # Define filenames and check existence of flood ancillary map
        static_hmc_path = data_settings["data"]["dynamic"]["fp"]["static_data_folder"].format(**dict_filled)
        out_hmc_path = data_settings["data"]["dynamic"]["fp"]["results_folder"].format(**dict_filled)
        out_flood_ancillary_map = os.path.join(paths["ancillary"], data_settings["data"]["dynamic"]["ancillary"]["file_name_flood"]).format(**dict_filled)

        if os.path.isfile(out_flood_ancillary_map) and not data_settings["algorithm"]["flags"]["recompute_domains"]:
            logging.info(" --> Domain " + domain + " has already been computed! SKIPPING!")
            flood_maps_ready = flood_maps_ready + [out_flood_ancillary_map]
            continue

        # Read and calculate max discharge
        logging.info(" ----> Extract results...")
        try:
            dis_max, river_mask, lon, lat = extract_max_hmc_results(out_hmc_path, date_now, forecast_end)
            logging.info(" ----> Extract results...DONE")
            make_tif(dis_max, lon, lat, os.path.join(paths["ancillary"], data_settings["data"]["dynamic"]["ancillary"]["file_name_max"]).format(**dict_filled))
        except FileNotFoundError:
            missing_domains = missing_domains + [domain]
            logging.error(" --> Map not found, are you sure domain " + domain + " has already run? \n --> Skip to next domain!")
            continue

        # Classify discharge maps
        alert_map = np.ones(dis_max.shape)
        logging.info(" ----> Assign flood hazard level...")
        # Load static map for filtering
        area = check_raster(xr.open_rasterio(os.path.join(static_hmc_path, "{domain}.area.txt").format(**dict_filled)))
        areacell = check_raster(xr.open_rasterio(os.path.join(static_hmc_path, "{domain}.areacell.txt").format(**dict_filled)))
        area_km = (area * areacell / (10**6)).squeeze()
        mask = np.where((river_mask==1) & (area_km>=data_settings["data"]["dynamic"]["thresholds"]["area_km2"]),1,0)

        # Load tresholds and classify
        rps = [str(i) for i in data_settings["data"]["static"]["discharge_thresholds"]["return_periods"]]
        for val, rp in enumerate(rps, start=2):
            th_map_file = xr.open_rasterio(os.path.join(
                data_settings["data"]["static"]["discharge_thresholds"]["folder"],
                data_settings["data"]["static"]["discharge_thresholds"]["file_name"]).format(
                domain=domain, return_period=rp))
            th_map = check_raster(th_map_file)
            th_map[th_map <= 0] = np.Inf
            alert_map = np.where(((dis_max >= th_map) & (dis_max >= data_settings["data"]["dynamic"]["thresholds"]["discharge_min"])), val, alert_map)
        alert_map = np.where(mask == 1, alert_map, 0)
        alert_map_out = deepcopy(alert_map)

        # Reclassify hazard classes
        if data_settings["algorithm"]["flags"]["convert_hazard_classes"]:
            logging.info(" ----> Convert hazard classes")
            haz_class_out = np.ones(alert_map_out.shape) * 0
            dict_conversion = data_settings["data"]["hazard"]["conversion_table"]
            for class_out in dict_conversion.keys():
                classes_in = dict_conversion[class_out]
                haz_class_out[np.isin(alert_map_out, classes_in)] = int(class_out)
            alert_map_out = haz_class_out
        make_tif(alert_map_out, lon, lat, os.path.join(paths["ancillary"],data_settings["data"]["dynamic"]["ancillary"]["file_name_alert"]).format(**dict_filled), dtype='int16')
        logging.info(" ----> Assign flood hazard level...DONE")

        # Mosaic weigth map for the domain
        logging.info(" ----> Merge flood map...")
        impact_dict["domain"] = domain

        flood_map, lat_mosaic, lon_mosaic = create_flood_map(alert_map, impact_dict)
        make_tif(np.real(flood_map.values), lon_mosaic, lat_mosaic, out_flood_ancillary_map)
        logging.info(" ----> Merge flood map...DONE")

        flood_maps_ready = flood_maps_ready + [out_flood_ancillary_map]

    # Check for all domains to be computed
    if len(flood_maps_ready) < len(data_settings["data"]["dynamic"]["fp"]["domains"]):
        logging.warning(" --> WARNING! Domains " + ','.join(missing_domains) + " are missing")
        if not data_settings["algorithm"]["flags"]["compute_partial_results"]:
            logging.error(" --> ERROR! Not all the domains has been computed!")
            raise RuntimeError

    # Merge maps of all the domains
    igad_weight_map = xr.open_rasterio(flood_maps_ready[0]) * 0
    for val in flood_maps_ready:
        igad_weight_map.values = igad_weight_map.values + xr.open_rasterio(val).values

    logging.info(" ----> Save merged flood map...")
    dict_filled['domain'] = "merged"
    out_flood_merged_map = os.path.join(paths["ancillary"],data_settings["data"]["dynamic"]["ancillary"]["file_name_flood"]).format(**dict_filled)
    out_file_flood_tif = os.path.join(paths["output_maps"], data_settings['data']['dynamic']['outcome']['file_name_flood_map']).format(**dict_filled)
    igad_weight_map.rio.to_raster(out_flood_merged_map)

    out_mask = igad_weight_map / igad_weight_map
    out_mask.rio.write_crs("epsg:4326", inplace=True)
    out_mask.rio.write_nodata(0, inplace=True)
    out_mask.astype(np.int16).rio.to_raster(out_file_flood_tif, compress='DEFLATE', dtype='int16')
    logging.info(" ----> Save merged flood map...DONE")
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Assign impact levels
    for exposed_element in data_settings["data"]["impacts"]["exposed_map"].keys():
        dict_filled["exposed_element"] = exposed_element
        impact_dict["exposed_element"] = exposed_element

        logging.info(" ---> Assign flood country warning levels for " + exposed_element + "...")
        shp_fo = data_settings["data"]["static"]["warning_regions"]
        shp_df = gpd.read_file(shp_fo)

        shp_df_hydro_model = shp_df.copy()
        out_file_shp = os.path.join(paths["output_shape"], data_settings['data']['dynamic']['outcome']['file_name_shape_impacts'].format(**dict_filled))

        if np.nanmax(igad_weight_map) > 0 or data_settings["algorithm"]["flags"]["compute_stock_when_no_impact"]:
            logging.info("--> Classify warning levels..")
            classify_warning_levels_impact_based(shp_df_hydro_model, igad_weight_map, out_file_shp, impact_dict)
            logging.info("--> Classify warning levels..DONE")
        else:
            logging.info("--> No impacts forecasted in the current forecast..")
            shp_df_hydro_model["stock"] = -9999
            shp_df_hydro_model["flood_tot"] = 0
            shp_df_hydro_model["flood_perc"] = 0
            shp_df_hydro_model.to_file(out_file_shp)

        logging.info(" --> Classify flood country warning levels...DONE")

    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Write outputs
    logging.info(" --> Write outputs and clean system...")
    logging.info(" ---> Write output shapefile...")

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
# Mosaic flood maps
def create_flood_map(th_levels, impact_dict):
    logging.info(" --> Tailoring flood map")
    decode_map = xr.open_rasterio(impact_dict["decode_map"].format(domain=impact_dict["domain"])).squeeze()
    decode_map_values = check_raster(decode_map)
    decode_map_values[decode_map_values < 0] = -1

    for level, associated_rp in enumerate(impact_dict["flood_maps"]["associated_rp"], start=2):
        logging.info(' ---> Import hazard level ' + str(level))
        flood_map_level = xr.open_rasterio(impact_dict["flood_maps"]["file_name"].format(return_period=str(associated_rp))).squeeze().astype(np.int32)
        flood_map_level.values = check_raster(flood_map_level)
        flood_map_level.values[flood_map_level.values > 99999] = 0
        flood_map_level.values[flood_map_level.values < 0] = 0
        if level==2:
            mosaic_flood_map = flood_map_level.copy()*0
            lat_mosaic = np.sort(flood_map_level.y.values)
            lon_mosaic = flood_map_level.x.values
        if level in np.unique(th_levels):
            codes_in = decode_map_values[th_levels==level]
            codes_in = codes_in[codes_in>0]
            flood_map_level.values[np.isin(flood_map_level.values,codes_in,invert=True)] = 0
            mosaic_flood_map = xr.where(flood_map_level>1,level,mosaic_flood_map)

    mosaic_weigthed_map = mosaic_flood_map.copy()
    mosaic_weigthed_map.values = mosaic_weigthed_map.values*0
    for key, level in enumerate(impact_dict["weight_hazard_levels"],start=2):
        mosaic_weigthed_map.values = np.where(mosaic_flood_map.values == key, level, mosaic_weigthed_map.values )

    return mosaic_weigthed_map.astype(np.float32), lat_mosaic, lon_mosaic

# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Classify a shapefile of countries with an alert level map with an impact-based approach
def classify_warning_levels_impact_based(shp_df, mosaic_flood_map, out_file_shp, impact_dict, hazard = "flood"):
    if impact_dict["risk_thresholds"]["absolute"] is None or impact_dict["risk_thresholds"]["relative"] is None:
        no_tresholds = True
    else:
        no_tresholds = False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        shp_df_step = shp_df.copy()

        shp_df_step["stock"] = -9999.0
        shp_df_step[hazard + "_tot"] = -9999.0
        shp_df_step[hazard + "_perc"] = -9999.0
        if no_tresholds is False:
            shp_df_step[hazard + "_level"] = -9999.0

        logging.info(" ---> Loop through the alert zones for " + hazard + " risk...")

        for index, row in shp_df_step.iterrows():
            logging.info(" ----> Computing zone " + str(index +1) + " of " + str(len(shp_df_step)))
            bbox = row["geometry"].bounds
            clipped_pop = rx.open_rasterio(impact_dict["exposed_map"][impact_dict["exposed_element"]]).rio.clip_box(minx=bbox[0],miny=bbox[1],maxx=bbox[2],maxy=bbox[3])
            clipped_pop.values[clipped_pop.values<0] = 0
            lon_bbox = clipped_pop.x.values
            lat_bbox = clipped_pop.y.values
            alert_bbox = mosaic_flood_map.reindex({"x":lon_bbox, "y":lat_bbox}, method="nearest")
            country_bbox = rasterize([(row['geometry'], index+1)], {"lon":lon_bbox, "lat":lat_bbox})

            weigth_map = np.where((country_bbox==index+1),alert_bbox.values,np.nan)
            aff_people = np.nansum(weigth_map * np.squeeze(clipped_pop.values) * (row[impact_dict["lack_coping_capacity_col"]] / 10))
            tot_people = np.nansum(np.where(country_bbox == index + 1, np.squeeze(clipped_pop.values), np.nan))
            if tot_people == 0:
                impact_rate = 0
            else:
                impact_rate =  aff_people / tot_people

            if no_tresholds is False:
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

            shp_df_step.at[index, hazard + "_tot"] = aff_people
            shp_df_step.at[index, hazard + "_perc"] = impact_rate
            shp_df_step.at[index, "stock"] = tot_people

        logging.info(" ---> Loop through the alert zones for " + hazard + " risk...DONE")

        logging.info(" ---> Save shapefile for " + hazard + " risk...")
        shp_df_step.to_file(out_file_shp)
        logging.info(" ---> Save shapefile for " + hazard + " risk...DONE")

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
def extract_max_hmc_results(out_hmc_path, date_start, date_end):
    first_step = True
    for time_now in pd.date_range(date_start, date_end, freq="H"):
        file = os.path.join(out_hmc_path, "hmc.output-grid.{dateout}.nc".format(dateout=time_now.strftime("%Y%m%d%H%M")))
        if os.path.isfile(file + ".gz"):
            os.system("gunzip -k " + file + ".gz || true")
        try:
            file_now = xr.open_dataset(file)
            map_now = file_now["Discharge"].values
        except:
            logging.error(" ERROR! Output file " + file + " not found!")
            raise FileNotFoundError
        os.system("rm " + file)
        
        if first_step:
            dis_max = map_now
            mask = np.where(file_now["SM"].values < 0, 0, 1)
            lon = np.unique(file_now["Longitude"].values)
            lat = np.unique(file_now["Latitude"].values)
            first_step = False
        else:
            dis_max = np.maximum(dis_max, map_now)
    return dis_max, mask, lon, lat
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

# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
def make_tif(val, lon, lat, out_filename, crs='epsg:4326', nodata=-9999, dtype='float32'):
    out_ds = xr.DataArray(val, dims=["y", "x"], coords={"y": lat, "x": lon})
    out_ds = out_ds.where(out_ds != nodata).rio.write_crs(crs, inplace=True).rio.write_nodata(nodata, inplace=True)
    out_ds.values = out_ds.values.astype(dtype)
    out_ds.rio.to_raster(out_filename, driver="GTiff", crs='EPSG:4326', height=len(lat), width=len(lon), dtype=out_ds.dtype, compress="DEFLATE")

# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
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

def check_raster(df):
    if df.y.values[2] < df.y.values[1]:
        oriented_df_value = np.flipud(df.squeeze().values)
    else:
        oriented_df_value = df.squeeze().values
    return oriented_df_value

# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Call script from external library
if __name__ == "__main__":
    main()
# ----------------------------------------------------------------------------
