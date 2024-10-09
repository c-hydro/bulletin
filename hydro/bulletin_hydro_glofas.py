"""
bulletin - hydro - GLOFAS
__date__ = '20240820'
__version__ = '2.1.0'
__author__ =
        'Andrea Libertino (andrea.libertino@cimafoundation.org',
__library__ = 'bulletin'
General command line:
### python fp_bulletin_hydro_glofas.py -settings_file "settings.json" -time "YYYY-MM-DD HH:MM"
Version(s):
20240820 (2.1.0) --> Update to new Copernicus CDS
                     Fixex bug for maps weighted less than 1

20240517 (2.0.0) --> Update to GloFAS v4
20220511 (1.3.0) --> Added production of mosaic flood area map
20220329 (1.2.0) --> Merged hazard classes for rearcompatibility
20220324 (1.1.0) --> Add impact-based assessment
20211111 (1.0.0) --> Beta release for Africa Continental Watch
"""
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Complete library
import cdsapi
import os
import numpy as np
import datetime as dt

import pandas as pd
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
import rioxarray as rx
from copy import deepcopy

# -------------------------------------------------------------------------------------
# Script Main
def main():
    # -------------------------------------------------------------------------------------
    # Version and algorithm information
    alg_name = 'bulletin - Hydrological warning with GLOFAS '
    alg_version = '2.1.0'
    alg_release = '2024-08-20'
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
    output_shape_hazard_empty = data_settings['data']['dynamic']['outcome']['folder_shape_hazard']
    output_shape_impacts_empty = data_settings['data']['dynamic']['outcome']['folder_shape_impacts']
    ancillary_path_empty = data_settings['data']['dynamic']['ancillary']['folder']

    dict_empty = data_settings['algorithm']['template']
    dict_filled = dict_empty.copy()

    for key in dict_empty.keys():
        dict_filled[key] = date_now.strftime(dict_empty[key])

    paths["output"] = output_path_empty.format(**dict_filled)
    paths["output_shape_hazard"] = output_shape_hazard_empty.format(**dict_filled)
    paths["output_shape_impacts"] = output_shape_impacts_empty.format(**dict_filled)
    paths["ancillary"] = ancillary_path_empty.format(**dict_filled)
    paths["ancillary_raw"] = os.path.join(paths["ancillary"],"raw","")

    os.makedirs(paths["output"], exist_ok=True)
    os.makedirs(paths["output_shape_hazard"], exist_ok=True)
    os.makedirs(paths["output_shape_impacts"], exist_ok=True)
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
    if data_settings["algorithm"]["flags"]["download_frc"]:
        logging.info(" ---> Download from CDS...")
        nc_avg_name = download_from_cds(date_now, dict_filled, data_settings, paths)
    else:
        ancillary_avg_path = os.path.join(paths["ancillary"], "average")
        nc_avg_name = os.path.join(ancillary_avg_path, "glofas_fc_avg_time_{step}.nc")
        logging.info(" ---> Use already downloded forecast: " + nc_avg_name)
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
            area = rx.open_rasterio(data_settings['data']['static']['area']).reindex(
                {"x": dis_max.lon.values, "y": dis_max.lat.values}, method='nearest').squeeze()
            mask = np.where(area>=data_settings["data"]["dynamic"]["thresholds"]["area_km2"],1,0)
            alert_map = np.ones(area.shape)
            alert_max = np.ones(area.shape)
            first_step = False

        for val, rp in enumerate(data_settings["data"]["static"]["discharge_thresholds"]["return_periods"], start=2):
            th_map = rx.open_rasterio(os.path.join(data_settings["data"]["static"]["discharge_thresholds"]["folder"],
                                                   data_settings["data"]["static"]["discharge_thresholds"][
                                                       "file_name"]).format(domain=None, return_period=rp)).reindex(
                {"x": dis_max.lon.values, "y": dis_max.lat.values}, method='nearest').squeeze()
            th_map.values[th_map.values <= 0] = np.inf
            alert_map = np.where((dis_max.values >= th_map.values) & (dis_max.values >= data_settings["data"]["dynamic"]["thresholds"]["discharge_min"]), val, alert_map)
        alert_level_days[step] = np.where(mask == 1, alert_map, 0)

        alert_max = np.maximum(alert_max,alert_level_days[step])

    logging.info(" ----> Save discharge maps...")
    out_map = xr.DataArray(alert_max, dims=["lat","lon"], coords={"lon": dis_max.lon.values, "lat": dis_max.lat.values})
    out_file_tif = os.path.join(paths["output"], data_settings['data']['dynamic']['outcome']['file_name_levels']).format(**dict_filled)
    out_file_flood_tif = os.path.join(paths["output"], data_settings['data']['dynamic']['outcome']['file_name_flood_map']).format(**dict_filled)
    th_levels = out_map.copy()

    if data_settings["algorithm"]["flags"]["convert_hazard_classes"]:
        logging.info(" ----> Convert hazard classes")
        haz_class_in = out_map.values
        haz_class_out = np.ones(haz_class_in.shape)
        dict_conversion = data_settings["data"]["hazard"]["conversion_table"]
        for class_out in dict_conversion.keys():
            classes_in = dict_conversion[class_out]
            haz_class_out[np.isin(haz_class_in,classes_in)] = int(class_out)
        out_map.values = haz_class_out

    out_map.to_netcdf(os.path.join(paths["ancillary"], "temp.nc"))
    os.system("gdal_translate " + os.path.join(paths["ancillary"], "temp.nc") + " -a_srs EPSG:4326 " + out_file_tif)
    logging.info(" ----> Save discharge maps...DONE")

    logging.info(" ---> Classify discharge maps...DONE")
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Assign hazard and impact levels
    logging.info(" ---> Assign flood country warning levels...")
    shp_fo = data_settings["data"]["static"]["warning_regions"]
    shp_df = gpd.read_file(shp_fo)
    #th_levels = xr.open_rasterio(out_file_tif).squeeze().rename({"x": "lon", "y": "lat"})

    if data_settings["algorithm"]["flags"]["hazard_assessment"]:
        shp_df_hydro_model = shp_df.copy()
        shp_df_hydro_model["level_GLOFAS"] = -9999.0
        out_file_shp = os.path.join(paths["output_shape_hazard"],
                                    data_settings['data']['dynamic']['outcome']['file_name_shape_hazard'].format(
                                        **dict_filled))
        classify_warning_levels_pure_hazard(shp_df_hydro_model, th_levels, out_file_shp,
                                            data_settings["data"]["dynamic"]["thresholds"]["min_warning_pixel"])

    if data_settings["algorithm"]["flags"]["impact_assessment"]:
        shp_df_hydro_model = shp_df.copy()
        shp_df_hydro_model["GLfl_level"] = -9999.0
        out_file_shp = os.path.join(paths["output_shape_impacts"],
                                    data_settings['data']['dynamic']['outcome']['file_name_shape_impacts'].format(
                                        **dict_filled))
        impact_dict = data_settings["data"]["impacts"]
        impact_dict["return_periods"] = data_settings["data"]["static"]["discharge_thresholds"]["return_periods"]
        logging.info( "--> Mosaic flood maps..")
        impact_dict["temp_folder"] = os.path.join(paths["ancillary"],"flood_maps","")
        os.makedirs(impact_dict["temp_folder"], exist_ok=True)
        mosaic_flood_map, mosaic_weigthed_map = create_flood_map(th_levels, impact_dict)
        logging.info(" --> Write flood map...")
        os.system('gdal_calc.py -A ' + mosaic_flood_map + " --outfile=" + out_file_flood_tif + ' --calc="A/A" --type=Byte --NoDataValue=0 --co="COMPRESS=DEFLATE" --overwrite --quiet')
        logging.info("--> Classify warning levels..")
        classify_warning_levels_impact_based(shp_df_hydro_model, mosaic_weigthed_map, out_file_shp, impact_dict)
        logging.info("--> Classify warning levels..DONE")


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
# Method to classify warning levels on a pure-hazard base
def classify_warning_levels_pure_hazard(shp_df_hydro_model, th_levels, output_file, min_warn_pixels):
        shapes = [(shape, n) for n, shape in enumerate(shp_df_hydro_model.geometry)]
        ds = xr.Dataset(coords={'lon': th_levels['lon'].values, 'lat': th_levels['lat'].values})
        ds['states'] = rasterize(shapes, ds.coords)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            logging.info(" ----> Loop through the alert zones...")
            for index, row in shp_df_hydro_model.iterrows():
                val_max = np.nanmax(th_levels.where(ds['states'] == index))
                while val_max > 1:
                    tot_over = np.count_nonzero(th_levels.where(ds['states'] == index) >= val_max)
                    if tot_over >= min_warn_pixels:
                        break
                    else:
                        val_max = val_max - 1
                shp_df_hydro_model["level_GLOFAS"].at[index] = val_max
            logging.info(" ----> Loop through the alert zones...DONE")

        logging.info(" ----> Write hazard shapefile...")
        shp_df_hydro_model.to_file(output_file)
# -------------------------------------------------------------------------------------

def intersect_aoi(mosaic_flood_map, flood_map_level_name, aoi_map, level, codes_in):
    logging.info(' ---> Open flood map for level')
    flood_map_level = rx.open_rasterio(flood_map_level_name, cache=False).rio.clip_box(*aoi_map.rio.bounds()).squeeze()  # .astype(np.int16)
    logging.info(' ---> Edit ranges')
    flood_map_level = flood_map_level.reindex_like(aoi_map, method='nearest')
    # if flood_map_level is > 99999 or <0 set to 0, else set to 1
    flood_map_level.values = np.where((flood_map_level.values > 99999) | (flood_map_level.values <= 0) | np.isnan(flood_map_level.values), 0, 1)

    if level == 2:
        logging.info(' ---> Create mosaic map')
        mosaic_flood_map = (flood_map_level.copy() * 0).astype(np.int16)

    #if level in np.unique(th_levels_reindex.values):
    # select the codes in codes_per_level[level] that are in the domain according to the conversion table
    # set to 0 the codes of ai_map that are not in codes_in
    logging.info(' ---> Search values in AOI')
    aoi_in = deepcopy(aoi_map)
    aoi_in.values = np.where(np.isin(aoi_map.values, codes_in["hydro"].values), 1, 0)
    logging.info(' ---> Mosaicing....')
    mosaic_flood_map += flood_map_level.astype(np.int16) * level * aoi_in.astype(np.int16)

    return mosaic_flood_map


# -------------------------------------------------------------------------------------
# Mosaic flood maps
def create_flood_map(th_levels, impact_dict):
    logging.info(" --> Tailoring flood map")
    decode_map = rx.open_rasterio(impact_dict["decode_map"]).squeeze()
    decode_map.values[decode_map.values <0] = -1
    th_levels_reindex = th_levels.reindex({"lon":decode_map.x.values, "lat":decode_map.y.values}, method='nearest')
    conversion_table = pd.read_csv(impact_dict["aoi"]["decode_table"]["filename"], sep=",", usecols=[impact_dict["aoi"]["decode_table"]["col_flood"], impact_dict["aoi"]["decode_table"]["col_hydro"], impact_dict["aoi"]["decode_table"]["col_domain"]], names=["flood", "hydro", "domain"], header=0)
    conversion_table = conversion_table[conversion_table["domain"].isin(np.unique(impact_dict["aoi"]["domains"]))]

    logging.info(" --> Check area of interest to activate per level...")
    codes_per_level = {}

    for level, associated_rp in enumerate(impact_dict["flood_maps"]["associated_rp"], start=2):
        if level in np.unique(th_levels_reindex.values):
            codes_per_level[level] = [i for i in decode_map.values[th_levels_reindex == level] if i>-1 and i in conversion_table["hydro"].values]
        else:
            codes_per_level[level] = []

    logging.info(" --> Looping trough domains for making flood map...")
    for aoi in impact_dict["aoi"]["domains"]:
        logging.info(' ---> Computing domain ' + str(aoi))
        aoi_map = impact_dict["aoi"]["domain_map"].format(domain=aoi)
        aoi_map = rx.open_rasterio(aoi_map, cache=False).squeeze()   #.astype(np.int32)
        aoi_map.rio.write_nodata(-9999, inplace=True)
        aoi_map.values[(aoi_map.values < 0) | (np.isnan(aoi_map.values))] = 0
        aoi_maps = {}

        if aoi_map.sizes["x"] * aoi_map.sizes["y"] > 1000000000:
            logging.info(" --> Very large domain, split in 4 parts")
            aoi_maps[0] = aoi_map.isel(x=slice(0, int(aoi_map.sizes["x"]/2)), y=slice(0, int(aoi_map.sizes["y"]/2)))
            aoi_maps[1] = aoi_map.isel(x=slice(int(aoi_map.sizes["x"]/2), aoi_map.sizes["x"]), y=slice(0, int(aoi_map.sizes["y"]/2)))
            aoi_maps[2] = aoi_map.isel(x=slice(0, int(aoi_map.sizes["x"]/2)), y=slice(int(aoi_map.sizes["y"]/2), aoi_map.sizes["y"]))
            aoi_maps[3] = aoi_map.isel(x=slice(int(aoi_map.sizes["x"]/2), aoi_map.sizes["x"]), y=slice(int(aoi_map.sizes["y"]/2), aoi_map.sizes["y"]))
        elif aoi_map.sizes["x"] * aoi_map.sizes["y"] > 800000000:
            logging.info(" --> Large domain, split in 3 parts")
            aoi_maps[0] = aoi_map.isel(x=slice(0, int(aoi_map.sizes["x"]/2)), y=slice(0, int(aoi_map.sizes["y"]/2)))
            aoi_maps[1] = aoi_map.isel(x=slice(int(aoi_map.sizes["x"]/2), aoi_map.sizes["x"]), y=slice(0, int(aoi_map.sizes["y"]/2)))
            aoi_maps[2] = aoi_map.isel(x=slice(0, int(aoi_map.sizes["x"]/2)), y=slice(int(aoi_map.sizes["y"]/2), aoi_map.sizes["y"]))
        elif aoi_map.sizes["x"] * aoi_map.sizes["y"] > 600000000:
            logging.info(" --> Quite large domain, split in 2 parts")
            aoi_maps[0] = aoi_map.isel(x=slice(0, int(aoi_map.sizes["x"]/2)), y=slice(0, aoi_map.sizes["y"]))
            aoi_maps[1] = aoi_map.isel(x=slice(int(aoi_map.sizes["x"]/2), aoi_map.sizes["x"]), y=slice(0, aoi_map.sizes["y"]))
        else:
            aoi_maps[0] = aoi_map
            logging.info(" --> Small domain, process as a whole")

        for group in aoi_maps.keys():
            logging.info(" ---> Processing part " + str(group+1) + " of " + str(len(aoi_maps)))
            mosaic_flood_map = None
            for level, associated_rp in enumerate(impact_dict["flood_maps"]["associated_rp"], start=2):
                logging.info(' ----> Import hazard level ' + str(level))
                logging.info(' ----> Find codes in')
                codes_in = conversion_table.loc[(conversion_table["hydro"].isin(codes_per_level[level])) & (conversion_table["domain"] == aoi)]
                # Load and clip the flood map
                if level == 2 or len(codes_in) > 0:
                    flood_map_level_name = impact_dict["flood_maps"]["file_name"].format(return_period=str(associated_rp), domain=aoi)
                    mosaic_flood_map = intersect_aoi(mosaic_flood_map, flood_map_level_name, aoi_maps[group], level, codes_in)
                else:
                    logging.info(' ---> No codes in domain ' + str(aoi) + ' for level ' + str(level))
            mosaic_flood_map.rio.write_crs("epsg:4326", inplace=True)
            mosaic_flood_map.rio.write_nodata(0, inplace=True)
            logging.info(" --> All levels processed for the domain, now apply the weights...")
            mosaic_weigthed_map = mosaic_flood_map.copy().astype(np.float32)
            mosaic_weigthed_map.values = mosaic_weigthed_map.values * 0.0
            for key, level in enumerate(impact_dict["weight_hazard_levels"], start=2):
                mosaic_weigthed_map.values = np.where(mosaic_flood_map.values == key, level, mosaic_weigthed_map.values)
            logging.info(' ---> Computing domain ' + str(aoi) + '...DONE')

            logging.info(' ---> Save mosaic flood map in the ancillary folder')
            mosaic_flood_map.astype(np.int16).rio.to_raster(os.path.join(impact_dict["temp_folder"], "flood_map_{domain}_{key_group}.tif").format(domain=aoi, key_group=str(group)),compress='DEFLATE', dtype='int16')
            mosaic_weigthed_map.astype(np.float32).rio.to_raster(os.path.join(impact_dict["temp_folder"], "weight_map_{domain}_{key_group}.tif").format(domain=aoi, key_group=str(group)),compress='DEFLATE', dtype='float32')

    logging.info(" --> Merging the flood maps")
    os.system("gdal_merge.py -o " + os.path.join(impact_dict["temp_folder"], "flood_map_merged.tif") + " " + impact_dict["temp_folder"] + "flood_map_*.tif -ot Int16 -co COMPRESS=DEFLATE -co BIGTIFF=YES -n 0")

    logging.info(" --> Merging the weight maps")
    os.system("gdal_merge.py -o " + os.path.join(impact_dict["temp_folder"], "weight_map_merged.tif") + " " + impact_dict["temp_folder"] + "weight_map_*.tif -ot Float32 -co COMPRESS=DEFLATE -co BIGTIFF=YES -n 0")

    return os.path.join(impact_dict["temp_folder"], "flood_map_merged.tif"), os.path.join(impact_dict["temp_folder"], "weight_map_merged.tif") #rx.open_rasterio(os.path.join(impact_dict["temp_folder"], "flood_map_merged.tif")).squeeze()

# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Classify a shapefile of countries with an alert level map with an impact-based approach
def classify_warning_levels_impact_based(shp_df, mosaic_flood_map, out_file_shp, impact_dict):
    hazard = "GLfl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        shp_df_step = shp_df.copy()

        shp_df_step["pop_total"] = -9999.0
        shp_df_step[hazard + "AffPpl"] = -9999.0
        shp_df_step[hazard + "AffPrc"] = -9999.0
        shp_df_step[hazard + "_level"] = -9999.0

        logging.info(" ---> Loop through the alert zones for " + hazard + " risk...")

        for index, row in shp_df_step.iterrows():
            logging.info(" ----> Computing zone " + str(index +1) + " of " + str(len(shp_df_step)))
            bbox = row["geometry"].bounds
            try:
                clipped_flood = rx.open_rasterio(mosaic_flood_map).rio.clip_box(minx=bbox[0],miny=bbox[1],maxx=bbox[2],maxy=bbox[3])
                clipped_pop = rx.open_rasterio(impact_dict["exposed_map"]).rio.clip_box(minx=bbox[0],miny=bbox[1],maxx=bbox[2],maxy=bbox[3])
            except rx.exceptions.NoDataInBounds:
                logging.warning("WARNING! No data in selected bounding box! Is the geometry out of the raster domain?")
                shp_df_step.at[index, hazard + "_level"] = 0
                shp_df_step.at[index, hazard + "AffPpl"] = 0
                shp_df_step.at[index, hazard + "AffPrc"] = 0
                shp_df_step.at[index, "pop_total"] = 0
                continue
            clipped_pop.values[clipped_pop.values<0] = 0
            lon_bbox = clipped_pop.x.values
            lat_bbox = clipped_pop.y.values
            alert_bbox = clipped_flood.reindex({"x":lon_bbox, "y":lat_bbox}, method="nearest")
            country_bbox = rasterize([(row['geometry'], index+1)], {"lon":lon_bbox, "lat":lat_bbox})

            weigth_map = np.where((country_bbox==index+1),alert_bbox.values,np.nan)
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
         'system_version': ['operational'],
         'variable': 'river_discharge_in_the_last_24_hours',
         'hydrological_model': ['lisflood'],
         'format': 'grib2',
         'product_type': ['control_forecast', 'ensemble_perturbed_forecasts'],
         'year': [str(date_now.year)],
         'month': [str(date_now.month).zfill(2)],
         'day': [str(date_now.day).zfill(2)],
         'leadtime_hour': time_steps,
         'area': bounding_box,
         'download_format': 'unarchived'
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
