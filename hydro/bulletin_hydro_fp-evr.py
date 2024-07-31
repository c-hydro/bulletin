"""
bulletin - hydro - floodPROOFS
__date__ = '20240731'
__version__ = '1.0.1'
__author__ =
        'Andrea Libertino (andrea.libertino@cimafoundation.org',
__library__ = 'bulletin'
General command line:
### python bulletin_hydro_fp-evr.py -settings_file "settings.json" -time "YYYY-MM-DD HH:MM"
Version(s):
20240731 (1.0.1) --> Add the possibility to have multiple sub-categories for each category of exposed elements
                     Add the possibility to have a multiplier for each category of exposed elements
                     Fix several bugs
20240715 (1.0.0) --> Beta release
"""
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Complete library
import os
import pickle
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
# -------------------------------------------------------------------------------------
# Script Main
def main():
    # -------------------------------------------------------------------------------------
    # Version and algorithm information
    alg_name = 'bulletin - Hydrological warning with FloodProofs and modified IBF approach'
    alg_version = '1.0.1'
    alg_release = '2024-07-31'
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
    flood_maps_ready = {}
    missing_domains = []

    hazard_dict = data_settings["data"]["hazard"]
    impact_dict = data_settings["data"]["impacts"]
    #impact_dict["return_periods"] = data_settings["data"]["static"]["discharge_thresholds"]["return_periods"]

    # If exists, load the levels of the sections from the pickle file, if not, create an empty dictionary
    if os.path.isfile(os.path.join(paths["ancillary"], "levels_sections.pkl")):
        with open(os.path.join(paths["ancillary"], "levels_sections.pkl"), "rb") as f:
            levels_sections = pickle.load(f)
    else:
        levels_sections = {}

    for domain in data_settings["data"]["dynamic"]["fp"]["domains"]:
        logging.info(" --> Compute fp forecast with date: " + date_now.strftime("%Y-%m-%d %H:%M") + " for domain " + domain)
        dict_filled["domain"] = domain

        # Define filenames and check existence of flood ancillary map
        static_hmc_path = data_settings["data"]["dynamic"]["fp"]["static_data_folder"].format(**dict_filled)
        out_hmc_path = data_settings["data"]["dynamic"]["fp"]["results_folder"].format(**dict_filled)
        out_flood_ancillary_map = os.path.join(paths["ancillary"], data_settings["data"]["dynamic"]["ancillary"]["file_name_flood"]).format(**dict_filled)

        if os.path.isfile(out_flood_ancillary_map) and not data_settings["algorithm"]["flags"]["recompute_domains"]:
            logging.info(" --> Domain " + domain + " has already been computed! SKIPPING!")
            flood_maps_ready[domain] = out_flood_ancillary_map
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
        logging.info(" ----> Assign flood hazard level...")

        # Load static map for filtering
        area = check_raster(xr.open_rasterio(os.path.join(static_hmc_path, "{domain}.area.txt").format(**dict_filled)))
        areacell = check_raster(xr.open_rasterio(os.path.join(static_hmc_path, "{domain}.areacell.txt").format(**dict_filled)))
        area_km = (area * areacell / (10**6)).squeeze()
        mask = np.where((river_mask==1) & (area_km>=data_settings["data"]["dynamic"]["thresholds"]["area_km2"]),1,0)

        # Load gev parameters and calculate T
        params = {}
        for par in [1,2,3]:
            params["theta" + str(par)] = check_raster(xr.open_rasterio(data_settings["data"]["static"]["gev_parameters"]["theta" + str(par)].format(**dict_filled)))

        rp = calculate_return_period(dis_max, params)
        rp = np.where(rp > np.max(np.array(hazard_dict["return_period"])),
                      np.max(np.array(hazard_dict["return_period"])), rp)
        # Round the return period to the lower integer
        rp = np.where((mask == 1) & (np.isnan(rp) == 0), np.round(rp), 0).astype(np.int32)
        make_tif(rp, lon, lat, os.path.join(paths["output_maps"],data_settings["data"]["dynamic"]["outcome"]["file_name_return_period"]).format(**dict_filled), dtype='int32', nodata=0)
        logging.info(" ----> Assign flood hazard level...DONE")

        # Mosaic weigth map for the domain
        logging.info(" ----> Merge flood map...")
        impact_dict["domain"] = domain
        hazard_dict["domain"] = domain

        flood_map, lat_mosaic, lon_mosaic, levels_sections[domain] = create_flood_map(rp, hazard_dict)
        make_tif(flood_map.values.astype(np.float32), lon_mosaic, lat_mosaic, out_flood_ancillary_map)
        logging.info(" ----> Merge flood map...DONE")

        flood_maps_ready[domain] = out_flood_ancillary_map
        # save a pickle file with the levels of the sections in the ancillary folder
        with open(os.path.join(paths["ancillary"], "levels_sections.pkl"), "wb") as f:
            pickle.dump(levels_sections, f)

    # Check for all domains to be computed
    if len(flood_maps_ready.keys()) < len(data_settings["data"]["dynamic"]["fp"]["domains"]):
        logging.warning(" --> WARNING! Domains " + ','.join(missing_domains) + " are missing")
        if not data_settings["algorithm"]["flags"]["compute_partial_results"]:
            logging.error(" --> ERROR! Not all the domains has been computed!")
            raise RuntimeError

    # Merge maps of all the domains
    merged_flood_map = xr.open_rasterio(flood_maps_ready[list(flood_maps_ready.keys())[0]]) * 0
    for val in flood_maps_ready.keys():
        merged_flood_map.values = merged_flood_map.values + xr.open_rasterio(flood_maps_ready[val]).values

    logging.info(" ----> Save merged flood map...")
    dict_filled['domain'] = "merged"
    out_file_flood_tif = os.path.join(paths["output_maps"], data_settings['data']['dynamic']['outcome']['file_name_flood_map']).format(**dict_filled)

    make_tif(merged_flood_map.values.astype(np.float32).squeeze(), merged_flood_map.x.values, merged_flood_map.y.values, out_file_flood_tif, dtype='float32', nodata=-9999)

    logging.info(" ----> Save merged flood map...DONE")
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Calculate impacts
    logging.info(" ----> Calculate impacts...")
    # Load the pickle file with the levels of the sections
    with open(os.path.join(paths["ancillary"], "levels_sections.pkl"), "rb") as f:
        levels_sections = pickle.load(f)

    admin_shape = impact_dict["admin"]["filename"]
    impacts_shape = gpd.read_file(admin_shape)

    for exposed_element in impact_dict["admin"]["abs_column"].keys():
        impacts_shape[impact_dict["admin"]["abs_column"][exposed_element]] = impacts_shape[
            impact_dict["admin"]["abs_column"][exposed_element]].fillna(0)

    impacts_table = pd.DataFrame(index=impacts_shape[impact_dict["admin"]["join_column"]])
    for exposed_element in impact_dict["MUL"]:
        impacts_table["tot_" + exposed_element] = 0.0

    for domain in flood_maps_ready.keys():
        logging.info(" ----> Analyse domain " + domain)
        hydro_to_admin = pd.read_csv(impact_dict["hydro_to_admin"]["filename"].format(domain=domain), header=0)
        # check if there is any value of hydro_to_admin["admin"] that is not in the impacts_table.index. In this case, drop the rows in the hydro_to_admin dataframe and raise a warning listing the missing admin units
        missing_admin = hydro_to_admin[~hydro_to_admin["admin"].isin(impacts_table.index)]
        if len(missing_admin) > 0:
            logging.warning(" --> WARNING! The following admin units are not present in the shapefile: " + ', '.join([str(i) for i in missing_admin["admin"].values]))
            hydro_to_admin = hydro_to_admin[hydro_to_admin["admin"].isin(impacts_table.index)]
        for rp in levels_sections[domain]:
            logging.info(" ----> Merge return period " + str(rp))
            filtered_hydro_to_admin = hydro_to_admin[hydro_to_admin['hydro'].isin(levels_sections[domain][rp])]
            for exposed_element in impact_dict["MUL"]:
                logging.info(" ----> Analyse element " + exposed_element)
                for mul in filtered_hydro_to_admin["MUL"]:
                    if mul < 0:
                        continue
                    if type(impact_dict["MUL"][exposed_element]["files"]) == dict:
                        for sub_category in impact_dict["MUL"][exposed_element]["files"]:
                            for file in impact_dict["MUL"][exposed_element]["files"][sub_category]:
                                mul_file = file.format(mul=str(int(mul)), domain=domain, exposed_element=exposed_element)
                                if not os.path.isfile(mul_file):
                                    continue
                                impact_mul = pd.read_csv(mul_file, names=["rp", "abs", "std"], index_col=["rp"]).loc[rp, "abs"]
                                admin = filtered_hydro_to_admin[filtered_hydro_to_admin["MUL"] == mul]["admin"].values[0]
                                if admin in missing_admin["admin"].values:
                                    continue
                                if "multiplier" in impact_dict["MUL"][exposed_element]:
                                    multiplier = impact_dict["MUL"][exposed_element]["multiplier"]
                                else:
                                    multiplier = 1
                                impacts_table.at[admin, "tot_" + exposed_element] += impact_mul * multiplier
                                if "tot_" + exposed_element + "_" + sub_category not in impacts_table.columns:
                                    logging.info(" ----> Category " + exposed_element + " has sub-category " + sub_category)
                                    impacts_table["tot_" + exposed_element + "_" + sub_category] = 0.0
                                # check if "multiplier" is a key in impact_dict["MUL"][exposed_element]
                                impacts_table.at[admin, "tot_" + exposed_element + "_" + sub_category] += impact_mul * multiplier

                    elif type(impact_dict["MUL"][exposed_element]["files"]) == list:
                        mul_file_list = impact_dict["MUL"][exposed_element]["files"]
                        for file in mul_file_list:
                            mul_file = file.format(mul=str(int(mul)), domain=domain, exposed_element=exposed_element)
                            if not os.path.isfile(mul_file):
                                continue
                            impact_mul = pd.read_csv(mul_file, names=["rp", "abs", "std"], index_col=["rp"]).loc[rp, "abs"]
                            admin = filtered_hydro_to_admin[filtered_hydro_to_admin["MUL"] == mul]["admin"].values[0]
                            if admin in missing_admin["admin"].values:
                                continue
                            if "multiplier" in impact_dict["MUL"][exposed_element]:
                                multiplier = impact_dict["MUL"][exposed_element]["multiplier"]
                            else:
                                multiplier = 1
                            impacts_table.at[admin, "tot_" + exposed_element] += impact_mul * multiplier
                    else:
                        logging.error("ERROR! The type of impact_dict['MUL'][exposed_element] should be either a list (even a singular one) or a dictionary!")
                        raise ValueError
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Write the output shapefile
    for exposed_element in impact_dict["MUL"]:
        impact_shape_element = impacts_shape.copy()
        # Import the impact for the total element
        impact_shape_element["flood_tot"] = impact_shape_element[impact_dict["admin"]["join_column"]].map(impacts_table["tot_" + exposed_element])
        # Check for sub-categories
        for sub_category in impacts_table.columns:
            if sub_category.startswith("tot_" + exposed_element + "_"):
                impact_shape_element[sub_category.replace("_" + exposed_element,"")] = impacts_table[sub_category].values
        dict_filled["exposed_element"] = exposed_element
        # Check if the actual element is in the keys for which is possible to calculate the %
        if exposed_element in impact_dict["admin"]["abs_column"]:
            impact_shape_element["flood_perc"] = impact_shape_element["flood_tot"] / impact_shape_element[impact_dict["admin"]["abs_column"][exposed_element]]
            impact_shape_element["flood_perc"] = impact_shape_element["flood_perc"].fillna(0)
        if impact_dict["classification"]["category"] == exposed_element:
            logging.info(" ----> Classify impact levels according to " + exposed_element)
            impact_shape_element["flood_level"] = -9999.0
            for index, row in impact_shape_element.iterrows():
                impact_rate = row["flood_perc"]
                aff_people = row["flood_tot"]
                for risk_lev, (risk_th_abs, risk_th_rel) in enumerate(zip(impact_dict["classification"]["risk_thresholds"]["absolute"],
                                                                          impact_dict["classification"]["risk_thresholds"]["relative"]),
                                                                      start=1):
                    if risk_th_abs is None: risk_th_abs = 0
                    if risk_th_rel is None: risk_th_rel = 0

                    if risk_th_abs == 0 and risk_th_rel == 0:
                        logging.error(" ERROR! Either a relative or an absolute threshold value should be provided for each risk class!")
                        raise ValueError("Both absolute and relative trhesholds are none for class " + str(risk_lev))
                    elif impact_rate >= risk_th_rel and aff_people >= risk_th_abs:
                        risk = risk_lev
                    else:
                        break
                impact_shape_element.at[index, "flood_level"] = risk
        impact_shape_element.crs = "EPSG:4326"
        impact_shape_element.to_file(os.path.join(paths["output_shape"], data_settings['data']['dynamic']['outcome']['file_name_shape_impacts']).format(**dict_filled))
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
def calculate_return_period(Q,params):
    value = 1 - params["theta3"] * ((Q - params["theta1"]) / params["theta2"])
    value[value < 0] = 0
    y = np.where(params["theta3"] == 0, (Q - params["theta1"])/params["theta2"], (-1/params["theta3"]) * np.log(value))
    P = np.exp(-np.exp(-y))
    T = 1/(1-P)
    return T
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Mosaic flood maps
def create_flood_map(rp, hazard_dict):
    logging.info(" --> Tailoring flood map")
    section_map = xr.open_rasterio(hazard_dict["section_map"].format(domain=hazard_dict["domain"])).squeeze()
    section_map_values = check_raster(section_map)
    section_map_values[section_map_values < 0] = -1

    levels_sections = {}

    # check if all the rp_in_map are in the hazard_dict["return_period"] list, if not, replace them with the closest inferior value
    rp_in_map = np.unique(rp.flatten())
    for T in rp_in_map[rp_in_map > 1]:
        if T not in hazard_dict["return_period"]:
            closest = hazard_dict["return_period"][np.abs(hazard_dict["return_period"] - T).argmin()]
            rp[rp == T] = closest

    first_map = True
    for T in np.unique(rp.flatten()):
        if T <= 1 or np.isnan(T):
            pass
        else:
            logging.info(' ---> Import maps for RP ' + str(int(T)) + ' years')
            flood_map_level = xr.open_rasterio(hazard_dict["flood_maps"].format(return_period=str(int(T)).zfill(4)), cache=False).squeeze()
            flood_map_level.values = check_raster(flood_map_level)
            flood_map_level.values[flood_map_level.values > 999999] = 0
            flood_map_level.values[flood_map_level.values < 0] = 0
            if first_map:
                mosaic_flood_map = flood_map_level.copy()*0.0
                lat_mosaic = np.sort(flood_map_level.y.values)
                lon_mosaic = flood_map_level.x.values
                decode_map = check_raster(xr.open_rasterio(hazard_dict["decode_map"]).squeeze())
                first_map = False
            codes_in = section_map_values[rp==T]
            codes_in = codes_in[codes_in>0]
            levels_sections[T] = codes_in

            mask = np.isin(decode_map, codes_in)
            flood_map_level.values = flood_map_level.values * mask
            mosaic_flood_map += flood_map_level

    # if there is no overcome of the treshold produce a map full of zeroes
    if first_map is True:
        flood_map_level = xr.open_rasterio(hazard_dict["flood_maps"].format(return_period=str(int(np.min(hazard_dict["return_period"]))).zfill(4)),
                                           cache=False).squeeze()
        mosaic_flood_map = flood_map_level.copy() * 0.0
        lat_mosaic = np.sort(flood_map_level.y.values)
        lon_mosaic = flood_map_level.x.values

    return mosaic_flood_map.astype(np.float32), lat_mosaic, lon_mosaic, levels_sections

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
    out_ds.values = np.where(out_ds.values == nodata, nodata, out_ds.values.astype(dtype))
    out_ds = out_ds.rio.write_crs(crs, inplace=True).rio.write_nodata(nodata, inplace=True)
    out_ds.rio.to_raster(out_filename, driver="GTiff", crs='EPSG:4326', height=len(lat), width=len(lon), dtype=out_ds.dtype, compress="DEFLATE", nodata=nodata)

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
