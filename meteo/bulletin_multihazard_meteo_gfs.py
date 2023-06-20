"""
bulletin - meteo multihazard - GFS025
__date__ = '20230619'
__version__ = '1.2.2'
__author__ =
        'Andrea Libertino (andrea.libertino@cimafoundation.org',
__library__ = 'bulletin'
General command line:
### python fp_bulletin_meteo_gfs.py -settings_file "settings.json" -time "YYYY-MM-DD HH:MM"
Version(s):
20230619 (1.2.2) --> Added possibility of choose what variables to consider
20220615 (1.2.1) --> Fixed bug with realtive/absolute tresholds
20220202 (1.2.0) --> Add impact assessment
20211111 (1.1.0) --> Separate hydro components
                     Add backup procedure
20200326 (1.0.0) --> Beta release for FloodProofs Africa
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
    alg_name = 'bulletin - Multihazard meteo warning for GFS '
    alg_version = '1.2.2'
    alg_release = '2023-06-19'
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
    date_run = pytz.utc.localize(datetime.strptime(alg_time, "%Y-%m-%d %H:%M"))

    date_from = date_run - timedelta(hours=data_settings['data']['dynamic']['time']['past_time_search_window_h'])
    date_to = date_run + timedelta(hours=data_settings['data']['dynamic']['time']['future_time_search_window_h'])
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Create directories
    template_time_step = fill_template_time_step(data_settings['algorithm']['template'], date_run)

    ancillary_folder = data_settings['data']['dynamic']['ancillary']['folder'].format(**template_time_step)
    ancillary_file = os.path.join(ancillary_folder, data_settings['data']['dynamic']['ancillary']['file_name']).format(
        **template_time_step)
    ancillary_frc_file = os.path.join(ancillary_folder, "forecast.nc")

    output_folder = data_settings['data']['dynamic']['outcome']['folder'].format(**template_time_step)
    output_file = os.path.join(output_folder, data_settings['data']['dynamic']['outcome']['file_name']).format(
        **template_time_step)

    output_folder_shape_hazard = data_settings['data']['dynamic']['outcome']['folder_shape_hazard'].format(
        **template_time_step)
    output_file_shape_hazard = os.path.join(output_folder_shape_hazard,
                                            data_settings['data']['dynamic']['outcome'][
                                                'file_name_shape_hazard']).format(**template_time_step)

    output_folder_shape_impacts = data_settings['data']['dynamic']['outcome']['folder_shape_impacts'].format(
        **template_time_step)
    output_file_shape_impacts = os.path.join(output_folder_shape_impacts,
                                            data_settings['data']['dynamic']['outcome'][
                                                'file_name_shape_impacts']).format(**template_time_step)
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_folder_shape_impacts, exist_ok=True)
    os.makedirs(output_folder_shape_hazard, exist_ok=True)
    os.makedirs(ancillary_folder, exist_ok=True)
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
    # Collect data
    variable_names = [data_settings['data']['dynamic']['variables'][i]['name'] for i in
                      data_settings['data']['dynamic']['variables'].keys()]
    variables_dic = {}

    # -------------------------------------------------------------------------------------
    # Invalid settings
    if (data_settings["algorithm"]["settings"]["download_forecast_with_drops"] is True and data_settings["algorithm"]["settings"]["use_local_forecast_file"] is True) or \
        (data_settings["algorithm"]["settings"]["download_forecast_with_drops"] is False and data_settings["algorithm"]["settings"]["use_local_forecast_file"] is False):
        logging.error("ERROR! Select if download forecast with drops2 or use local forecast!")
        raise ValueError

    # Download with drops
    elif data_settings["algorithm"]["settings"]["download_forecast_with_drops"]:
        # Connect to drops2 for downloading data
        from drops2 import coverages
        from drops2.utils import DropsCredentials

        # Set up drops2 connection
        logging.info(" --> Set up drops2 connection...")
        drops_settings = data_settings['data']['dynamic']['drops2']
        if not all([drops_settings['DropsUser'], drops_settings['DropsPwd']]):
            netrc_handle = netrc.netrc()
            try:
                drops_settings['DropsUser'], _, drops_settings['DropsPwd'] = netrc_handle.authenticators(
                    drops_settings['DropsAddress'])
            except:
                logging.error(
                    ' --> Netrc authentication file not found in home directory! Generate it or provide user and password in the settings!')
                raise FileNotFoundError(
                    'Verify that your .netrc file exists in the home directory and that it includes proper credentials!')

        DropsCredentials.set(drops_settings['DropsAddress'], drops_settings['DropsUser'], drops_settings['DropsPwd'])
        logging.info(" --> Set up drops2 connection...DONE")

        # Download data
        data_id = drops_settings['DropsDataId']
        logging.info(" --> Download data from drops2 for " + data_id)

        logging.info(
            " ---> Time window for forecast file search : " + date_from.strftime("%Y%m%d%H%M") + " - " + date_to.strftime(
                "%Y%m%d%H%M"))
        model_dates_raw = coverages.get_dates(data_id, date_from.strftime("%Y%m%d%H%M"), date_to.strftime("%Y%m%d%H%M"))
        model_dates = [i for i in model_dates_raw if i <= date_to and i >= date_from]

        if len(model_dates) == 0:
            logging.error(" ---> ERROR! No forecast files available in the selected time window!")
            raise FileNotFoundError

        logging.info(" ---> " + str(len(model_dates)) + " forecast files found: " + ",".join(
            [i.strftime("%Y%m%d%H%M") for i in model_dates]))
        logging.info(" ---> Proceed with last forecast file available: " + model_dates[0].strftime("%Y%m%d%H%M"))

        date_ref = model_dates[0]
        forecast_end = date_ref + timedelta(hours=data_settings['data']['dynamic']['time']['forecast_length_h'] - 1)

        for type, variable in zip(data_settings['data']['dynamic']['variables'].keys(), variable_names):
            logging.info(" ---> Download variable: " + type + "...")
            logging.info(" ----> Variable name: " + variable)
            level = data_settings['data']['dynamic']['variables'][type]["level"]

            # Check on forecast integrity, only for first variable
            if variable == variable_names[0]:
                logging.info(" ----> Check forecast file integrity...")
                timeline = coverages.get_timeline(data_id, date_ref, variable, level)
                if len(timeline) == 0:
                    logging.error(" ---> ERROR! Forecast file has no time-step, verify file integrity!")
                    raise TypeError
                logging.info(" ----> Check forecast file integrity...DONE")

            date_selected = data_settings['data']['dynamic']['variables'][type]["date_selected"]

            logging.info(" ----> Get data from drops2...")

            data_drops = coverages.get_data(data_id, date_ref, variable, level, date_selected=date_selected)
            data_lon = np.unique(data_drops.longitude.values)
            data_lat = np.unique(data_drops.latitude.values)

            variables_dic[type] = xr.DataArray(dims=["time", "lat", "lon"],
                                               coords={"lon": data_lon, "lat": data_lat, "time": data_drops.time.values},
                                               data=data_drops[variable].values)
            logging.info(" ----> Get data from drops2...DONE")
            logging.info(" ---> Download variable: " + type + "...DONE")

    # -------------------------------------------------------------------------------------
    # Use local file
    elif data_settings["algorithm"]["settings"]["use_local_forecast_file"]:
        date_ref = deepcopy(date_run)
        forecast_end = date_ref + timedelta(hours=data_settings['data']['dynamic']['time']['forecast_length_h'] - 1)

        for type, variable in zip(data_settings['data']['dynamic']['variables'].keys(), variable_names):
            logging.info(" ---> Import local variable: " + type + "...")
            logging.info(" ----> Variable name: " + variable)

            data_drops = xr.open_dataset(data_settings['data']['dynamic']['variables'][type]["file"].format(
        **template_time_step))
            if data_drops == ancillary_frc_file:
                check_previous_dload = True
            data_lon = np.unique(data_drops.lon.values)
            if data_drops.lat.values[-1] < data_drops.lat.values[1]:
                data_lat = np.unique(data_drops.lat.values)[::-1]
            else:
                data_lat = np.unique(data_drops.lat.values)

            if variable == '10v' or variable == '10u':
                try:
                    data = data_drops[variable].loc[:, 10.0, :, :].values
                except:
                    data = data_drops[variable].values
            else:
                data = data_drops[variable].values

            variables_dic[type] = xr.DataArray(dims=["time", "lat", "lon"],
                                               coords={"lon": data_lon, "lat": data_lat,
                                                       "time": data_drops.time.values}, data=data)
            logging.info(" ----> Get local data...DONE")
            logging.info(" ---> Download variable: " + type + "...DONE")

        logging.info(" ---> Use of local available forecast...DONE")
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Postprocess data
    logging.info(" --> Postprocess forecast dataframe...")

    data = xr.Dataset(variables_dic)

    if debug_mode and not check_previous_dload:
            data.to_netcdf(ancillary_frc_file)

    logging.info(" ---> Crop forecast over domain...")
    min_lon = data_settings["data"]["dynamic"]["bbox"]['lon_left']
    max_lon = data_settings["data"]["dynamic"]["bbox"]['lon_right']
    min_lat = data_settings["data"]["dynamic"]["bbox"]['lat_bottom']
    max_lat = data_settings["data"]["dynamic"]["bbox"]['lat_top']

    lon = deepcopy(data['lon'].values)
    lon[lon > 180] = lon[lon > 180] - 360
    data = data.assign_coords(lon=lon)

    mask_lon = (data.lon >= min_lon) & (data.lon <= max_lon)
    mask_lat = (data.lat >= min_lat) & (data.lat <= max_lat)
    data = data.where(mask_lon & mask_lat, drop=True).sortby("lon")
    logging.info(" ---> Crop forecast over domain...DONE")

    logging.info(" ---> Round time steps to 3-hourly exact resolution...")
    time_round = np.array([pd.Timestamp(i).round('60min').to_pydatetime() for i in data.time.values])
    data = data.assign_coords(time=time_round)
    logging.info(" ---> Round time steps to 3-hourly exact resolution...DONE")

    logging.info(" --> Postprocess forecast dataframe...DONE!")
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Postprocess variables
    logging.info(" --> Postprocess variables...")

    processed_variables = {}

    if "rain" in data_settings["algorithm"]["settings"]["hazards"]:
        if not "cumulated_var" in data_settings["data"]["dynamic"]["variables"]["rain"].keys():
            logging.warning(" --> Cumulation of rainfall forecast not specified! Rainfall is considered as cumulated")
            data_settings["data"]["dynamic"]["variables"]["rain"]["cumulated_var"] = True

        logging.info(" ---> Decumulate accumulated rainfall...")
        if data_settings["data"]["dynamic"]["variables"]["rain"]["cumulated_var"] is True:
            ds_decum = xr.concat([data['rain'][0:1,:,:], data['rain'].diff("time")], "time")
            logging.info(" ---> Decumulate accumulated rainfall ...DONE!")
        else:
            logging.info(" ---> Rainfall is already at hourly time step! Skipping")
            ds_decum = data['rain']
        logging.info(" ---> Cumulate with rolling window...")
        processed_variables["rain"] = ds_decum.rolling(time=8, min_periods=8, center=True).sum().shift(time=-1)
        logging.info(" ---> Cumulate with rolling window...DONE!")


    if "wind" in data_settings["algorithm"]["settings"]["hazards"]:
        logging.info(" ---> Merge wind components...")
        processed_variables["wind"] = np.sqrt((data['u-wind']**2)+(data['v-wind']**2))
        logging.info(" ---> Merge wind components...DONE!")

    ds_out = xr.Dataset(processed_variables)
    ds_out_daily = ds_out.resample({"time":'D'}, skipna=True).max()
    ds_out_daily = ds_out_daily.reindex(time=pd.date_range(date_ref,forecast_end , freq='D').tz_localize(None), method='nearest')
    ds_out_daily.to_netcdf(ancillary_file)
    alert_daily = deepcopy(ds_out_daily)
    logging.info(" --> Postprocess variables...DONE")
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Classify rainfall alert level
    if "rain" in data_settings["algorithm"]["settings"]["hazards"]:
        logging.info(" --> Classify rainfall alert level...")

        thresholds = data_settings['data']['static']['rain_thresholds']['quantiles']
        limits = data_settings['data']['static']['rain_thresholds']['limits']
        threshold_map = {}

        for th in thresholds:
            threshold_map[str(th)] = xr.open_rasterio(
                os.path.join(data_settings['data']['static']['rain_thresholds']['folder'],
                             data_settings['data']['static']['rain_thresholds']['file_name']).format(quantile=str(th)))
        df_threshold = xr.Dataset(threshold_map).squeeze().reindex(
            {"x": ds_out_daily["lon"].values, "y": ds_out_daily["lat"].values}, method="nearest").rename_dims(
            {"x": "lon", "y": "lat"})

        for th, lims in zip(thresholds, limits):
            df_threshold[str(th)] = xr.where(df_threshold[str(th)] < lims[0], lims[0], df_threshold[str(th)])
            df_threshold[str(th)] = xr.where(df_threshold[str(th)] > lims[1], lims[1], df_threshold[str(th)])

        # for debug purposes only
        if debug_mode:
            df_threshold.to_netcdf(os.path.join(ancillary_folder, "thresholds_rain.nc"))

        alert_maps = np.where(ds_out_daily["rain"].values >= 0, 1, np.nan)
        for val, th in enumerate(thresholds, start=2):
            alert_maps = np.where(ds_out_daily["rain"].values >= df_threshold[str(th)].values, val, alert_maps)

        alert_daily["rain"].values = alert_maps
        logging.info(" --> Classify rainfall alert level...DONE")
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    if "wind" in data_settings["algorithm"]["settings"]["hazards"]:
        # Classify wind alert level
        logging.info(" --> Classify wind alert level...")

        thresholds = data_settings['data']['static']['wind_thresholds']['values']
        alert_maps = np.where(ds_out_daily["wind"].values >= 0, 1, np.nan)

        for val, th in enumerate(thresholds, start=2):
            alert_maps = np.where(ds_out_daily["wind"].values >= th, val, alert_maps)

        alert_daily["wind"].values = alert_maps
        logging.info(" --> Classify wind alert level...DONE")
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Apply sea mask
    if data_settings["algorithm"]["flags"]["mask_sea"]:
        logging.info(" --> Apply sea-mask...")
        mask = xr.open_rasterio(data_settings["data"]["static"]["sea_mask"]).squeeze().rename({"x": "lon", "y": "lat"})
        alert_daily = xr.where(
            mask.reindex({"lon": alert_daily["lon"].values, "lat": alert_daily["lat"].values}, method="nearest") != 1,
            alert_daily, np.nan)
        logging.info(" --> Apply sea-mask...DONE")
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Assign hazard and impact levels
    hazards = data_settings["algorithm"]["settings"]["hazards"]

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
        logging.info(" --> Classify meteo country warning level based on impacts...DONE")

    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # Write output and clean
    logging.info(" --> Write output...")
    if alert_daily.lat.values[2] < alert_daily.lat.values[1]:
        logging.warning(" ---> Lat coordinate is flipped!")
        alert_daily = alert_daily.assign_coords(lat = alert_daily.lat.values[::-1])
        for haz in hazards:
            alert_daily[haz].values = np.flipud(alert_daily[haz].values)
            alert_daily[haz].values = np.flipud(alert_daily[haz].values)
    alert_daily.to_netcdf(output_file)

    if data_settings["algorithm"]["flags"]["clear_ancillary_data"]:
        os.system("rm " + ancillary_file)

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
            for lev, weigth in enumerate(impact_dict["weight_hazard_levels"][hazard], start=2):
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
# Call script from external library
if __name__ == "__main__":
    main()
# ----------------------------------------------------------------------------


