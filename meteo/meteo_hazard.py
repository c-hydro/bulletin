import logging
import os
import netrc
import datetime as dt
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from common.io_handler import IOHandler, update_file_paths
from common.grids_handler import GridsHandler


@dataclass
class MeteoVariable:
    name: str
    varname: str
    filename: str | None = None
    level: str | int | float | None = None
    date_selected: list[int] | None = None
    accumulated: bool = False


def _to_naive_utc_timestamp(value) -> pd.Timestamp:
    """
    Convert datetime-like values to timezone-naive UTC pandas timestamps.
    """
    timestamp = pd.Timestamp(value)
    if timestamp.tz is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


class MeteoForecastInput:
    """
    Read and standardize meteorological forecast data.

    Supported sources:
      - local NetCDF files
      - drops2 coverages, when the optional drops2 package is available
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    def read_local_variables(
        self,
        variables: dict[str, MeteoVariable],
        time_tokens: dict[str, str] | None = None,
    ) -> dict[str, xr.DataArray]:
        """
        Read local variables from NetCDF files.

        :param variables: Dictionary of variables to read.
        :param time_tokens: Optional path-format tokens.
        :return: Dictionary of xarray DataArray objects.
        """
        data_vars: dict[str, xr.DataArray] = {}
        time_tokens = time_tokens or {}

        for hazard_name, variable in variables.items():
            if variable.filename is None:
                raise ValueError(f"Missing filename for variable '{hazard_name}'")

            file_name = update_file_paths(variable.filename, time_tokens)
            self.logger.info(f"Opening local forecast for '{hazard_name}': {file_name}")
            ds = IOHandler.open_netcdf_dataset(file_name)

            if variable.varname not in ds:
                raise KeyError(f"Variable '{variable.varname}' not found in {file_name}")

            da = ds[variable.varname]
            da = self._select_level_if_present(da, variable.level)
            da = self._normalize_coordinates(da)
            data_vars[hazard_name] = da

        return data_vars

    def read_drops_variables(
        self,
        variables: dict[str, MeteoVariable],
        drops_settings: dict,
        date_from: dt.datetime,
        date_to: dt.datetime,
    ) -> tuple[dict[str, xr.DataArray], dt.datetime]:
        """
        Download variables from drops2.

        :param variables: Dictionary of variables to download.
        :param drops_settings: drops2 connection settings.
        :param date_from: Lower search time.
        :param date_to: Upper search time.
        :return: Tuple of downloaded variables and selected reference time.
        """
        from drops2 import coverages
        from drops2.utils import DropsCredentials

        if not all([drops_settings.get("DropsUser"), drops_settings.get("DropsPwd")]):
            netrc_handle = netrc.netrc()
            try:
                drops_settings["DropsUser"], _, drops_settings["DropsPwd"] = netrc_handle.authenticators(
                    drops_settings["DropsAddress"]
                )
            except Exception as exc:
                raise FileNotFoundError(
                    "Verify that your .netrc file exists in the home directory and that it includes proper credentials."
                ) from exc

        DropsCredentials.set(drops_settings["DropsAddress"], drops_settings["DropsUser"], drops_settings["DropsPwd"])

        data_id = drops_settings["DropsDataId"]
        self.logger.info(f"Searching drops2 forecast for {data_id}")
        date_from_ts = _to_naive_utc_timestamp(date_from)
        date_to_ts = _to_naive_utc_timestamp(date_to)
        model_dates_raw = coverages.get_dates(data_id, date_from_ts.strftime("%Y%m%d%H%M"), date_to_ts.strftime("%Y%m%d%H%M"))
        model_dates = [_to_naive_utc_timestamp(i).to_pydatetime() for i in model_dates_raw]
        model_dates = [i for i in model_dates if date_from_ts <= pd.Timestamp(i) <= date_to_ts]

        if len(model_dates) == 0:
            raise FileNotFoundError("No forecast files available in the selected time window")

        date_ref = sorted(model_dates, reverse=True)[0]
        self.logger.info(f"Using drops2 forecast reference time: {date_ref:%Y%m%d%H%M}")

        data_vars: dict[str, xr.DataArray] = {}
        for hazard_name, variable in variables.items():
            self.logger.info(f"Downloading variable '{hazard_name}' ({variable.varname})")
            data_drops = coverages.get_data(
                data_id,
                date_ref,
                variable.varname,
                variable.level,
                date_selected=variable.date_selected,
            )

            lon = np.unique(data_drops.longitude.values)
            lat = np.unique(data_drops.latitude.values)
            time_index = np.array([pd.Timestamp(i).round("60min").to_pydatetime() for i in data_drops.time.values])

            da = xr.DataArray(
                dims=["time", "lat", "lon"],
                coords={"lon": lon, "lat": lat, "time": time_index},
                data=data_drops[variable.varname].values,
                name=hazard_name,
            )
            data_vars[hazard_name] = self._normalize_coordinates(da)

        return data_vars, date_ref

    def build_dataset(self, data_vars: dict[str, xr.DataArray]) -> xr.Dataset:
        """
        Build a dataset from standardized variables.
        """
        return xr.Dataset(data_vars)

    def crop_bbox(self, data: xr.Dataset, bbox: dict | None = None) -> xr.Dataset:
        """
        Crop dataset using bbox keys lon_left, lon_right, lat_bottom, lat_top.
        """
        return GridsHandler.crop_bbox(data, bbox)

    def slice_time(self, data: xr.Dataset, forecast_end: dt.datetime | None = None) -> xr.Dataset:
        """
        Keep only time steps up to forecast_end.
        """
        if forecast_end is None:
            return data
        forecast_end_ts = _to_naive_utc_timestamp(forecast_end)
        selected_times = [i for i in data.time.values if _to_naive_utc_timestamp(i) <= forecast_end_ts]
        return data.sel(time=selected_times)

    def preprocess_hazards(
        self,
        data: xr.Dataset,
        hazards: list[str],
        variables_settings: dict,
        forecast_resolution_h: int | float = 1,
        rain_window_h: int | float = 24,
    ) -> xr.Dataset:
        """
        Convert raw variables to hazard variables used for classification.
        """
        processed: dict[str, xr.DataArray] = {}

        if "rain" in hazards:
            if "rainc" in data and "rainnc" in data:
                self.logger.info("Combining rainfall components rainc + rainnc")
                rain = data["rainc"] + data["rainnc"]
                accumulated = bool(
                    variables_settings.get("rainc", {}).get("accumulated", False)
                    and variables_settings.get("rainnc", {}).get("accumulated", False)
                )
            elif "rain" in data:
                rain = data["rain"]
                accumulated = bool(variables_settings.get("rain", {}).get("accumulated", False))
            else:
                raise NotImplementedError(
                    "Rain hazard is active. Provide 'rain' or the 'rainc' and 'rainnc' components."
                )

            if accumulated:
                self.logger.info("Decumulating accumulated rainfall")
                rain_step = xr.concat([rain.isel(time=slice(0, 1)), rain.diff("time")], dim="time")
            else:
                rain_step = rain.copy()

            steps = max(int(round(float(rain_window_h) / float(forecast_resolution_h))), 1)
            self.logger.info(f"Cumulating rainfall with {steps} forecast steps")
            rain_acc = rain_step.rolling(time=steps, center=True).sum().shift(time=-1)
            processed["rain"] = xr.where(rain_acc < 0, 0, rain_acc)

        if "wind" in hazards:
            if "u-wind" not in data or "v-wind" not in data:
                raise NotImplementedError("Wind hazard is active. Provide 'u-wind' and 'v-wind'.")
            self.logger.info("Merging wind components")
            processed["wind"] = np.sqrt((data["u-wind"] ** 2) + (data["v-wind"] ** 2))

        for hazard in hazards:
            if hazard not in processed:
                if hazard not in data:
                    raise KeyError(f"Hazard '{hazard}' not available in forecast dataset")
                self.logger.info(f"Variable '{hazard}' is already in standard format")
                processed[hazard] = data[hazard].copy()

        return xr.Dataset(processed)

    def daily_maxima(self, data: xr.Dataset, date_ref: dt.datetime, forecast_end: dt.datetime) -> xr.Dataset:
        """
        Calculate daily maximum values over the forecast period.
        """
        return GridsHandler.daily_maxima(data, date_ref, forecast_end)

    def _select_level_if_present(self, da: xr.DataArray, level: str | int | float | None) -> xr.DataArray:
        if level in [None, "-"]:
            return da
        return GridsHandler.select_level_if_present(da, level)

    def _normalize_coordinates(self, da: xr.DataArray) -> xr.DataArray:
        return GridsHandler.normalize_lat_lon_coordinates(da)


class MeteoHazardAssessment:
    """
    Classify meteorological hazard maps from daily forecast maxima.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    def classify_hazard_with_maps(
        self,
        hazard: str,
        thresholds_settings: dict,
        ds_out_daily: xr.Dataset,
        alert_daily: xr.Dataset,
    ) -> xr.Dataset:
        """
        Classify a hazard using threshold maps.
        """
        thresholds = thresholds_settings["quantiles"]
        limits = thresholds_settings.get("limits", [[-np.inf, np.inf] for _ in thresholds])
        threshold_map = {}

        for threshold in thresholds:
            file_name = os.path.join(thresholds_settings["folder"], thresholds_settings["file_name"]).format(
                quantile=str(threshold)
            )
            threshold_map[str(threshold)] = IOHandler.read_raster(file_name, memory_map=False)

        ds_threshold = xr.Dataset(threshold_map).squeeze()
        ds_threshold = ds_threshold.rename({"x": "lon", "y": "lat"}) if "x" in ds_threshold.coords else ds_threshold
        ds_threshold = ds_threshold.reindex({"lon": ds_out_daily["lon"].values, "lat": ds_out_daily["lat"].values}, method="nearest")

        for threshold, lims in zip(thresholds, limits):
            ds_threshold[str(threshold)] = xr.where(ds_threshold[str(threshold)] < lims[0], lims[0], ds_threshold[str(threshold)])
            ds_threshold[str(threshold)] = xr.where(ds_threshold[str(threshold)] > lims[1], lims[1], ds_threshold[str(threshold)])

        alert_maps = np.where(ds_out_daily[hazard].values >= 0, 1, np.nan)
        for level, threshold in enumerate(thresholds, start=2):
            alert_maps = np.where(ds_out_daily[hazard].values >= ds_threshold[str(threshold)].values, level, alert_maps)

        alert_daily[hazard].values = alert_maps
        return alert_daily

    def classify_hazard_with_values(
        self,
        hazard: str,
        thresholds_settings: dict,
        ds_out_daily: xr.Dataset,
        alert_daily: xr.Dataset,
    ) -> xr.Dataset:
        """
        Classify a hazard using scalar thresholds.
        """
        thresholds = thresholds_settings["values"]
        alert_maps = np.where(ds_out_daily[hazard].values >= 0, 1, np.nan)

        for level, threshold in enumerate(thresholds, start=2):
            alert_maps = np.where(ds_out_daily[hazard].values >= threshold, level, alert_maps)

        alert_daily[hazard].values = alert_maps
        return alert_daily

    def classify_hazard_with_values_inverse(
        self,
        hazard: str,
        thresholds_settings: dict,
        ds_out_daily: xr.Dataset,
        alert_daily: xr.Dataset,
    ) -> xr.Dataset:
        """
        Classify a hazard where lower values imply higher alert levels.
        """
        thresholds = list(thresholds_settings["values"])[::-1]
        alert_maps = np.where(ds_out_daily[hazard].values >= 0, 1, np.nan)

        for level, threshold in enumerate(thresholds, start=2):
            alert_maps = np.where(ds_out_daily[hazard].values <= threshold, level, alert_maps)

        alert_daily[hazard].values = alert_maps
        return alert_daily

    def classify(
        self,
        ds_out_daily: xr.Dataset,
        hazards: list[str],
        thresholds_settings: dict,
    ) -> xr.Dataset:
        """
        Classify all active hazards.
        """
        alert_daily = deepcopy(ds_out_daily)

        for hazard in hazards:
            self.logger.info(f"Classifying {hazard} alert level")
            hazard_thresholds = thresholds_settings[hazard]
            threshold_type = str(hazard_thresholds["type"]).lower()
            if threshold_type == "map":
                alert_daily = self.classify_hazard_with_maps(hazard, hazard_thresholds, ds_out_daily, alert_daily)
            elif threshold_type == "value":
                alert_daily = self.classify_hazard_with_values(hazard, hazard_thresholds, ds_out_daily, alert_daily)
            elif threshold_type == "value_inverse":
                alert_daily = self.classify_hazard_with_values_inverse(hazard, hazard_thresholds, ds_out_daily, alert_daily)
            else:
                raise NotImplementedError("Threshold format not recognised. Choose 'map', 'value' or 'value_inverse'.")

        return alert_daily

    def apply_sea_mask(self, alert_daily: xr.Dataset, sea_mask_file: str | None) -> xr.Dataset:
        """
        Apply a sea mask to the alert dataset.
        """
        if sea_mask_file is None:
            return alert_daily

        self.logger.info("Applying sea mask")
        mask = IOHandler.read_raster(sea_mask_file, memory_map=False)
        mask = mask.rename({"x": "lon", "y": "lat"}) if "x" in mask.coords else mask
        mask = mask.reindex({"lon": alert_daily["lon"].values, "lat": alert_daily["lat"].values}, method="nearest")
        return xr.where(mask != 1, alert_daily, np.nan)
