import logging
from dataclasses import dataclass

import pandas as pd
import geopandas as gpd

from common.io_handler import IOHandler
from common.grids_handler import GridsHandler


@dataclass
class ModelInput:
    name: str
    filename: str
    varname: str
    weight: float = 1.0
    decumulate_rain: bool = False


class PluvialInputManager:
    """
    Read rainfall time series from NetCDF and compute rolling-window maxima by duration.

    This class does NOT read JSON settings. It only receives plain arguments.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    def parse_duration_key(self, duration_key: str) -> int:
        """
        Convert duration string like '24h' to integer hours.
        """
        key = str(duration_key).strip().lower()
        if not key.endswith("h"):
            raise ValueError(f"Unsupported duration format: {duration_key}. Expected like '24h'.")
        return int(key[:-1])

    def read_model_timeseries(
        self,
        model: ModelInput,
        points_gdf: gpd.GeoDataFrame,
        point_id_field: str,
        method: str = "nearest",
        spatial_aggregation: dict | None = None,
    ) -> pd.DataFrame:
        """
        Open a NetCDF and extract rainfall time series at point locations.

        spatial_aggregation can enable 3x3 weighted quantile extraction.
        """
        self.logger.info(f"Opening NetCDF for model '{model.name}': {model.filename}")
        ds = IOHandler.open_netcdf_dataset(model.filename)

        self.logger.info(f"Extracting time series for model '{model.name}' variable '{model.varname}'")
        ts_df = GridsHandler.extract_timeseries_at_points(
            ds=ds,
            varname=model.varname,
            points_gdf=points_gdf,
            point_id_field=point_id_field,
            method=method,
            spatial_aggregation=spatial_aggregation,
        )
        return ts_df

    def compute_duration_maxima(
        self,
        ts_df: pd.DataFrame,
        duration_keys: list[str],
        rolling_mode: str = "sum",
        point_agg: str = "max",
    ) -> dict[str, float]:
        """
        Compute a scalar maximum for each duration.

        rolling_mode: 'sum' or 'mean'
        point_agg: 'max' or 'mean' across points
        """
        if ts_df is None or ts_df.empty:
            raise ValueError("Empty time series dataframe")

        rolling_mode = str(rolling_mode).lower()
        if rolling_mode not in ["sum", "mean"]:
            raise ValueError("rolling_mode must be 'sum' or 'mean'")

        point_agg = str(point_agg).lower()
        if point_agg not in ["max", "mean"]:
            raise ValueError("point_agg must be 'max' or 'mean'")

        time_index = pd.DatetimeIndex(ts_df.index)
        dt_hours = GridsHandler.infer_time_step_hours(time_index)

        out: dict[str, float] = {}
        for dur_key in duration_keys:
            dur_hours = self.parse_duration_key(dur_key)
            window = int(round(dur_hours / dt_hours))
            window = max(window, 1)

            if rolling_mode == "sum":
                rolled = ts_df.rolling(window=window, min_periods=window).sum()
            else:
                rolled = ts_df.rolling(window=window, min_periods=window).mean()

            if point_agg == "max":
                per_time = rolled.max(axis=1)
            else:
                per_time = rolled.mean(axis=1)

            out[dur_key] = float(per_time.max())

        return out

    def run(
        self,
        models: list[ModelInput],
        points_gdf: gpd.GeoDataFrame,
        point_id_field: str,
        duration_keys: list[str],
        rolling_mode: str = "sum",
        point_agg: str = "max",
        skip_missing_models: bool = True,
        spatial_aggregation: dict | None = None,
        extraction_method: str = "nearest",
    ) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
        """
        End-to-end input stage:
          - read TS for each model (optionally spatial aggregated)
          - optionally decumulate per model
          - compute maxima by duration

        Returns:
          maxima_by_model[model_name][duration] = max value
          weights_by_model[model_name] = model weight
        """
        maxima_by_model: dict[str, dict[str, float]] = {}
        weights_by_model: dict[str, float] = {}

        for model in models:
            try:
                ts_df = self.read_model_timeseries(
                    model=model,
                    points_gdf=points_gdf,
                    point_id_field=point_id_field,
                    method=extraction_method,
                    spatial_aggregation=spatial_aggregation,
                )

                if model.decumulate_rain:
                    self.logger.info(f"Decumulating rainfall for model '{model.name}'")
                    ts_df = GridsHandler.decumulate_timeseries(ts_df)

                maxima = self.compute_duration_maxima(
                    ts_df=ts_df,
                    duration_keys=duration_keys,
                    rolling_mode=rolling_mode,
                    point_agg=point_agg,
                )

                maxima_by_model[model.name] = maxima
                weights_by_model[model.name] = float(model.weight)

            except Exception as exc:
                if skip_missing_models:
                    self.logger.warning(f"Skipping model '{model.name}' due to error: {exc}")
                    continue
                raise

        if len(maxima_by_model) == 0:
            raise RuntimeError("No model maxima computed (all models missing or failed).")

        return maxima_by_model, weights_by_model
