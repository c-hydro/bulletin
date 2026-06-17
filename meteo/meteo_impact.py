import logging
import os
import warnings
from copy import deepcopy

import geopandas as gpd
import numpy as np
import rioxarray as rx
import xarray as xr
from common.io_handler import IOHandler
from common.grids_handler import GridsHandler


class MeteoImpactAssessment:
    """
    Convert gridded meteo alert levels to admin-level hazard and impact outputs.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    def classify_warning_levels_pure_hazard(
        self,
        hazard: str,
        hazard_short: str,
        admin_gdf: gpd.GeoDataFrame,
        alert_daily: xr.Dataset,
        min_warning_threshold: int = 1,
    ) -> gpd.GeoDataFrame:
        """
        Classify warning regions using hazard levels only.
        """
        shapes = [(shape, n) for n, shape in enumerate(admin_gdf.geometry)]
        ds = xr.Dataset(coords={"lon": alert_daily["lon"].values, "lat": alert_daily["lat"].values})
        ds["states"] = GridsHandler.rasterize_shapes(shapes, ds.coords)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            out_gdf = deepcopy(admin_gdf)
            out_gdf[hazard_short + "_level"] = -9999.0
            alert_step = alert_daily[hazard]

            self.logger.info(f"Looping through alert zones for {hazard} hazard")
            for index, _ in out_gdf.iterrows():
                val_max = np.nanmax(alert_step.where(ds["states"] == index))
                while val_max > 1:
                    tot_over = np.count_nonzero(alert_step.where(ds["states"] == index) >= val_max)
                    if tot_over >= min_warning_threshold:
                        break
                    val_max = val_max - 1
                out_gdf.at[index, hazard_short + "_level"] = val_max

        return out_gdf

    def classify_warning_levels_impact_based(
        self,
        hazard: str,
        hazard_short: str,
        admin_gdf: gpd.GeoDataFrame,
        alert_daily: xr.Dataset,
        impact_settings: dict,
        exposed_element: str,
    ) -> gpd.GeoDataFrame:
        """
        Classify warning regions using hazard levels and exposed elements.
        """
        alert_daily_max = alert_daily.max(dim="time")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            out_gdf = deepcopy(admin_gdf)
            out_gdf["stock"] = -9999.0
            out_gdf[hazard_short + "_tot"] = -9999.0
            out_gdf[hazard_short + "_perc"] = -9999.0
            out_gdf[hazard_short + "_level"] = -9999.0

            exposed_map = impact_settings["exposed_map"][exposed_element]
            lack_capacity_col = impact_settings["lack_coping_capacity_col"]
            hazard_weights = impact_settings["weight_hazard_levels"][hazard]
            risk_thresholds = impact_settings["risk_thresholds"]

            self.logger.info(f"Looping through alert zones for {hazard} risk")
            for index, row in out_gdf.iterrows():
                self.logger.info(f"Computing zone {index + 1} of {len(out_gdf)}")
                bbox = row["geometry"].bounds
                clipped_exp = rx.open_rasterio(exposed_map).rio.clip_box(
                    minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3]
                )
                clipped_exp.values[clipped_exp.values < 0] = 0

                lon_bbox = clipped_exp.x.values
                lat_bbox = clipped_exp.y.values
                alert_bbox = alert_daily_max.reindex({"lon": lon_bbox, "lat": lat_bbox}, method="nearest")
                country_bbox = GridsHandler.rasterize_shapes([(row["geometry"], index + 1)], {"lon": lon_bbox, "lat": lat_bbox})

                weight_map = np.where(country_bbox == index + 1, 0, np.nan)
                for level, weight in enumerate(hazard_weights, start=2):
                    weight_map = np.where(alert_bbox[hazard].values == level, weight_map + weight, weight_map)

                aff_people = np.nansum(
                    weight_map * np.squeeze(clipped_exp.values) * (row[lack_capacity_col] / 10)
                )
                tot_people = np.nansum(np.where(country_bbox == index + 1, np.squeeze(clipped_exp.values), np.nan))
                impact_rate = 0 if tot_people == 0 else aff_people / tot_people
                risk = self.assign_risk(impact_rate, aff_people, risk_thresholds)

                out_gdf.at[index, hazard_short + "_level"] = risk
                out_gdf.at[index, hazard_short + "_tot"] = aff_people
                out_gdf.at[index, hazard_short + "_perc"] = impact_rate
                out_gdf.at[index, "stock"] = tot_people

        return out_gdf

    def assign_risk(self, value_rel: float, value_abs: float, risk_thresholds: dict) -> int:
        """
        Assign a risk level using relative and absolute thresholds.
        """
        risk = 0
        for risk_level, (risk_th_abs, risk_th_rel) in enumerate(
            zip(risk_thresholds["absolute"], risk_thresholds["relative"]), start=1
        ):
            if risk_th_abs is None:
                risk_th_abs = 0
            if risk_th_rel is None:
                risk_th_rel = 0

            if risk_th_abs == 0 and risk_th_rel == 0:
                raise ValueError(f"Both absolute and relative thresholds are none for class {risk_level}")
            if value_rel >= risk_th_rel and value_abs >= risk_th_abs:
                risk = risk_level
            else:
                break
        return risk

    def save(self, gdf: gpd.GeoDataFrame, out_name: str) -> None:
        """
        Save an admin-level output file.
        """
        IOHandler.create_directories([os.path.dirname(out_name)])
        self.logger.info(f"Saving shapefile: {out_name}")
        gdf.to_file(out_name)
