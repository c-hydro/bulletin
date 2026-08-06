import logging
import os
import time
import warnings
from copy import deepcopy
from typing import Sequence

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

    @staticmethod
    def _nearest_indices(source_coords: np.ndarray, target_coords: np.ndarray) -> np.ndarray:
        """
        Return indices of the nearest source coordinates for 1D target coordinates.

        The implementation is vectorized and supports ascending or descending
        coordinates. In case of an exact tie, the larger coordinate is selected,
        consistently with xarray/pandas nearest reindexing.
        """
        source = np.asarray(source_coords, dtype=float)
        target = np.asarray(target_coords, dtype=float)

        if source.ndim != 1 or target.ndim != 1:
            raise ValueError("Source and target coordinates must be one-dimensional")
        if source.size == 0:
            raise ValueError("Source coordinates cannot be empty")
        if source.size == 1:
            return np.zeros(target.size, dtype=np.int64)

        descending = source[0] > source[-1]
        ordered = source[::-1] if descending else source

        right = np.searchsorted(ordered, target, side="left")
        right = np.clip(right, 0, ordered.size - 1)
        left = np.clip(right - 1, 0, ordered.size - 1)

        distance_left = np.abs(target - ordered[left])
        distance_right = np.abs(ordered[right] - target)

        # On ties, select the larger coordinate, matching xarray reindex(method="nearest").
        choose_right = distance_right <= distance_left
        nearest = np.where(choose_right, right, left)

        if descending:
            nearest = ordered.size - 1 - nearest
        return nearest.astype(np.int64, copy=False)

    def classify_warning_levels_impact_based_multi_hazard(
        self,
        hazards: Sequence[str],
        hazards_short: Sequence[str],
        admin_gdf: gpd.GeoDataFrame,
        alert_daily: xr.Dataset,
        impact_settings: dict,
        exposed_element: str,
        progress_every: int = 25,
    ) -> dict[str, gpd.GeoDataFrame]:
        """
        Classify impact-based warning levels for several hazards in one pass.

        The exposed raster is opened once, while each administrative zone still
        reads only its own bounding box. The population subset and zone mask are
        reused for every hazard, keeping memory bounded and reducing raster I/O.
        """
        hazards = list(hazards)
        hazards_short = list(hazards_short)
        if len(hazards) != len(hazards_short):
            raise ValueError("hazards and hazards_short must have the same length")
        if not hazards:
            return {}

        alert_daily_max = alert_daily.max(dim="time")
        alert_lon = np.asarray(alert_daily_max["lon"].values)
        alert_lat = np.asarray(alert_daily_max["lat"].values)
        alert_maps = {
            hazard: np.asarray(alert_daily_max[hazard].transpose("lat", "lon").values)
            for hazard in hazards
        }

        exposed_map = impact_settings["exposed_map"][exposed_element]
        lack_capacity_col = impact_settings["lack_coping_capacity_col"]
        risk_thresholds = impact_settings["risk_thresholds"]
        hazard_weights = {
            hazard: impact_settings["weight_hazard_levels"][hazard]
            for hazard in hazards
        }

        out_by_hazard: dict[str, gpd.GeoDataFrame] = {}
        for hazard, hazard_short in zip(hazards, hazards_short):
            out_gdf = deepcopy(admin_gdf)
            out_gdf["stock"] = -9999.0
            out_gdf[hazard_short + "_tot"] = -9999.0
            out_gdf[hazard_short + "_perc"] = -9999.0
            out_gdf[hazard_short + "_level"] = -9999.0
            out_by_hazard[hazard] = out_gdf

        zone_count = len(admin_gdf)
        progress_every = max(1, int(progress_every))
        started = time.perf_counter()

        self.logger.info(
            "Looping through %s alert zones for %s risk (%s)",
            zone_count,
            ", ".join(hazards),
            exposed_element,
        )

        exposed_raster = rx.open_rasterio(exposed_map, cache=False)
        try:
            if exposed_raster.rio.crs is None:
                raise ValueError(f"Exposed raster has no CRS: {exposed_map}")
            if admin_gdf.crs is not None and admin_gdf.crs != exposed_raster.rio.crs:
                self.logger.warning(
                    "Warning-region CRS (%s) differs from exposed-raster CRS (%s). "
                    "Using the original geometries to preserve current behaviour.",
                    admin_gdf.crs,
                    exposed_raster.rio.crs,
                )

            for position, (index, row) in enumerate(admin_gdf.iterrows(), start=1):
                geometry = row.geometry
                self.logger.debug("Computing zone %s of %s", position, zone_count)

                if geometry is None or geometry.is_empty:
                    self.logger.warning("Zone %s has an empty geometry. Skipping.", index)
                    continue

                bbox = geometry.bounds
                clipped_exp = exposed_raster.rio.clip_box(
                    minx=bbox[0],
                    miny=bbox[1],
                    maxx=bbox[2],
                    maxy=bbox[3],
                )

                exposed_values = np.asarray(clipped_exp.values).squeeze().copy()
                exposed_values[~np.isfinite(exposed_values)] = 0.0
                exposed_values[exposed_values < 0] = 0.0

                lon_bbox = np.asarray(clipped_exp.x.values)
                lat_bbox = np.asarray(clipped_exp.y.values)
                zone_grid = GridsHandler.rasterize_shapes(
                    [(geometry, 1)],
                    {"lon": lon_bbox, "lat": lat_bbox},
                ).values
                zone_mask = zone_grid == 1

                population = np.where(zone_mask, exposed_values, 0.0)
                total_exposed = float(np.sum(population, dtype=np.float64))

                lon_indices = self._nearest_indices(alert_lon, lon_bbox)
                lat_indices = self._nearest_indices(alert_lat, lat_bbox)
                lack_capacity_factor = float(row[lack_capacity_col]) / 10.0

                for hazard, hazard_short in zip(hazards, hazards_short):
                    alert_window = alert_maps[hazard][np.ix_(lat_indices, lon_indices)]
                    weight_map = np.zeros(alert_window.shape, dtype=np.float64)
                    for level, weight in enumerate(hazard_weights[hazard], start=2):
                        weight_map[alert_window == level] = float(weight)

                    affected = float(
                        np.nansum(weight_map * population * lack_capacity_factor)
                    )
                    impact_rate = 0.0 if total_exposed == 0.0 else affected / total_exposed
                    risk = self.assign_risk(impact_rate, affected, risk_thresholds)

                    out_gdf = out_by_hazard[hazard]
                    out_gdf.at[index, hazard_short + "_level"] = risk
                    out_gdf.at[index, hazard_short + "_tot"] = affected
                    out_gdf.at[index, hazard_short + "_perc"] = impact_rate
                    out_gdf.at[index, "stock"] = total_exposed

                if position == 1 or position % progress_every == 0 or position == zone_count:
                    elapsed = time.perf_counter() - started
                    rate = position / elapsed if elapsed > 0 else 0.0
                    remaining = (zone_count - position) / rate if rate > 0 else 0.0
                    self.logger.info(
                        "Computed zone %s of %s [%.1f zones/s, ETA %.1f s]",
                        position,
                        zone_count,
                        rate,
                        remaining,
                    )
        finally:
            exposed_raster.close()

        return out_by_hazard

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
        Classify one hazard using the windowed multi-hazard implementation.

        This wrapper preserves the original public API.
        """
        outputs = self.classify_warning_levels_impact_based_multi_hazard(
            hazards=[hazard],
            hazards_short=[hazard_short],
            admin_gdf=admin_gdf,
            alert_daily=alert_daily,
            impact_settings=impact_settings,
            exposed_element=exposed_element,
        )
        return outputs[hazard]

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
