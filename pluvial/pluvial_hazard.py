import logging
import numpy as np
import xarray as xr

from common.io_handler import IOHandler


class PluvialHazardAssessment:
    """
    Blend model maxima, select flood scenarios per duration, and build a final hazard raster.

    scenario_merging:
      - 'combine': pixel-wise maximum across selected scenario maps
      - 'worst': select the map with the highest scenario id; if ties, combine only the tied maps
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    def blend_scalar(
        self,
        model_to_value: dict[str, float],
        model_to_weight: dict[str, float],
        method: str = "weighted_mean",
    ) -> float:
        vals: list[float] = []
        wts: list[float] = []

        for model, v in model_to_value.items():
            if v is None:
                continue
            try:
                fv = float(v)
            except Exception:
                continue
            if np.isnan(fv):
                continue
            vals.append(fv)
            wts.append(float(model_to_weight.get(model, 1.0)))

        if len(vals) == 0:
            return float("nan")

        method = str(method).lower()
        if method == "max":
            return float(np.max(vals))

        if method == "weighted_mean":
            wsum = float(np.sum(wts))
            if wsum <= 0:
                return float(np.mean(vals))
            return float(np.sum(np.array(vals) * np.array(wts)) / wsum)

        raise ValueError("method must be 'weighted_mean' or 'max'")

    def select_scenario(
        self,
        value_mm: float,
        thresholds_mm: list[float],
        scenarios: list[int],
    ) -> int | None:
        """
        Select the highest scenario whose threshold is exceeded.
        If none exceeded -> None.
        """
        if value_mm is None:
            return None
        if isinstance(value_mm, float) and np.isnan(value_mm):
            return None
        if len(thresholds_mm) != len(scenarios):
            raise ValueError("thresholds_mm and scenarios must have the same length")

        selected = None
        for thr, sc in zip(thresholds_mm, scenarios):
            if value_mm >= float(thr):
                selected = int(sc)
            else:
                break
        return selected

    def compute_duration_scenarios(
        self,
        maxima_by_model: dict[str, dict[str, float]],
        weights_by_model: dict[str, float],
        thresholds_by_duration: dict[str, dict],
        duration_keys: list[str],
        model_blend: str = "weighted_mean",
    ) -> tuple[dict[str, float], dict[str, int | None]]:
        blended: dict[str, float] = {}
        scenario_by_duration: dict[str, int | None] = {}

        for dur in duration_keys:
            model_to_value = {m: maxima_by_model[m].get(dur, float("nan")) for m in maxima_by_model.keys()}
            b = self.blend_scalar(model_to_value, weights_by_model, method=model_blend)
            blended[dur] = float(b)

            th_cfg = thresholds_by_duration[dur]
            thresholds_mm = th_cfg.get("thresholds_mm", th_cfg.get("tresholds_mm"))
            scenarios = th_cfg.get("scenarios")

            if thresholds_mm is None or scenarios is None:
                raise KeyError(f"Missing thresholds/scenarios for duration '{dur}'")

            scenario_by_duration[dur] = self.select_scenario(float(b), list(thresholds_mm), list(scenarios))

        return blended, scenario_by_duration

    def build_final_hazard_raster(
        self,
        scenario_by_duration: dict[str, int | None],
        flood_map_template: str,
        scenario_merging: str = "combine",
    ) -> xr.DataArray | None:
        """
        Build final hazard raster based on scenario merging strategy.
        """
        scenario_merging = str(scenario_merging).lower()
        if scenario_merging not in ["combine", "worst"]:
            raise ValueError("scenario_merging must be 'combine' or 'worst'")

        selected = {dur: sc for dur, sc in scenario_by_duration.items() if sc is not None}
        if len(selected) == 0:
            return None

        # Decide which durations to use
        if scenario_merging == "combine":
            durations_to_use = list(selected.keys())
        else:
            max_scenario = max(selected.values())
            durations_with_max = [dur for dur, sc in selected.items() if sc == max_scenario]
            if len(durations_with_max) == 1:
                durations_to_use = durations_with_max
                self.logger.info(
                    f"scenario_merging='worst': selecting single duration '{durations_to_use[0]}' "
                    f"with scenario {max_scenario}"
                )
            else:
                durations_to_use = durations_with_max
                self.logger.info(
                    f"scenario_merging='worst': {len(durations_to_use)} durations share max scenario {max_scenario}; "
                    f"combining those maps (pixel-wise max)"
                )

        # Load maps
        maps: list[xr.DataArray] = []
        for dur in durations_to_use:
            sc = selected[dur]
            path = flood_map_template.format(duration=dur, scenario=sc)
            self.logger.info(f"Reading scenario map for {dur} (scenario={sc}): {path}")
            da = IOHandler.read_raster(path, memory_map=False)
            maps.append(da)

        if len(maps) == 0:
            return None

        # Combine maps (pixel-wise max)
        final_da = maps[0]
        for m in maps[1:]:
            final_da = xr.ufuncs.maximum(final_da, m)

        # Restore georeferencing after xarray ufuncs
        ref = maps[0]
        try:
            final_da = final_da.rio.set_spatial_dims(x_dim=ref.rio.x_dim, y_dim=ref.rio.y_dim, inplace=False)
        except Exception:
            final_da = final_da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)

        if ref.rio.crs is not None:
            final_da = final_da.rio.write_crs(ref.rio.crs, inplace=False)

        try:
            final_da = final_da.rio.write_transform(ref.rio.transform(), inplace=False)
        except Exception:
            pass

        try:
            nd = ref.rio.nodata
            if nd is not None:
                final_da = final_da.rio.write_nodata(nd, inplace=False)
        except Exception:
            pass

        final_da.attrs.update(ref.attrs)
        return final_da

    def save_hazard_raster(self, hazard_da: xr.DataArray, out_path: str, nodata: float | None = None) -> None:
        """
        Save final hazard raster to GeoTIFF.
        """
        self.logger.info(f"Writing final hazard raster: {out_path}")
        IOHandler.write_raster(hazard_da, out_path, nodata=nodata, compress="DEFLATE")

    def run(
        self,
        maxima_by_model: dict[str, dict[str, float]],
        weights_by_model: dict[str, float],
        thresholds_by_duration: dict[str, dict],
        duration_keys: list[str],
        flood_map_template: str,
        model_blend: str = "weighted_mean",
        scenario_merging: str = "combine",
        out_raster_path: str | None = None,
        nodata: float | None = None,
    ) -> tuple[dict[str, float], dict[str, int | None], xr.DataArray | None]:
        """
        End-to-end hazard stage:
          - blend model maxima
          - select scenario per duration
          - build hazard raster (combine/worst)
          - optionally save to disk

        Returns:
          blended_by_duration, scenario_by_duration, hazard_da
        """
        blended, scenario_by_duration = self.compute_duration_scenarios(
            maxima_by_model=maxima_by_model,
            weights_by_model=weights_by_model,
            thresholds_by_duration=thresholds_by_duration,
            duration_keys=duration_keys,
            model_blend=model_blend,
        )

        hazard_da = self.build_final_hazard_raster(
            scenario_by_duration=scenario_by_duration,
            flood_map_template=flood_map_template,
            scenario_merging=scenario_merging,
        )

        if hazard_da is not None and out_raster_path is not None:
            self.save_hazard_raster(hazard_da, out_raster_path, nodata=nodata)

        return blended, scenario_by_duration, hazard_da
