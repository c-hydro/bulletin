import logging
import os
from copy import deepcopy

import geopandas as gpd
import numpy as np

from common.io_handler import IOHandler


class MeteoImpactMerger:
    """
    Merge impact shapefiles already produced for multiple meteo models.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

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

    def merge_hazard(
        self,
        hazard: str,
        hazard_short: str,
        models: dict,
        file_template: str,
        risk_thresholds: dict,
        out_file: str,
        raise_error_if_missing: bool = False,
    ) -> gpd.GeoDataFrame:
        """
        Merge one hazard across models using model weights.
        """
        first_step = True
        total_weight = 0.0
        output_gdf = None

        for model, model_settings in models.items():
            self.logger.info(f"Reading model {model}")
            file_in = file_template.format(
                model=model,
                hazard=hazard,
                HAZARD=hazard.upper(),
                haz=hazard_short,
                HAZ=hazard_short.upper(),
            )

            if not os.path.isfile(file_in):
                if raise_error_if_missing:
                    raise FileNotFoundError(file_in)
                self.logger.warning(f"File not found: {file_in}")
                continue

            model_data = gpd.read_file(file_in)
            weight = float(model_settings.get("weight", 1.0))
            value_col = hazard_short + "_tot"

            if first_step:
                output_gdf = deepcopy(model_data)
                output_gdf[value_col] = output_gdf[value_col] * weight
                first_step = False
            else:
                output_gdf[value_col] = output_gdf[value_col] + model_data[value_col] * weight

            total_weight = total_weight + weight

        if output_gdf is None or total_weight == 0:
            raise RuntimeError(f"No valid model data found for hazard '{hazard}'")

        value_col = hazard_short + "_tot"
        perc_col = hazard_short + "_perc"
        level_col = hazard_short + "_level"

        output_gdf[value_col] = output_gdf[value_col] / total_weight
        output_gdf[perc_col] = output_gdf[value_col] / output_gdf["stock"]
        output_gdf[perc_col] = output_gdf[perc_col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        output_gdf[level_col] = output_gdf.apply(
            lambda row: self.assign_risk(row[perc_col], row[value_col], risk_thresholds), axis=1
        )

        IOHandler.create_directories([os.path.dirname(out_file)])
        self.logger.info(f"Writing merged output: {out_file}")
        output_gdf.to_file(out_file)
        return output_gdf

    def run(
        self,
        hazards: list[str],
        hazards_short: list[str],
        models: dict,
        file_template: str,
        risk_thresholds: dict,
        out_file_template: str,
        raise_error_if_missing: bool = False,
    ) -> dict[str, gpd.GeoDataFrame]:
        """
        Merge all requested hazards.
        """
        outputs: dict[str, gpd.GeoDataFrame] = {}
        for hazard, hazard_short in zip(hazards, hazards_short):
            out_file = out_file_template.format(
                hazard=hazard,
                HAZARD=hazard.upper(),
                haz=hazard_short,
                HAZ=hazard_short.upper(),
            )
            outputs[hazard] = self.merge_hazard(
                hazard=hazard,
                hazard_short=hazard_short,
                models=models,
                file_template=file_template,
                risk_thresholds=risk_thresholds,
                out_file=out_file,
                raise_error_if_missing=raise_error_if_missing,
            )
        return outputs
