import os
import logging
import time
import warnings

import numpy as np
import rioxarray as rx
import pandas as pd
import geopandas as gpd
from common.grids_handler import GridsHandler
from common.io_handler import update_file_paths

class ImpactAssessment:
    def __init__(self, admin_shape: gpd.GeoDataFrame):
        """
        Initialize the ImpactAssessment.

        :param admin_shape: GeoDataFrame of the administrative shape.
        """
        self.admin_shape = admin_shape

    def process_impact_files(self, rp: int, filtered_hydro_to_admin: pd.DataFrame, impacts_table: pd.DataFrame, impact_files: dict, apply_defense: bool) -> None:
        """
        Process impact files for a given return period.

        :param rp: Return period.
        :param filtered_hydro_to_admin: Filtered DataFrame mapping hydro to admin.
        :param impacts_table: DataFrame of impacts.
        :param impact_files: Dictionary of impact files.
        """
        # Iterate over each exposed element
        for exposed_element in impact_files:
            logging.info("Analyse element " + exposed_element)
            # Iterate over each row in the filtered hydro_to_admin mapping
            for index, row in filtered_hydro_to_admin.iterrows():
                mul = row["mul"]
                if mul < 0:
                    continue
                impact_files_list = impact_files[exposed_element]["files"]
                if isinstance(impact_files_list, dict):
                    for sub_category in impact_files_list:
                        for file in impact_files_list[sub_category]:
                            self.process_mul(file, rp, row, impacts_table, exposed_element, sub_category, impact_files, apply_defense=apply_defense)
                elif isinstance(impact_files_list, list):
                    for file in impact_files_list:
                        self.process_mul(file, rp, row, impacts_table, exposed_element, impact_files=impact_files, apply_defense=apply_defense)
                else:
                    logging.error("ERROR! The type of impact_files['hydro'][exposed_element] should be either a list (even a singular one) or a dictionary!")
                    raise ValueError

    def process_mul(self, file: str, rp: int, row: pd.Series, impacts_table: pd.DataFrame, exposed_element: str, sub_category: str = None, impact_files: dict = None, apply_defense : bool = False) -> None:
        """
        Process a single impact file.

        :param file: Path to the impact file.
        :param rp: Return period.
        :param row: Row from the hydro_to_admin DataFrame.
        :param impacts_table: DataFrame of impacts.
        :param exposed_element: Exposed element name.
        :param sub_category: Sub-category name.
        :param impact_files: Dictionary of impact files.
        :param apply_defense: Flag to apply flood defenses defense.
        """
        mul = row["mul"]
        mul_file = file.format(mul=str(int(mul)))
        if not os.path.isfile(mul_file):
            return

        # Read the impact data from the file
        impact_data = pd.read_csv(mul_file, names=["rp", "abs", "std"], index_col=["rp"])

        if rp not in impact_data.index:
            return

        impact_mul = impact_data.loc[rp, "abs"]
        admin = row.name
        if apply_defense:
            defense = row["defense"]
        else:
            defense = 0

        if defense > 0 and rp < defense:
            impact_mul = 0
        elif defense > 0:
            rp_defense = int(defense)
            if rp_defense in impact_data.index:
                impact_mul -= impact_data.loc[rp_defense, "abs"]

        multiplier = impact_files[exposed_element].get("multiplier", 1)
        exluded_multipier = impact_files[exposed_element].get("excluded_multiplier", [])
        if sub_category is not None:
            if sub_category in exluded_multipier:
                multiplier = 1
        # Update the impacts table with the impact data
        impacts_table.at[admin, "flood_tot_" + exposed_element] += impact_mul * multiplier
        if sub_category:
            if "flood_tot_" + exposed_element + "_" + sub_category not in impacts_table.columns:
                logging.info("Category " + exposed_element + " has sub-category " + sub_category)
                impacts_table["flood_tot_" + exposed_element + "_" + sub_category] = 0.0
            impacts_table.at[admin, "flood_tot_" + exposed_element + "_" + sub_category] += impact_mul * multiplier

    def run(self, levels_sections: dict, hydro_to_admin: pd.DataFrame, impact_files: dict, apply_defense: bool = False) -> pd.DataFrame:
        """
        Run the impact assessment.

        :param levels_sections: Dictionary of levels and sections.
        :param hydro_to_admin: DataFrame mapping hydro to admin.
        :param impact_files: Dictionary of impact files.
        :param apply_defense: Flag to apply flood protection.
        :return: DataFrame of impacts.
        """
        logging.info("Calculate impacts...")

        # Initialize impacts table
        impacts_table = pd.DataFrame(index=self.admin_shape.index)
        for exposed_element in impact_files:
            impacts_table["flood_tot_" + exposed_element] = 0.0

        # Process each return period
        for rp in levels_sections:
            logging.info("Merge return period " + str(rp))
            filtered_hydro_to_admin = hydro_to_admin[hydro_to_admin["hydro"].isin(levels_sections[rp])]
            self.process_impact_files(rp, filtered_hydro_to_admin, impacts_table, impact_files, apply_defense)

        return impacts_table


    @staticmethod
    def _assign_overlay_risk(
        value_rel: float,
        value_abs: float,
        risk_thresholds: dict,
    ) -> int:
        """Classify an impact using paired absolute and relative thresholds."""
        risk = 0
        for risk_level, (risk_th_abs, risk_th_rel) in enumerate(
            zip(risk_thresholds["absolute"], risk_thresholds["relative"]),
            start=1,
        ):
            risk_th_abs = 0 if risk_th_abs is None else risk_th_abs
            risk_th_rel = 0 if risk_th_rel is None else risk_th_rel
            if risk_th_abs == 0 and risk_th_rel == 0:
                raise ValueError(
                    f"Both absolute and relative thresholds are none for class {risk_level}"
                )
            if value_rel >= risk_th_rel and value_abs >= risk_th_abs:
                risk = risk_level
            else:
                break
        return risk

    def empty_overlay(self, hazard: str = "flood") -> gpd.GeoDataFrame:
        """Return a zero-impact administrative layer for an empty flood mosaic."""
        output = self.admin_shape.copy()
        output["pop_total"] = 0.0
        output[hazard + "AffPpl"] = 0.0
        output[hazard + "AffPrc"] = 0.0
        output[hazard + "_level"] = 0.0
        return output

    def run_overlay(
        self,
        weighted_flood_map: str,
        impact_settings: dict,
        hazard: str = "flood",
    ) -> gpd.GeoDataFrame:
        """
        Calculate impacts by overlaying a weighted flood raster and exposure.

        This method is an opt-in alternative to ``run``, which keeps the
        established MUL-based behavior unchanged for all existing workflows.

        :param weighted_flood_map: Hazard-weighted flood mosaic.
        :param impact_settings: Exposure, vulnerability, and risk settings.
        :param hazard: Prefix used for output columns.
        :return: Administrative GeoDataFrame containing overlay impacts.
        """
        logging.info("Calculate impacts from raster overlay...")
        output = self.admin_shape.copy()
        output["pop_total"] = -9999.0
        output[hazard + "AffPpl"] = -9999.0
        output[hazard + "AffPrc"] = -9999.0
        output[hazard + "_level"] = -9999.0

        zone_count = len(output)
        progress_every = max(1, int(impact_settings.get("progress_every", 25)))
        started = time.perf_counter()

        logging.info(
            "Looping through %s impact zones for %s risk",
            zone_count,
            hazard,
        )

        # Keep both large rasters open, but continue reading only the bounding
        # box of the current administrative zone. This reduces repeated file
        # opening without loading the full rasters into memory.
        flood_raster = rx.open_rasterio(weighted_flood_map, cache=False)
        exposure_raster = rx.open_rasterio(
            impact_settings["exposed_map"],
            cache=False,
        )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                for position, (index, row) in enumerate(output.iterrows(), start=1):
                    minx, miny, maxx, maxy = row.geometry.bounds
                    try:
                        clipped_flood = flood_raster.rio.clip_box(
                            minx=minx,
                            miny=miny,
                            maxx=maxx,
                            maxy=maxy,
                        ).squeeze()
                        clipped_exposure = exposure_raster.rio.clip_box(
                            minx=minx,
                            miny=miny,
                            maxx=maxx,
                            maxy=maxy,
                        ).squeeze()
                    except rx.exceptions.NoDataInBounds:
                        output.at[index, hazard + "_level"] = 0
                        output.at[index, hazard + "AffPpl"] = 0
                        output.at[index, hazard + "AffPrc"] = 0
                        output.at[index, "pop_total"] = 0
                    else:
                        clipped_exposure.values = np.where(
                            clipped_exposure.values < 0,
                            0,
                            clipped_exposure.values,
                        )
                        lon_bbox = clipped_exposure.x.values
                        lat_bbox = clipped_exposure.y.values
                        alert_bbox = clipped_flood.reindex(
                            {"x": lon_bbox, "y": lat_bbox},
                            method="nearest",
                        )
                        admin_mask = GridsHandler.rasterize_shapes(
                            [(row.geometry, 1)],
                            {"lon": lon_bbox, "lat": lat_bbox},
                        )

                        weight_map = np.where(
                            admin_mask == 1,
                            alert_bbox.values / 100,
                            np.nan,
                        )
                        lack_capacity = (
                            row[impact_settings["lack_coping_capacity_col"]] / 10
                        )
                        affected = np.nansum(
                            weight_map
                            * np.squeeze(clipped_exposure.values)
                            * lack_capacity
                        )
                        total = np.nansum(
                            np.where(
                                admin_mask == 1,
                                np.squeeze(clipped_exposure.values),
                                np.nan,
                            )
                        )
                        affected_rate = 0 if total == 0 else affected / total
                        risk = self._assign_overlay_risk(
                            affected_rate,
                            affected,
                            impact_settings["risk_thresholds"],
                        )

                        output.at[index, hazard + "_level"] = risk
                        output.at[index, hazard + "AffPpl"] = affected
                        output.at[index, hazard + "AffPrc"] = affected_rate
                        output.at[index, "pop_total"] = total

                    if (
                        position == 1
                        or position % progress_every == 0
                        or position == zone_count
                    ):
                        elapsed = time.perf_counter() - started
                        rate = position / elapsed if elapsed > 0 else 0.0
                        remaining = (
                            (zone_count - position) / rate if rate > 0 else 0.0
                        )
                        logging.info(
                            "Computed impact zone %s of %s "
                            "[%.1f zones/s, ETA %.1f s]",
                            position,
                            zone_count,
                            rate,
                            remaining,
                        )
        finally:
            flood_raster.close()
            exposure_raster.close()

        return output

def initialize_subdomain_inputs(domain: str, subdomain: str, domain_shape: gpd.GeoDataFrame, mul_files: dict, hydro_to_admin_table: dict) -> tuple[pd.DataFrame, dict]:
    """
    Initialize subdomain inputs.

    :param domain: Domain name.
    :param subdomain: Subdomain name.
    :param domain_shape: GeoDataFrame of the domain shape.
    :param mul_files: Dictionary of MUL files.
    :param hydro_to_admin_table: Dictionary of hydro to admin table information.
    :return: Tuple of hydro_to_admin DataFrame and impact files dictionary.
    """
    replacements = {'domain': domain, 'subdomain': subdomain}
    impact_files = update_file_paths(mul_files, replacements)
    hydro_to_admin_file = update_file_paths(hydro_to_admin_table['filename'], replacements)

    # Load and filter hydro_to_admin table
    src = pd.read_csv(hydro_to_admin_file)

    admin_col = hydro_to_admin_table['admin_column']
    hydro_col = hydro_to_admin_table.get('hydro_column')
    defense_col = hydro_to_admin_table.get('defense_column')
    mul_col = hydro_to_admin_table['mul_column']

    # Build the working frame
    df = pd.DataFrame(index=src.index)
    df['admin'] = src[admin_col]

    # hydro: use provided column if valid, else default -9999
    if hydro_col is not None and hydro_col in src.columns:
        df['hydro'] = src[hydro_col]
    else:
        raise ValueError(f"Hydro column '{hydro_col}' not found in the source data. Association mul-hydro domain is needed!")

    # defense: use provided column if valid, else default 0
    if defense_col is not None and defense_col in src.columns:
        df['defense'] = src[defense_col]
    else:
        df['defense'] = 0

    # mul: required
    df['mul'] = src[mul_col]

    # Keep only admins in domain_shape index and index by admin
    df = df[df['admin'].isin(domain_shape.index)].set_index('admin')

    return df, impact_files
