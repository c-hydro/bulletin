import os
import logging
import pandas as pd
import geopandas as gpd
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