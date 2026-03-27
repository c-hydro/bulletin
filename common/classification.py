import logging
import geopandas as gpd

class ImpactClassifier:
    def __init__(self, shapefile_path: str, classification_info: dict, hazard: str):
        """
        Initialize the ImpactClassifier.

        :param shapefile_path: Path to the shapefile.
        :param classification_info: Dictionary containing classification information.
        :param hazard: Hazard name.
        """
        self.shapefile_path = shapefile_path
        self.abs_column = classification_info['absolute_values']
        self.risk_thresholds = classification_info['thresholds']
        self.hazard = hazard

    def classify(self) -> None:
        """
        Classify impact levels based on thresholds.
        """
        logging.info(f"Classifying impact levels for {self.shapefile_path}")
        short_hazard = self.hazard[:6]
        if short_hazard != self.hazard:
            logging.warning(f"Shortened hazard name to {short_hazard} to satisfy shapefile limits")
            self.hazard = short_hazard
        gdf = gpd.read_file(self.shapefile_path)
        gdf[short_hazard + "_perc"] = gdf[short_hazard + "_tot"] / gdf[self.abs_column]
        gdf[short_hazard + "_perc"] = gdf[short_hazard + "_perc"].fillna(0)
        gdf[short_hazard + "_clas"] = -9999.0

        risk = 0
        for index, row in gdf.iterrows():
            impact_rate = row[short_hazard + "_perc"]
            aff_people = row[short_hazard + "_tot"]
            for risk_lev, (risk_th_abs, risk_th_rel) in enumerate(
                    zip(self.risk_thresholds["absolute"], self.risk_thresholds["relative"]), start=1):
                if risk_th_abs is None: risk_th_abs = 0
                if risk_th_rel is None: risk_th_rel = 0

                if risk_th_abs == 0 and risk_th_rel == 0:
                    raise ValueError("Both absolute and relative thresholds are none for class " + str(risk_lev))
                elif impact_rate >= risk_th_rel and aff_people >= risk_th_abs:
                    risk = risk_lev
                else:
                    break
            gdf.at[index, short_hazard + "_clas"] = risk

        gdf.to_file(self.shapefile_path)