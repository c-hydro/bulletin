import logging
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask


class PluvialImpactAssessment:
    """
    Compute admin-level impacts from a hazard raster.

    Outputs:
      - frac_aff: fraction (0-1) of valid raster cells > affected_value_threshold
      - perc_cells: percentage (0-100) of valid raster cells > affected_value_threshold
      - lev: classified level based on hazard_classification thresholds
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    def classify_level(self, affected_fraction: float, hazard_classification: dict[str, float]) -> int:
        """
        Classify level based on affected fraction thresholds.
        hazard_classification example: {"1":0.1, "2":0.5, "3":0.8}
        """
        if affected_fraction is None or (isinstance(affected_fraction, float) and np.isnan(affected_fraction)):
            return 0

        items = sorted(((int(k), float(v)) for k, v in hazard_classification.items()), key=lambda x: x[0])
        lev = 0
        for level, thr in items:
            if affected_fraction >= thr:
                lev = level
            else:
                break
        return lev

    def compute_admin_impacts(
        self,
        admin_gdf: gpd.GeoDataFrame,
        hazard_raster_path: str,
        hazard_classification: dict[str, float],
        affected_value_threshold: float = 0.0,
    ) -> gpd.GeoDataFrame:
        """
        Compute admin impacts and return a GeoDataFrame with new columns.
        """
        admin = admin_gdf.copy()

        with rasterio.open(hazard_raster_path) as src:
            self.logger.info(f"Computing admin impacts using raster: {hazard_raster_path}")

            if admin.crs is not None and src.crs is not None and admin.crs != src.crs:
                admin = admin.to_crs(src.crs)

            nodata = src.nodata
            frac_aff: list[float] = []

            for _, row in admin.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    frac_aff.append(float("nan"))
                    continue

                try:
                    out_img, _ = mask(src, [geom], crop=True, filled=True, nodata=nodata)
                except ValueError:
                    # Polygon outside raster extent
                    frac_aff.append(0.0)
                    continue

                arr = out_img[0]

                valid = np.ones_like(arr, dtype=bool)
                if nodata is not None:
                    valid = arr != nodata

                valid_count = int(np.sum(valid))
                if valid_count == 0:
                    frac_aff.append(0.0)
                    continue

                affected = valid & (arr > float(affected_value_threshold))
                frac = float(np.sum(affected)) / float(valid_count)
                frac_aff.append(frac)

            admin["frac_aff"] = frac_aff
            admin["perc_cells"] = [float(v) * 100.0 if np.isfinite(v) else float("nan") for v in frac_aff]
            admin["lev"] = [self.classify_level(v, hazard_classification) for v in frac_aff]

        return admin

    def save_admin_impacts(self, admin_gdf: gpd.GeoDataFrame, out_path: str) -> None:
        self.logger.info(f"Saving classified admin shapefile: {out_path}")
        admin_gdf.to_file(out_path)

    def run(
        self,
        admin_gdf: gpd.GeoDataFrame,
        hazard_raster_path: str,
        hazard_classification: dict[str, float],
        out_vector_path: str | None = None,
        affected_value_threshold: float = 0.0,
    ) -> gpd.GeoDataFrame:
        """
        End-to-end impact stage:
          - compute impacts
          - optionally save to disk

        Returns:
          GeoDataFrame with frac_aff, perc_cells, lev
        """
        admin_out = self.compute_admin_impacts(
            admin_gdf=admin_gdf,
            hazard_raster_path=hazard_raster_path,
            hazard_classification=hazard_classification,
            affected_value_threshold=affected_value_threshold,
        )

        if out_vector_path is not None:
            self.save_admin_impacts(admin_out, out_vector_path)

        return admin_out
