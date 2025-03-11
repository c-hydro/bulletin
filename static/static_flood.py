import logging
import os
from common.hydro_tools import HydroTools
import geopandas as gpd

def make_river_shapefile(static_directory: str, out_shape: str, area_limit: float, domain: str, dissolve: bool = False) -> None:
    """
    Create a shapefile of the river network based on the specified area limit.

    This function extracts the river network from the static data directory,
    creates a GeoDataFrame, and saves it as a shapefile. Optionally, the
    GeoDataFrame can be dissolved by the 'log_ups' column.

    :param static_directory: Path to the static data directory, formatted with the domain name.
    :param out_shape: Path to the output shapefile, formatted with the domain name.
    :param area_limit: Minimum area limit to include in the river network.
    :param domain: Domain name to format the paths.
    :param dissolve: Whether to dissolve the GeoDataFrame by the 'log_ups' column.
    """
    # Format paths with the domain name
    static_directory = static_directory.format(domain=domain)
    out_shape = out_shape.format(domain=domain)

    # Define output file path
    out_directory = os.path.dirname(out_shape)
    os.makedirs(out_directory, exist_ok=True)

    # Ensure output directory exists
    os.makedirs(out_directory, exist_ok=True)

    # Create HydroTools instance
    hydro_tools = HydroTools(static_directory, domain)

    # Extract network information
    gdf = hydro_tools.extract_river_geodataframe(area_limit, maps_to_extract=None, include_log_area=True)

    # Save GeoDataFrame to shapefile
    if dissolve:
        gdf = gdf.dissolve(by="log_ups")
    gdf.to_file(out_shape, driver="ESRI Shapefile")
    logging.info("Static shapefile has been created successfully.")

