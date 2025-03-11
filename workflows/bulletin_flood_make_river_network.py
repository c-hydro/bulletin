import logging
from common.logging_handler import set_logging_stream
from static.static_flood import make_river_shapefile

# Set up logging
logger_level = logging.INFO
set_logging_stream()
static_directory = "/home/andrea/backup/Desktop/Working_dir/New Folder 3/PPT_Workshop/Continuum/data_static/{domain}/gridded"
out_shape = "/home/andrea/Desktop/Working_dir/IGAD_IBF/new_dissolved/{domain}/{domain}.river_area.shp"
area_limit = 100
for domain in ["IGAD_D1", "IGAD_D2", "IGAD_D3", "IGAD_D4", "IGAD_D5", "IGAD_D6", "IGAD_D7", "IGAD_D8", "IGAD_D9", "IGAD_D10", "IGAD_D11", "IGAD_D12", "IGAD_D14", "IGAD_D15"]:
    make_river_shapefile(static_directory, out_shape, area_limit, domain, dissolve=True)