import logging
from common.logging_handler import set_logging_stream
from static.static_flood import make_river_shapefile

# Set up logging
logger_level = logging.INFO
set_logging_stream()
static_directory = "/home/andrea/Projects/Mozambique/{domain}/land"
out_shape = "/home/andrea/Projects/Mozambique/{domain}/{domain}.river_area.shp"
area_limit = 60
for domain in ["mozambique_ara_sul_1", "mozambique_ara_sul_2", "mozambique_ara_sul_3", "mozambique_ara_norte_1", "mozambique_ara_norte_2", "mozambique_zambeze"]:
    make_river_shapefile(static_directory, out_shape, area_limit, domain, dissolve=True)