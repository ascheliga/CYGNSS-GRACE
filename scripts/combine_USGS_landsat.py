# **Data download steps**
# 1) Get shell script from NASA Earthdata interface.
#       This is where you specify temporal and spatial bounds.
# 2) Shorten shell script to just desired bands.
#       Use `scripts/filter_USGS_landsat_download.py`
# 3) Run the shortened shell script. Recommended to put the script in its own folder.
# 4) Run below gdal code from Alex Georges to combine files into one.

import itertools
import os
import re

from codebase.area_subsets import combine_landsat_geotiffs
from codebase.utils import search_with_exception_handling

dirpath = "/global/scratch/users/ann_scheliga/aux_dam_datasets/Landsat8/"


## Grab all date options
# List all files in directory
all_files = os.listdir(dirpath)
all_files.sort()  # Order files for easier debugging
# Pattern for the date (20##DOY)
search_pattern = r"(20[0-9]{5})"
# Pull dates from all Landsat files
all_dates = [
    search_with_excpetion_handling(search_pattern, i_file).group(0)
    for i_file in all_files
    if i_file.startswith("HLS")
]
# Get unique dates to loop through
dates = list(set(all_dates))

# Bands of interest to loop through
bands_oi = ["B03", "B05"]
# Creates a list of all date-band combinations
criteria_list = list(itertools.product(dates, bands_oi))

# Create a mosaic geotiff for each combo
for criteria in criteria_list:
    combine_landsat_geotiffs(*criteria)
