import glob
import os
from pathlib import Path

import pandas as pd

import codebase

## DEFINE VARIABLES
datadir = "/global/scratch/users/ann_scheliga/CYGNSS_daily/"
dam_name = "glen canyon"
print('Start CYGNSS daily area calculation for:', dam_name)
## END DEFINE VARIABLES

## DEFINE BBOX
res_shp = codebase.load_data.load_GRanD()
subset_gpd = codebase.area_subsets.check_for_multiple_dams(dam_name, res_shp)
subset_bbox = subset_gpd.geometry.buffer(0).bounds
## END DEFINE BBOX

## LIST OF FILENAMES
# All the CYGNSS daily .nc files
os.chdir(datadir)
nc_list = glob.glob("*.nc")
nc_list.sort()
## END LIST OF FILENAMES

## CREATE EMPTY-ISH DATAFRAME
r_pattern = r"[0-9]{4}-[0-9]{2}-[0-9]{2}"
IDs = [
    codebase.utils.search_with_exception_handling(item=file, r_pattern=r_pattern)
    for file in nc_list
]
df = pd.DataFrame(nc_list, columns=["Filename"], index=IDs)
## END CREATE EMPTY-ISH DATAFRAME

## AREA CALCULATION
print('Start area calculation')
df["Filename"].apply(
    codebase.area_calcs.calculate_area_from_filename,
    bbox_vals=subset_bbox,
    filepath=datadir,
)
print('Finish area calculation')
## END AREA CALCULATION

## SAVE RESULTS
filename = dam_name.replace(' ', '_') + "_area.csv"
df.to_csv(datadir + filename)
## END SAVE RESULTS
