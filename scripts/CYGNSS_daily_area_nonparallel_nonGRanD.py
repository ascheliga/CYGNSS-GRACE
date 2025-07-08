import glob
import os

import pandas as pd

import codebase

## DEFINE VARIABLES
datadir = "/global/scratch/users/ann_scheliga/CYGNSS_daily/"
subset_bbox = {
    "minx": -81.68618774452528,
    "miny": 28.84624862701476,
    "maxx": -81.42883300742223,
    "maxy": 29.054162979154427,
}
## END DEFINE VARIABLES

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
df["area"] = df["Filename"].apply(
    codebase.area_calcs.calculate_area_from_filename,
    bbox_vals=subset_bbox,
    filepath=datadir,
)
## END AREA CALCULATION

df.index.name = 'date'

## SAVE RESULTS
df.to_csv(datadir + "test_02235200.csv")
## END SAVE RESULTS
