import os
from pathlib import Path

from codebase import dataprocessing

## Other variables and filepaths
grdc_dir = "/global/scratch/users/ann_scheliga/aux_dam_datasets/GRDC_CRB/"
met_dir = "/global/scratch/users/ann_scheliga/era5_data/"
res_dir = "/global/scratch/users/ann_scheliga/CYGNSS_daily/time_series/"
basin_data_dir = "/global/scratch/users/ann_scheliga/basin_forcing_processed/"

## TOPO ATTRIBUTES ##
station_fns = [Path(grdc_dir) / f for f in os.listdir(grdc_dir) if "stationbasins" in f]

output_path = Path(basin_data_dir) / "attributes" / "topo_attr.csv"
_ = [dataprocessing.write_topo_features(fn, output_path) for fn in station_fns]

_ = dataprocessing.sort_csv_from_file(
    output_path,
    "grdc_no",
    read_kwargs={"sep": ";"},
    write_kwargs={"header": True, "index": False},
)
