import os
from pathlib import Path

from codebase import dataprocessing

## Other variables and filepaths
grdc_dir = "/global/scratch/users/ann_scheliga/aux_dam_datasets/GRDC_CRB/"
met_dir = "/global/scratch/users/ann_scheliga/era5_data/"
res_dir = "/global/scratch/users/ann_scheliga/CYGNSS_daily/time_series/"
basin_data_dir = "/global/scratch/users/ann_scheliga/basin_forcing_processed/"

cats_to_run = ['clim'] # 'topo'


## TOPO ATTRIBUTES ##
if 'topo' in cats_to_run:
    station_fns = [
        Path(grdc_dir) / fn for fn in os.listdir(grdc_dir) if "stationbasins" in fn
    ]

    topo_path = Path(basin_data_dir) / "attributes" / "topo_attr.csv"
    _ = [dataprocessing.write_topo_features(fn, topo_path) for fn in station_fns]

    _ = dataprocessing.sort_csv_from_file(
        topo_path,
        "grdc_no",
        read_kwargs={"sep": ";"},
        write_kwargs={"header": True, "index": False},
    )


## CLIM ATTRIBUTES ##
if 'clim' in cats_to_run:
    basin_fns = [
        Path(basin_data_dir) / f for f in os.listdir(basin_data_dir) if ".pkl" in f
    ]

    clim_path = Path(basin_data_dir) / "attributes" / "clim_attr.csv"
    _ = [dataprocessing.write_clim_features(fn, clim_path) for fn in basin_fns]

    _ = dataprocessing.sort_csv_from_file(
        clim_path,
        "grdc_no",
        read_kwargs={"sep": ";"},
        write_kwargs={"header": True, "index": False},
    )
