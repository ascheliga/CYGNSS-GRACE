import os
import numpy as np
import dask
import dask.multiprocessing
# from pandas import DataFrame
from dask.distributed import Client, LocalCluster

# import codebase

## DEFINE VARIABLES
datadir = "/global/scratch/users/ann_scheliga/CYGNSS_daily/"
subset_bbox = np.array([[1390844.6661,  748605.087 , 1416195.1462,  770265.0627]])
## END DEFINE VARIABLES

## DASK PARALLELIZATION
# Taken straight from Savio HPC Docs page.


# Threads scheduler
dask.config.set(scheduler="threads", num_workers=2)

# Processes scheduler
dask.config.set(scheduler="processes", num_workers=2)

# Distributed scheduler
# Fine to use this on a single node and it provides some nice functionality
# If you experience issues with worker memory then try the processes scheduler
cluster = LocalCluster(n_workers=24)
c = Client(cluster)
## END DASK SETUP


## LIST OF FILENAMES
# All the CYGNSS daily .nc files
all_files = os.listdir(datadir)
all_files.sort()
some_files = all_files[:100]
## END LIST OF FILENAMES

## PARALLEL FOR LOOP
futures = c.map(
    codebase.area_calcs.calculate_area_from_filename,
    some_files,
    filepath=datadir,
    bbox_vals=subset_bbox,
    ID_pattern=r"[0-9]{4}-[0-9]{2}-[0-9]{2}",
)
## END PARALLEL FOR LOOP

## COMPILE RESULTS
results = c.gather(futures)
df = DataFrame(results, columns=["ID", "Area sqm"])
df.to_csv("/global/scratch/users/ann_scheliga/CYGNSS_daily/test_02235200.csv")

## END COMPILE RESULTS
