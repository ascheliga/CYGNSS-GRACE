## DEFINE VARIABLES
datadir = '/global/scratch/users/ann_scheliga/CYGNSS_daily/powell/'
dam_name = 'glen canyon'
## END DEFINE VARIABLES

## DASK PARALLELIZATION
# Taken straight from Savio HPC Docs page.

import dask
import codebase
import pandas as pd

# Threads scheduler
dask.config.set(scheduler='threads', num_workers = 24)

# Processes scheduler
import dask.multiprocessing
dask.config.set(scheduler='processes', num_workers = 24)  

# Distributed scheduler 
# Fine to use this on a single node and it provides some nice functionality
# If you experience issues with worker memory then try the processes scheduler
from dask.distributed import Client, LocalCluster
cluster = LocalCluster(n_workers = 24)
c = Client(cluster)
## END DASK SETUP

## DEFINE BBOX
res_shp = codebase.load_data.load_GRanD()
subset_gpd = codebase.area_subsets.check_for_multiple_dams(dam_name, res_shp)
subset_bbox = subset_gpd.geometry.buffer(0).bounds
## END DEFINE BBOX

## LIST OF FILENAMES
# All the CYGNSS daily .nc files
all_files = os.listdir(datadir)
all_files.sort()
## END LIST OF FILENAMES

## PARALLEL FOR LOOP
futures = c.map(func = codebase.area_calcs.calculate_area_from_filename,
    all_files,
    filepath= datadir
    bbox_vals = subset_bbox,
    ID_pattern = r'[0-9]{4}-[0-9]{2}-[0-9]{2}
)
## END PARALLEL FOR LOOP

## COMPILE RESULTS
results = c.gather(futures)
df = pd.DataFrame(results, columns =['ID', 'Area sqm'])
df.to_csv(datadir+'area_calc.csv')

## END COMPILE RESULTS