def load_CYGNSS_05(cygnss_filename = 'CYGNSS_watermask_0_5_with_lakes.nc',
    cygnss_filepath = '/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CYGNSS/Data/CYGNSS_L1_v3_1_data_products/Monthly_maps_watermasks_glob_netCDF/WetCHARTs_size_0_5_deg/',
    xarray_loc = '/global/home/users/ann_scheliga/.local/lib/python3.7/site-packages/xarray'):
    import sys
    sys.path.append(xarray_loc)
    import xarray as xr
    import matplotlib.pyplot as plt
    cygnss_raw = xr.open_dataset(cygnss_filepath+cygnss_filename, decode_times=False)
    fw=cygnss_raw['fw']
    return fw
