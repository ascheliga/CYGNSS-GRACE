def load_CYGNSS_05(cygnss_filename = 'CYGNSS_watermask_0_5_with_lakes.nc',
    cygnss_filepath = '/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CYGNSS/Data/CYGNSS_L1_v3_1_data_products/Monthly_maps_watermasks_glob_netCDF/WetCHARTs_size_0_5_deg/',
    # xarray_loc = '/global/home/users/ann_scheliga/.local/lib/python3.7/site-packages/xarray'
    ):
    """
    Inputs
    ------
    cygnss_filename : str
        default = 'CYGNSS_watermask_0_5_with_lakes.nc'
    cygnss_filepath : str
        default = '/global/scratch/users/cgerlein/fc_ecohydrology_scratch/CYGNSS/Data/CYGNSS_L1_v3_1_data_products/Monthly_maps_watermasks_glob_netCDF/WetCHARTs_size_0_5_deg/'
    
    Outputs
    -------
    fw : xarray.DataArray
    """
    import sys
    # sys.path.append(xarray_loc)
    import xarray as xr
    import matplotlib.pyplot as plt
    cygnss_raw = xr.open_dataset(cygnss_filepath+cygnss_filename,decode_coords='all', decode_times=False)
    fw=cygnss_raw['fw']
    return fw

def load_GRACE(grace_filename = 'gsfc.glb_.200204_202211_rl06v2.0_obp-ice6gd.h5',
    grace_filepath = '/global/scratch/users/ann_scheliga/',
    land_subset = True,
    uncertainty = False,
    formatting = True):
    """

    Long description
    ----------------


    Inputs
    ------
    grace_filename : str 
        GRACE HDF5 filename without path location
        default = 'gsfc.glb_.200204_202211_rl06v2.0_obp-ice6gd.h5'
    grace_filepath : str
        absolute filepath to GRACE HDF5
        default = '/global/scratch/users/ann_scheliga/'
    land_subset : boolean
        whether to restrict GRACE output to just land locations
        default = True
    uncertainty : boolean
        whether to include uncertainty data in output.
        data loading takes much longer when uncertainties are included
        default = False
    formatting : boolean 
        whether to include labeling and dtype formatting options to the output
        default = True

    Outputs
    -------
    """
    import numpy as np
    import pandas as pd
    import h5py
    import geopandas as gpd
    from shapely.geometry import Polygon
    f = h5py.File(grace_filepath + grace_filename,'r')
    grace_dict = dict()

    # MASCONS #
    mascon_cols = list(f['mascon'])       # grab dataset names in mascon group
    mascon_cols.remove('location_legend') # remove unused dataset name
    mascon_df = pd.DataFrame()            # create empty pd.DataFrame
    for key in mascon_cols:               # fill df
        mascon_df[key] = np.array(f['mascon'][key]).T.squeeze()
    # Convert longitude from [0 to 360] to (-180 to 180]
    mascon_df.loc[mascon_df['lon_center'] >180,'lon_center'] = mascon_df.loc[mascon_df['lon_center'] >180,'lon_center'] - 360
    if land_subset:
        land_bool = mascon_df['location'] == 80
        mascon_df = mascon_df.loc[land_bool,:]
    if formatting:
        mascon_df.index = mascon_df['labels'].astype(int)
   # Convert from lat/lon coordinates to polygons then to GeoDataFrame
    coord_corners = pd.DataFrame(columns = ['NE','SE','SW','NW','close'])
    min_lon = mascon_df['lon_center'] - mascon_df['lon_span']/2
    min_lat = mascon_df['lat_center'] - mascon_df['lat_span']/2
    max_lon = mascon_df['lon_center'] + mascon_df['lon_span']/2
    max_lat = mascon_df['lat_center'] + mascon_df['lat_span']/2
    coord_corners['NE'] = list(zip(max_lon , max_lat))
    coord_corners['SE'] = list(zip(max_lon , min_lat))
    coord_corners['SW'] = list(zip(min_lon , min_lat))
    coord_corners['NW'] = list(zip(min_lon , max_lat))
    coord_corners['close'] = coord_corners['NE']
    coord_geom = coord_corners.apply(Polygon,axis=1)
    mascon_gdf = gpd.GeoDataFrame(data=mascon_df,geometry=coord_geom.values,crs="EPSG:4326")
    grace_dict['mascon'] = mascon_gdf

    # DATES #
    start_date = pd.Timestamp('2001-12-31')
    time_cols = list(f['time'])                # grab dataset names in time group
    time_cols.remove('list_ref_days_solution') # remove unused dataset name
    time_df = pd.DataFrame()                   # create empty pd.DataFrame for reference dates
    for key in time_cols[2:-1]:                # fill df with days since reference day
        time_df[key] = np.array(f['time'][key]).T.squeeze()
    date_df = time_df.apply(lambda x: pd.to_datetime(x, unit='D',origin=start_date),axis=1)
    date_df.columns = ['date_first','date_last','date_middle']
    date_df[['year_middle','doy_middle','frac_year_middle']] = pd.DataFrame(f['time']['yyyy_doy_yrplot_middle']).T
    grace_dict['date'] = date_df

    # CMWE SOLUTIONS #
    cmwe = pd.DataFrame(f['solution']['cmwe'])
    if land_subset:
        cmwe = cmwe.loc[land_bool,:]
    if formatting:
        cmwe.columns = date_df['date_middle']
        cmwe.index = mascon_df['labels'].astype(int)
    grace_dict['cmwe'] = cmwe

    # UNCERTAINTY #
    if uncertainty:
        uncertainty_cols = list(f['uncertainty'])
        uncertainty_df = pd.DataFrame()
        for key in uncertainty_cols[:-1]:
            uncertainty_df[key] = np.array(f['uncertainty'][key]).T.squeeze()
        noise_df = pd.DataFrame(f['uncertainty']['noise_2sigma'])
        uncertainty_df = pd.concat([uncertainty_df, noise_df],axis=1)
        if land_subset:
            uncertainty_df = uncertainty_df.loc[land_bool,:]
        if formatting:
            mascon_df.index = mascon_df['labels'].astype(int)
        grace_dict['uncertainty'] = uncertainty_df

    return grace_dict

def load_GRanD(GRanD_filename = 'GRanD_reservoirs_v1_3.shp',
    GRanD_filepath = '/global/scratch/users/ann_scheliga/dam_datasets/'):
    """
    Inputs
    ------
    GRanD_filename : str
        reservoir data filename without filepath
        default = 'GRanD_reservoirs_v1_3.shp'
    GRanD_filepath : str
        absolute filepath
        default = '/global/scratch/users/ann_scheliga/dam_datasets/'

    Outputs
    -------
    res_shp : GeoDataFrame
        un-formatted and unedited GRanD reservoir dataset
    """
    import geopandas as gpd
    res_shp = gpd.read_file(GRanD_filepath + GRanD_filename)
    return res_shp

def load_IMERG(imerg_filename = 'IMERG_allmonths_201801_202304_xr.nc',
    imerg_filepath = '/global/scratch/users/ann_scheliga/IMERG_monthly_data/'):
    """
    Inputs
    ------
    imerg_filename : str
        default = 'IMERG_allmonths_201801_202304_xr.nc'
    imerg_filepath : str
        default = '/global/scratch/users/ann_scheliga/IMERG_monthly_data/'
    
    Outputs
    -------
    imerg_raw : xarray.DataArray
    """
    import xarray as xr
    imerg_raw = xr.open_dataarray(imerg_filepath+'IMERG_allmonths_201801_202304_xr.nc')
    return imerg_raw

if __name__ == '__main__':
    test = load_IMERG()
    print(test)