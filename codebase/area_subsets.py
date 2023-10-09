def check_for_multiple_dams(dam_name,res_shp):
    dam_row = (res_shp['DAM_NAME'].str.lower())==(dam_name.lower())
    n_rows = dam_row.sum()
    if n_rows == 0:
        print('Dam name not found')
    elif n_rows > 1:
        print('Dam name',dam_name,'is redundant.',n_rows,'entires found.')
    return res_shp[dam_row]
def reservoir_name_to_point(dam_name,res_shp,idx = 0):
    """
    Must have already run: `res_shp = load_data.load_GRanD()`

    Inputs
    ------
    dam_name : str
        name of dam in dataset
    res_shp : GeoPandas GeoDataFrame
        from `load_data.load_GRanD()` function
        dataset of GRanD reservoirs
    idx : int
        default = 0
        use to select a specific dam by index if the dam name appears more than once
        ex: the dam name 'Pelican Lake' appears 4 times in the data. use idx = 3 to get the last occurrence

    Outputs
    -------
    coords_oi : tuple
        coordinates Of Interest
        form of (latitude , longitude)
        the given lat and lon values of the reservoir in the dataset
    """
    import numpy as np
    dam_row = (res_shp['DAM_NAME'].str.lower())==(dam_name.lower())
    n_rows = dam_row.sum()
    if n_rows == 0:
        print('Dam name not found')
    elif n_rows > 1 and (n_rows > idx):
        print('Dam name',dam_name,'is redundant.',n_rows,'entires found. Use idx input to help')
        dam_row = dam_row[dam_row].index[idx]
    elif n_rows <= idx:
        print('idx input too large. idx =',idx, 'for',n_rows, 'total dam rows')
    coords_oi = tuple(np.array(res_shp.loc[dam_row,['LAT_DD','LONG_DD']])[0])
    return coords_oi
def grace_point_subset(coords_i,grace_dict,buffer_val=0):
    """
    Must have already run: `grace_dict = load_data.load_GRACE()`
    
    Inputs
    ------
    coords_i: tuple of (lat,lon)
    grace_dict : dictionary of GRACE TWS info
        from `load_data.load_GRACE()`
        uses grace_dict 'mascon' and 'cmwe' keys
    buffer_val : float
        default = 0
        units of decimal degrees
        the extra length to extend the subset in a square from the central coordinate
    
    Outputs
    -------
    cmwe_i: pd.DataFrame
        GRACE cmwe solution of the mascon containing the input point
    mascon_i: pd.DataFrame
        GRACE mascon metadata of the mascon containing the input point
    
    """
    lat = coords_i[0]
    lon = coords_i[1]
    # Check if within longitude range
    lat_max = grace_dict['mascon']['lat_center'] + grace_dict['mascon']['lat_span']/2 + buffer_val
    lat_min = grace_dict['mascon']['lat_center'] - grace_dict['mascon']['lat_span']/2 - buffer_val
    lat_range = (lat>=lat_min) * (lat <= lat_max)
    # Check if within latitude range
    lon_max = grace_dict['mascon']['lon_center'] + grace_dict['mascon']['lon_span']/2 + buffer_val
    lon_min = grace_dict['mascon']['lon_center'] - grace_dict['mascon']['lon_span']/2 - buffer_val
    lon_range = (lon>=lon_min) * (lon <= lon_max)
    
    range_bool = lat_range * lon_range
    
    mascon_i = grace_dict['mascon'].loc[range_bool]
    cmwe_i = grace_dict['cmwe'].loc[mascon_i.index].squeeze()
    if 'geometry' in cmwe_i.index:
        cmwe_i.drop('geometry',axis=index,inplace=True)
    return cmwe_i , mascon_i
def precip_point_subset(coords_i,precip):
    """
    Must have already run: `precip = load_data.load_IMERG()`
    
    Inputs
    ------
    coords_i: tuple of (lat,lon)
    precip : xarray 

    Outputs
    -------
    precip_ts : Pandas Series
        IMERG timeseries with datetime object index
    """
    import numpy as np
    import pandas as pd
    import time_series_calcs
    # Select data
    precip_xr = precip.sel(lat=coords_i[0],lon=coords_i[1],method='nearest')
    dates_precip = time_series_calcs.IMERG_timestep_to_pdTimestamp(precip_xr['time'])
        # Time = seconds since 1980 Jan 06 (UTC), per original HDF5 IMERG file units
    precip_ts = pd.Series(data=precip_xr,index=dates_precip)
    return precip_ts
def cygnss_point_subset(coords_i,fw):
    """


    Inputs
    ------
    coords_i: tuple of (lat,lon)
    fw : xarray
        from fw = load_data.load_CYGNSS_05()

    Outputs
    -------
    precip_ts : Pandas Series
        CYGNSS timeseries with datetime object index
    """
    import numpy as np
    import pandas as pd
    # Select data
    fw_xr = fw.sel(lat=coords_i[0],lon=coords_i[1],method='nearest')
    dates_fw = time_series_calcs.CYGNSS_timestep_to_pdTimestamp(fw_xr['time'])
    fw_ts = pd.Series(data=fw_xr,index=dates_fw)
    return fw_ts
def grace_shape_subset(dam_name,res_shp,grace_dict,buffer_val=0):
    """
    Inputs
    ------
    dam_name : str
        name of dam in dataset
    res_shp : GeoPandas GeoDataFrame
        from `load_data.load_GRanD()` function
        dataset of GRanD reservoirs
        used to subset grace_dict mascons
    grace_dict : dictionary of GRACE TWS info
        from `load_data.load_GRACE()`
        uses grace_dict 'mascon' and 'cmwe' keys
    buffer_val : float
        default = 0
        units of decimal degrees
        the extra length to extend the subset in a square from the central coordinate

    Outputs
    -------
    subsetted_cmwe : pd.DataFrame
        GRACE cmwe solution of the mascons intersecting the input reservoir
    subsetted_mascon : pd.DataFrame
        GRACE mascon metadata of the mascons intersecting the input reservoir
    """
    shape_row = check_for_multiple_dams(dam_name,res_shp)
    shape_poly = shape_row['geometry'].buffer(buffer_val).values[0]
    bool_series = grace_dict['mascon'].intersects(shape_poly)
    subsetted_mascon = grace_dict['mascon'][bool_series]
    subsetted_cmwe = grace_dict['cmwe'][bool_series]
    return subsetted_cmwe , subsetted_mascon
def xr_shape_subset(dam_name,res_shp,input_xr,buffer_val=0,crs_code = 4326):
    """
    Inputs
    ------
    dam_name : str
        name of dam in dataset
    res_shp : GeoPandas GeoDataFrame
        from `load_data.load_GRanD()` function
        dataset of GRanD reservoirs
        used to subset input_xr
    input_xr : xarray DataArray
        must have 'lat' and 'lon' substrings in coord names
    buffer_val : float
        default = 0
        units of decimal degrees
        the extra length to extend the subset in a square from the central coordinate
    crs_code : int
        default = 4326 (WGS84)
        EPSG code for coordinate reference system
        crs_code applied to input_xr

    Outputs
    -------
    clip_rxr : xarray DataArray
        input_xr subset to the input reservoir
    """
    subset_gpd = check_for_multiple_dams(dam_name,res_shp)
    # Add crs to xr
    full_rxr = input_xr.rio.write_crs(crs_code)

    # Grab coordinate names
    x_name = [dim for dim in list(input_xr.dims) if 'lon' in dim][0]
    y_name = [dim for dim in list(input_xr.dims) if 'lat' in dim][0]

    # Set spatial dimensions to xr
    full_rxr.rio.set_spatial_dims(x_name,y_name,inplace=True)

    # Apply shp subset
    clip_rxr = full_rxr.rio.clip(subset_gpd.geometry.buffer(buffer_val) , subset_gpd.crs)
    return clip_rxr