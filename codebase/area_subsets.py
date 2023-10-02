def check_for_multiple_dams(dam_name,res_shp):
    dam_row = (res_shp['DAM_NAME'].str.lower())==(dam_name.lower())
    n_rows = dam_row.sum()
    if n_rows == 0:
        print('Dam name not found')
    elif n_rows > 1:
        print('Dam name',dam_name,'is redundant.',n_rows,'entires found.')
        dam_row = dam_row[dam_row]
    return dam_row
def reservoir_name_to_point(dam_name,res_shp,idx = 0):
    """
    Must have already run: `res_shp = load_data.load_GRanD()`

    Inputs
    ------
    dam_name : str
        name of dam in dataset
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
    # Select data
    precip_xr = precip.sel(lat=coords_i[0],lon=coords_i[1],method='nearest')
    dates_precip = np.array(list(map(lambda x: pd.Timestamp('1980-01-06') + pd.DateOffset(seconds=x),precip_xr['time'].values)))
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
    dates_fw = np.array(list(map(lambda x: pd.Timestamp('2018-08-01') + pd.DateOffset(months=x),fw_xr['time'])))
    fw_ts = pd.Series(data=fw_xr,index=dates_fw)
    return fw_ts
def grace_shape_subset(dam_name,res_shp,grace_dict,buffer_val=0):
    shape_row = check_for_multiple_dams(dam_name,res_shp)
    shape_poly = shape_row['geometry'].buffer(buffer_val).values[0]
    bool_series = grace_dict['mascon'].intersects(shape_poly)
    subsetted_mascon = grace_dict['mascon'][bool_series]
    subsetted_cmwe = grace_dict['cmwe'][bool_series]
    return subsetted_cmwe , subsetted_mascon