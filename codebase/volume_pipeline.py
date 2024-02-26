from xarray import DataArray
# 1a. subset fw data
# 1b. subset DEM data
# 1c. Align DEM and fw data subsets
def subset_DEM_and_CYGNSS_data_from_name(dam_name: str) -> tuple[DataArray]:
    from . import load_data
    from . import area_subsets
    subset_gpd = area_subsets.check_for_multiple_dams(dam_name,res_shp)
    subset_bbox = subset_gpd.geometry.buffer(0).bounds
    dem_DA = load_data.load_DEM_subset_as_rxrDA(subset_bbox)
    fw_DA = load_data.load_CYGNSS_001_all_months(subset_bbox)
    return dem_DA , fw_DA

def align_DEM_and_CYGNSS_coordinates(dem_DA: DataArray, fw_DA: DataArray) -> tuple[DataArray]:
    from numpy import testing
    testing.assert_allclose(dem_DA['lat'].values, fw_DA['lat'].values)
    dem_DA['lat'] = fw_DA['lat']
    testing.assert_allclose(dem_DA['lon'].values, fw_DA['lon'].values)
    dem_DA['lon'] = fw_DA['lon']
    return dem_DA , fw_DA

def format_CYGNSS_data_to_binary(fw_DA: DataArray) -> DataArray:
    from . import area_calcs
    from . import time_series_calcs
    fw_DA_binary = area_calcs.cygnss_convert_to_binary(fw_DA)
    fw_DA_binary['time'] = time_series_calcs.CYGNSS_timestep_to_pdTimestamp(fw_DA['time'])
    return fw_DA_binary

def subset_and_align_DEM_CYGNSS_data(dam_name: str) -> tuple[DataArray]:
    dem_DA , fw_DA = subset_DEM_and_CYGNSS_data_from_name(dam_name)
    dem_DA , fw_DA = align_DEM_and_CYGNSS_coordinates(dem_DA, fw_DA)
    fw_DA = format_CYGNSS_data_to_binary(fw_DA)
    return dem_DA , fw_DA
# 2. Create change over time fw dataArray

# 3. Assign each time step as shrinking or expanding

# 4a. Fit distribution of start timestep DEM
# 4b. Fit distribution of shrink/expand DEM

# 5. Calculate change in height from difference in distribution

# 6. Calculate area of start timestep DEM

# 7. Calculate volume from change in height x area