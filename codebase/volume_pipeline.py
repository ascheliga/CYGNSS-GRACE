from geopandas import GeoDataFrame
from numpy.typing import ArrayLike
from xarray import DataArray


# 1a. subset fw data
# 1b. subset DEM data
# 1c. Align DEM and fw data subsets
def subset_DEM_and_CYGNSS_data_from_name(
    dam_name: str, res_shp: GeoDataFrame
) -> tuple[DataArray, DataArray]:
    from . import area_subsets, load_data

    subset_gpd = area_subsets.check_for_multiple_dams(dam_name, res_shp)
    subset_bbox = subset_gpd.geometry.buffer(0).bounds
    dem_DA = load_data.load_DEM_subset_as_rxrDA(subset_bbox)
    fw_DA = load_data.load_CYGNSS_001_all_months(subset_bbox)
    return dem_DA, fw_DA


def align_DEM_and_CYGNSS_coordinates(
    dem_DA: DataArray, fw_DA: DataArray
) -> tuple[DataArray, DataArray]:
    from numpy import testing

    testing.assert_allclose(dem_DA["lat"].values, fw_DA["lat"].values)
    dem_DA["lat"] = fw_DA["lat"]
    testing.assert_allclose(dem_DA["lon"].values, fw_DA["lon"].values)
    dem_DA["lon"] = fw_DA["lon"]
    return dem_DA, fw_DA


def format_CYGNSS_data_to_binary(fw_DA: DataArray) -> DataArray:
    from . import area_calcs, time_series_calcs

    fw_DA_binary = area_calcs.cygnss_convert_to_binary(fw_DA)
    fw_DA_binary["time"] = time_series_calcs.CYGNSS_timestep_to_pdTimestamp(
        fw_DA["time"]
    )
    return fw_DA_binary


def create_aligned_DEM_CYGNSS_subsets(
    dam_name: str, res_shp: GeoDataFrame
) -> tuple[DataArray, DataArray]:
    dem_DA, fw_DA = subset_DEM_and_CYGNSS_data_from_name(dam_name, res_shp)
    dem_DA, fw_DA = align_DEM_and_CYGNSS_coordinates(dem_DA, fw_DA)
    fw_DA = format_CYGNSS_data_to_binary(fw_DA)
    return dem_DA, fw_DA


# 2. Create change over time fw dataArray
def difference_over_time(input_DA: DataArray, dim_label: str = "time") -> DataArray:
    diff_DA = input_DA.diff(dim_label)
    return diff_DA


# 3. Assign each time step as shrinking or expanding
def decide_expansion_or_shrinkage_timestep(input_DA: DataArray) -> int:
    expand_count = (input_DA == 1).sum()
    shrink_count = (input_DA == -1).sum()
    if expand_count * 2 <= shrink_count:
        return -1
    elif shrink_count * 2 <= expand_count:
        return 1
    else:
        return 0


def decide_expansion_or_shrinkage_vectorize(
    input_DA: DataArray, input_core_dims: list[str] | None = None
) -> DataArray:
    if input_core_dims is None:
        input_core_dims = ["lat", "lon"]
    from xarray import apply_ufunc

    change_type_DA = apply_ufunc(
        decide_expansion_or_shrinkage_timestep,
        input_DA,
        input_core_dims=[input_core_dims],
        vectorize=True,
    )
    return change_type_DA


# 4a. Fit distribution of start timestep DEM
# 4b. Fit distribution of shrink/expand DEM
def grab_data_array_values(input_DA: DataArray) -> ArrayLike:
    from numpy import isnan, squeeze

    vals_nparray = squeeze(input_DA.values.reshape((-1, 1)))
    vals_nparray_nonnan = vals_nparray[~isnan(vals_nparray)]
    return vals_nparray_nonnan


def grab_DEM_of_conditional_area(
    dem_DA: DataArray, cond_DA: DataArray, cond: int = 1
) -> DataArray:
    dem_cond_area = dem_DA.where(cond_DA == cond)
    return dem_cond_area


def fit_distribution_from_dataarray(
    input_DA: DataArray, distribution_name
) -> tuple[float, ...]:
    data_as_nparray = grab_data_array_values(input_DA)
    fit_params = distribution_name.fit(data_as_nparray)
    return fit_params


def fit_DEM_distribution_from_conditional_area(
    dem_DA: DataArray, cond_DA: DataArray, cond: int, distribution_name
) -> tuple[float, ...]:
    dem_cond_area = grab_DEM_of_conditional_area(dem_DA, cond_DA, cond)
    fit_params = fit_distribution_from_dataarray(dem_cond_area, distribution_name)
    return fit_params


# 5. Calculate change in height from difference in distribution
def loop_through_time_series_to_get_fit_params(dem_DA, cond_DA: DataArray, cond: int | DataArray, distribution_name):
    """
    Exists as a loop until I can figure out a way to vectorize this.
    I have not tested if this works.
    """
    import numpy as np
    t_max = len(cond_DA['time'])
    fit_params_array = np.empty(t_max)
    for t in np.arange(t_max):
        if isinstance(cond,int):
            cond_i = cond
        else:
            cond_i = con.isel(time=t)
        fit_params_array[t] = fit_DEM_distribution_from_conditional_area(dem_DA, cond_DA.isel(time=t), cond_i, distribution_name)
    return fit_params_array

def calculate_height_from_difference_in_norm_dist(norm_params_0: tuple[float,...],norm_params_1: tuple[float,...]) -> float:
    delta_h = norm_params_1[0] - norm_params_0[0] # mean minus mean
    return delta_h

def calculate_height_time_series_from_start_and_change_in_DEM(dem_DA: DataArray,fw_DA: DataArray, fw_diff_DA: DataArray,change_type_DA: DataArray) -> ArrayLike:
    from scipy.stats import norm
    start_timestep_params = loop_through_time_series_to_get_fit_params(dem_DA, fw_DA, 1, norm)
    change_in_area_params = loop_through_time_series_to_get_fit_params(dem_DA, fw_diff_DA,change_type_DA,norm)

    t_max = len(change_in_area_params)
    height_array = np.empty(t_max)
    i = 0
    for i_params_0, i_params_1 in zip(start_timestep_params[:t_max] , change_in_area_params):
        heights_array[i] = calculate_height_from_difference_in_norm_dist(i_params_0, i_params_1)
        i+=1


# 6. Calculate area of start timestep DEM

# 7. Calculate volume from change in height x area
