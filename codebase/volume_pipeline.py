from geopandas import GeoDataFrame
from numpy.typing import ArrayLike
from xarray import DataArray


def subset_DEM_and_CYGNSS_data_from_name(
    dam_name: str, res_shp: GeoDataFrame
) -> tuple[DataArray, DataArray]:
    """
    Load clipped DEM and CYGNSS data by name of dam.

    Long Description
    ----------------
    Grab the reservoir row from input `res_shp` GeoDataFrame.
    Convert reservoir geometry into bounding box pd.DataFrame.
    Load DEM and CYGNSS data within the bounding box.

    Inputs
    ------
    dam_name : str
        Name of dam in reservoir dataset.
        Case insensitive
    res_shp : (Geo)DataFrame
        DataFrame of reservoirs to subset from.
        Looks for `dam_name` input in column named 'DAM_NAME'

    Outputs
    -------
    dem_DA : rxr.DataArray
        0.01deg DEM
    fw_DA : rxr.DataArray
        0.01deg CYGNSS binary water mask
    """
    from . import area_subsets, load_data

    subset_gpd = area_subsets.check_for_multiple_dams(dam_name, res_shp)
    subset_bbox = subset_gpd.geometry.buffer(0).bounds
    dem_DA = load_data.load_DEM_subset_as_rxrDA(subset_bbox)
    fw_DA = load_data.load_CYGNSS_001_all_months(subset_bbox)
    return dem_DA, fw_DA


def align_DEM_and_CYGNSS_coordinates(
    dem_DA: DataArray, fw_DA: DataArray, coord_names: list[str] | None = None
) -> tuple[DataArray, DataArray]:
    """
    Reset DEM xr.DataArray coordinates to match CYGNSS coordinates.

    Long Description
    ----------------
    Assert the coordinate values are close,
    then set the DEM coordintes to the corresponding CYGNSS values.

    Inputs
    ------
    dem_DA , fw_DA : rxr.DataArray
        must be same size
        coordinates must be approximately equal
    coord_names : list (optional)
        default = ["lat","lon"]
        list of coordinate names to align

    Outputs
    -------
    dem_DA , fw_DA : rxr.DataArray
        same size with exactly equal coordinates
    """
    from numpy import testing

    if coord_names is None:
        coord_names = ["lat", "lon"]

    for coord_name in coord_names:
        testing.assert_allclose(dem_DA[coord_name].values, fw_DA[coord_name].values)
        dem_DA[coord_name] = fw_DA[coord_name]
    return dem_DA, fw_DA


def format_CYGNSS_data_to_binary(fw_DA: DataArray, true_val: float = 2) -> DataArray:
    """
    Convert data from categorical to binary with date formatting.

    Long Description
    ----------------
    Convert input CYGNSS data from categorical to binary.
    Convert timestep to pd.Timestamp objects.

    Inputs
    ------
    fw_DA : xr.DataArray
        Must have a "time" coordinate
    true_val : float
        default = 2
        Passed to `cygnss_convert_to_binary` function
        Value to be converted to True/1

    Outputs
    -------
    fw_DA_binary : xr.DataArray
        binary version of input DataArray
        "time" coordinate values as pd.Timestamp
    """
    from . import area_calcs, time_series_calcs

    fw_DA_binary = area_calcs.cygnss_convert_to_binary(fw_DA, true_val)
    fw_DA_binary["time"] = time_series_calcs.CYGNSS_timestep_to_pdTimestamp(
        fw_DA["time"]
    )
    return fw_DA_binary


def create_aligned_DEM_CYGNSS_subsets(
    dam_name: str, res_shp: GeoDataFrame
) -> tuple[DataArray, DataArray]:
    """
    Create DEM and formatted CYGNSS rxr.DataArrays for reservoir subset.

    Long Description
    ----------------
    Load DEM and CYGNSS DataArrays subsetted around reservoir.
    Align the spatial coordinates of DEM and CYGNSS.
    Format data to binary and time to pd.Timestamp.

    Inputs
    ------
    dam_name : str
        Name of dam in reservoir dataset.
        Case insensitive
    res_shp : (Geo)DataFrame
        DataFrame of reservoirs to subset from.
        Looks for `dam_name` input in column named 'DAM_NAME'

    Outputs
    -------
    dem_DA , fw_DA : rxr.DataArray
        extent is the bounding box of the reservoir geometry.
        have matching spatial coordinates (typ. "lat" , "lon")
        "time" coord formatted as pd.Timestamp
    """
    dem_DA, fw_DA = subset_DEM_and_CYGNSS_data_from_name(dam_name, res_shp)
    dem_DA, fw_DA = align_DEM_and_CYGNSS_coordinates(dem_DA, fw_DA)
    fw_DA = format_CYGNSS_data_to_binary(fw_DA)
    return dem_DA, fw_DA


def difference_over_time(input_DA: DataArray, dim_label: str = "time") -> DataArray:
    """
    Calculate the first order difference over time for the DataArray.

    Inputs
    ------
    input_DA ; xr.DataArray
        DataArray to be differenced
    dim_label : str
        default = "time"
        dimension to difference along

    Outputs
    -------
    diff_DA : xr.DataArray
        Differenced DataArray
        dimensions are the upper value of the difference period.
    """
    diff_DA = input_DA.diff(dim_label)
    return diff_DA


# 3. Assign each time step as shrinking or expanding
def decide_expansion_or_shrinkage_timestep(input_DA: DataArray) -> int:
    """
    Assign each time step as shrinking or expanding.

    Long Description
    ----------------
    Uses the entire input to determine if shrinking (-1), expanding (1), or neither (0).
    Determines expanding if 2/3 of pixels are positive.
    Determines shrinking if 2/3 of pixels are negative.

    Inputs
    ------
    input_DA : xr.DataArray
        A single timestep to determine condition of

    Outputs
    -------
    condition : int
        shrinking = -1
        expanfing = 1
        neither = 0
    """
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
def loop_through_time_series_to_get_fit_params(
    dem_DA: DataArray, cond_DA: DataArray, cond: int | DataArray, distribution_name
) -> list[tuple[float, ...] | float]:
    """
    Exists as a loop until I can figure out a way to vectorize this.
    I have not tested if this works.
    """
    import numpy as np

    t_max = len(cond_DA["time"])
    fit_params_list = [np.nan] * t_max
    for t in np.arange(t_max):
        if isinstance(cond, int):
            cond_i = cond
        elif isinstance(cond, DataArray):
            cond_i = cond.isel(time=t)
            if cond_i == 0:
                continue
        fit_params_list[t] = fit_DEM_distribution_from_conditional_area(
            dem_DA, cond_DA.isel(time=t), cond_i, distribution_name
        )
    return fit_params_list


def calculate_height_from_difference_in_norm_dist(
    norm_params_0: tuple[float, ...] | float, norm_params_1: tuple[float, ...] | float
) -> float:
    from numpy import isnan

    if isnan(norm_params_0).any() or isnan(norm_params_1).any():
        delta_h = 0.0
    elif isinstance(norm_params_0, tuple) and isinstance(norm_params_1, tuple):
        delta_h = norm_params_1[0] - norm_params_0[0]  # mean minus mean
    return delta_h


def calculate_height_time_series_from_start_and_change_in_DEM(
    dem_DA: DataArray,
    fw_DA: DataArray,
    fw_diff_DA: DataArray,
    change_type_DA: DataArray,
) -> ArrayLike:
    import numpy as np
    from scipy.stats import norm

    start_timestep_params = loop_through_time_series_to_get_fit_params(
        dem_DA, fw_DA, 1, norm
    )
    change_in_area_params = loop_through_time_series_to_get_fit_params(
        dem_DA, fw_diff_DA, change_type_DA, norm
    )

    t_max = len(change_in_area_params)
    heights_array = np.empty(t_max)
    i = 0
    for i_params_0, i_params_1 in zip(
        start_timestep_params[:t_max], change_in_area_params, strict=True
    ):
        heights_array[i] = calculate_height_from_difference_in_norm_dist(
            i_params_0, i_params_1
        )
        i += 1
    return heights_array


# 6. Calculate area of start timestep DEM
def calculate_rough_area_timestep(input_DA: DataArray) -> float:
    """
    Calculate area based on hard-coded nominal pixel area.
    Placeholder function while I get the pipeline up and running.
    Should replace with function(s) project into 2D space.
    """
    pixel_count = input_DA.sum()
    area_deg2 = 0.01 * 0.01 * pixel_count
    return area_deg2


def calculate_rough_area_vectorize(
    input_DA: DataArray,
    kwargs: dict | None = None,
) -> DataArray:
    if kwargs is None:
        kwargs = {"input_core_dims": [["lat", "lon"]], "vectorize": True}
    from xarray import apply_ufunc

    area_DA = apply_ufunc(calculate_rough_area_timestep, input_DA, **kwargs)
    return area_DA


def project_DA_from_crs_code(input_DA: DataArray, epsg_code: float) -> DataArray:
    import pycrs

    new_crs = pycrs.utils.crscode_to_string("epsg", epsg_code, "ogcwkt")
    output_DA = input_DA.rio.reproject(new_crs)
    return output_DA


# Consider looking at areal_average function in area_calc module.


# 7. Calculate volume from change in height x area
