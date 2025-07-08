# Import packages
import re
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike


def stat_check(
    input_df: pd.DataFrame, condition: str, pcut: float
) -> tuple[pd.DataFrame, pd.Series | list]:
    # def stat_check(input_df, condition, pcut):
    """
    Mask dataframe by slope sign (positive or negative) and p-value cutoff.

    Long Description
    ----------------

    Inputs
    ------
    input_df : Pandas DataFrame
        must have 'slope' and 'p-value' columns
    condition : str
        must have 'pos' or 'neg' for positive or negative slope
    pcut : float
        p-value cutoff to determine statistically-significant slopes

    Outputs
    -------
    ouput_df : Pandas DataFrame
        input_df sliced to only rows that meet the input condition
    bool_vec : Pandas Series
        boolean series with True for dataframe rows that meet the input condition
    """
    if "neg" in condition.lower():
        bool_vec = (input_df["slope"] < 0) & (input_df["p_value"] < pcut)
    elif "pos" in condition.lower():
        bool_vec = (input_df["slope"] > 0) & (input_df["p_value"] < pcut)
    else:
        bool_vec = []
        print("Invalid condition given")
    output_df = input_df[bool_vec]
    return output_df, bool_vec


def pos_neg_area_calc(input_df: pd.DataFrame, pcut: float) -> tuple[float, float]:
    """
    Provide the total area (km^2) that has a significant positive or negative trend.

    Long Description
    ----------------

    Inputs
    ------
    input_df : Pandas DataFrame
        dataframe to perform area calculations on
        must have 'slope', 'p-value', and 'area' columns
    p_cut : float
        p-value cutoff to determine statistically-significant slopes

    Outputs
    -------
    pos_area_km2, neg_area_km2 : float
        area in km^2 that has statistically positive and negative (respectively) slope
    """
    neg_df, _ = stat_check(input_df, "neg", pcut)
    pos_df, _ = stat_check(input_df, "pos", pcut)

    areacol_list = [
        input_df.columns.get_loc(col) for col in input_df.columns if "area" in col
    ]
    areacol = areacol_list[0]
    neg_area_km2 = neg_df.iloc[:, areacol].sum()
    pos_area_km2 = pos_df.iloc[:, areacol].sum()
    return pos_area_km2, neg_area_km2


def area_frac_calc(
    metrics_3dfs: list[pd.DataFrame],
    pcut: float,
    col_labels: list | None = None,
    idx_labels: list | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Provide fractional and total land area with a stat-sig trend for 3 datasets.

    Long Description
    ----------------
    Wrapper function of `pos_neg_area_calc` function for three input dataframes.

    Inputs
    ------
    metrics_3dfs : list of DataFrames
        must have 3 dataframes in list
        each dataframe must contain a 'slope', 'p-value', and 'area' column
    pcut : float
        p-value cutoff to determine statistically-significant slopes
    col_labels : list
        default = ['pos','neg']
        first two column labels of output dataframe
    idx_labels : list
        default = [0,1,2]

    Outputs
    -------
    frac_df : Pandas DataFrame
        3x3 dataframe containing positive, negative, and non-significant trend areas
         as fraction of total area
    km2_df : Pandas DataFrame
        3x3 dataframe containing positive, negative, and non-significant trend areas
         in units of square kilometers
    """
    if idx_labels is None:
        idx_labels = [0, 1, 2]
    if col_labels is None:
        col_labels = ["pos", "neg"]
    km2_data = np.array([pos_neg_area_calc(df, pcut) for df in metrics_3dfs])

    # Back-up version of km2_data calc
    # pos_area_0 , neg_area_0 = pos_neg_area_calc(metrics_3dfs[0],pcut)
    # pos_area_1 , neg_area_1 = pos_neg_area_calc(metrics_3dfs[1],pcut)
    # pos_area_2 , neg_area_2 = pos_neg_area_calc(metrics_3dfs[2],pcut)
    # km2_data = [[pos_area_0 , neg_area_0],
    #             [pos_area_1 , neg_area_1],
    #             [pos_area_2 , neg_area_2]]
    km2_df = pd.DataFrame(km2_data, columns=col_labels, index=idx_labels).astype(int)
    print("\nLand area in km^2\n---\n", km2_df)

    # get indices of the columns that contain pixel areas
    area_cols = [
        next(df.columns.get_loc(col) for col in df.columns if "area" in col)
        for df in metrics_3dfs
    ]

    # divide km2 values by total area
    frac_data = [
        km2_data.iloc[0, :] / (metrics_3dfs[0].iloc[:, area_cols[0]].sum()),
        km2_data.iloc[1, :] / metrics_3dfs[1].iloc[:, area_cols[1]].sum(),
        km2_data.iloc[2, :] / metrics_3dfs[2].iloc[:, area_cols[2]].sum(),
    ]
    # Add in non-significant fraction
    full_frac_data = [np.append(arr, 1 - np.sum(arr)) for arr in frac_data]

    # convert to dataframe
    col_labels.append("non")
    frac_df = pd.DataFrame(full_frac_data, columns=col_labels, index=idx_labels)

    print("\nFraction of total land\n---\n", frac_df)
    return frac_df, km2_df


def GRACE_areal_average(
    input_cmwe: pd.DataFrame, input_mascon: pd.DataFrame
) -> pd.Series:
    """
    Calculate the weighted average from given cmwe and mascons.

    Inputs
    ------
    input_cmwe : pd.DataFrame
        GRACE cmwe solutions
        each row is a mascon, each column is a timestep
    input_mascon : pd.DataFrame
        GRACE mascon metadata with a column called 'area_km2'
    """
    areal_average = input_cmwe.mul(input_mascon["area_km2"], axis="index").sum(
        axis=0
    ) / (input_mascon["area_km2"].sum())
    return areal_average


def cygnss_convert_to_binary(
    cygnss_DA: xr.DataArray, true_val: float | str = 2
) -> xr.DataArray:
    """
    Convert categorical CYGNSS maps to binary int.

    Long Description
    ----------------
    Does not convert from int to bool as some calculations require numeric data.
    Due to limited xarray-compatible fucntions, uses a two-step process:
        first, converts non-surface water to 0,
        second converts remaining non-zero values to 1.
    The true_val cannot be zero.

    Inputs
    ------
    cygnss_DA : xarray.DataArray
        data array with values 1-4
    true_val : numeric
        default = 2
        the value to become True/1

    Outputs
    -------
    convert_TF : DataArray
        values converted to 0/1 binary int
        attribute units and comments re-written
    """
    if not isinstance(cygnss_DA, xr.DataArray):
        raise TypeError("Input must be a DataArray")

    # Turn != 2 to 0
    convert_F = cygnss_DA.where(cygnss_DA == true_val, 0)
    # Turn == 2 to 1
    convert_TF = convert_F.where(convert_F == 0, 1)

    convert_TF.attrs["units"] = "Binary mask of surface water"
    convert_TF.attrs["comments"] = "Surface water = 1, ocean/land/no data = 0"

    return convert_TF


def grab_pixel_sizes_DA(
    input_DA: xr.DataArray, x_dim: str = "x", y_dim: str = "y", precision: int = 4
) -> tuple[ArrayLike, ArrayLike]:
    _x = input_DA.coords[x_dim]
    _y = input_DA.coords[y_dim]

    x_widths = np.unique((_x[1:].values - _x[:-1].values).round(decimals=precision))
    y_widths = np.unique((_y[1:].values - _y[:-1].values).round(decimals=precision))
    return x_widths, y_widths


def check_regular_area_DA(
    input_DA: xr.DataArray, pixel_size_kwargs: dict | None = None
) -> bool:
    if pixel_size_kwargs is None:
        pixel_size_kwargs = {}

    x_widths, y_widths = grab_pixel_sizes_DA(input_DA, **pixel_size_kwargs)

    if (len(x_widths) > 1) or (len(y_widths) > 1):
        return False
    else:
        return True


def grab_dims(input_DA: xr.DataArray) -> tuple[str, str]:
    try:
        x_dim = next(dim for dim in input_DA.dims if "x" in dim)
        y_dim = next(dim for dim in input_DA.dims if "y" in dim)
        print("Grabbed x/y dims")
    except Exception:
        try:
            x_dim = next(dim for dim in input_DA.dims if "lon" in dim)
            y_dim = next(dim for dim in input_DA.dims if "lat" in dim)
            print("Grabbed lat/lon dims. Consider reprojecting!")
        except Exception:
            raise ValueError("No dimensions found") from Exception
    finally:
        if not check_regular_area_DA(input_DA, {"x_dim": x_dim, "y_dim": y_dim}):
            raise Exception("Unequal pixel areas")
    return x_dim, y_dim


def reproject_to_equal_area(
    DA: xr.DataArray, x_dim: str, y_dim: str
) -> tuple[xr.DataArray, str, str]:
    """
    Reprojects to equal area projection and passes names of spatial dims.

    Long Description
    ----------------
    If crs name does not include 'area', reprojects to equal area ("ESRI:54017").
    If already projected, checks that the projection is equal area.

    Inputs
    ------
    DA: xarray.DataArray
    x_dim: str
        name of x-dimension in DataArray
        typical values: 'longitude', 'x'
    y_dim: str
        name of y-dimension in DataArray
        typical values: 'latitude', 'y'

    Outputs
    -------
    DA: xarray.DataArray
        reprojected DataArray
    x_dim: str
        name of x-dimension in reprojected DataArray
        typical values: 'longitude', 'x'
    y_dim: str
        name of y-dimension in reprojected DataArray
        typical values: 'latitude', 'y'
    """
    if "area" not in DA.spatial_ref.grid_mapping_name:
        DA = DA.rio.reproject("ESRI:54017")
        # Rename dims
        x_dim = next(dim for dim in DA.dims if "x" in dim)
        y_dim = next(dim for dim in DA.dims if "y" in dim)
        print("Projected to equal area")
    elif not check_regular_area_DA(DA, {"x_dim": x_dim, "y_dim": y_dim}):
        raise Exception("Unequal pixel areas")
    return DA, x_dim, y_dim


def CYGNSS_001_areal_average(
    cygnss_DA: xr.DataArray,
    x_dim: str = "x",
    y_dim: str = "y",
    with_index: str = "",
) -> np.ndarray | pd.Series:
    """
    Calculate the average of values in the provided DataArray.

    Long Description
    ----------------
    First reprojects to equal area or checks that the projection is equal area,
    then takes the unweighted average using np.nanmean.

    Inputs
    ------
    cygnss_DA : xarray.DataArray
        all non-nan values in the DataArray will contribute to the average.
    with_index : str
        coordinate in DataArray to serve as Pandas index
        ex: 'time'
    """
    import numpy as np
    from pandas import Series

    cygnss_DA, x_dim, y_dim = reproject_to_equal_area(cygnss_DA, x_dim, y_dim)

    # Average across spatial dims
    _x_dim_idx = cygnss_DA.dims.index(x_dim)
    _y_dim_idx = cygnss_DA.dims.index(y_dim)
    average = np.nanmean(cygnss_DA.values, axis=(_x_dim_idx, _y_dim_idx))

    if with_index:
        average = Series(data=average, index=cygnss_DA[with_index])

    return average


def CYGNSS_001_area_calculation(
    cygnss_DA: xr.DataArray, x_dim: str = "x", y_dim: str = "y", with_index: bool = True
) -> np.ndarray | pd.Series:
    """
    Calculate the sum of values in the provided DataArray.

    Long Description
    ----------------
    First reprojects to equal area or checks that the projection is equal area,
    then takes the summation using np.sum.

    Inputs
    ------
    cygnss_DA : xarray.DataArray
        all non-nan values in the DataArray will contribute to the average.
    with_index : bool
        whether to convert the array into a Series indexed on "time"
    """
    import numpy as np
    from pandas import Series

    cygnss_DA, x_dim, y_dim = reproject_to_equal_area(cygnss_DA, x_dim, y_dim)

    _x_width, _y_width = grab_pixel_sizes_DA(cygnss_DA, x_dim, y_dim)

    _x_dim_idx = cygnss_DA.dims.index(x_dim)
    _y_dim_idx = cygnss_DA.dims.index(y_dim)
    area_array = np.sum(cygnss_DA.values, axis=(_x_dim_idx, _y_dim_idx)) * (
        np.abs(_x_width) * np.abs(_y_width)
    )
    if with_index:
        area_array = Series(data=area_array, index=cygnss_DA["time"])
    return area_array


def project_DA_from_crs_code(input_DA: xr.DataArray, epsg_code: float) -> xr.DataArray:
    """
    Project input to given crs. It works.

    Inputs
    ------
    input_DA: xr.DataArray
        DataArray to reproject.
    epsg_code: float
        if epsg_code=0, will return input_DA without reprojection

    Outputs
    -------
    output_DA: DataArray
        reprojected DataArray
    """
    import pycrs

    if epsg_code == 0:
        output_DA = input_DA
    else:
        new_crs = pycrs.utils.crscode_to_string("epsg", epsg_code, "ogcwkt")
        output_DA = input_DA.rio.reproject(new_crs)
    return output_DA


def calculate_area_from_filename(
    filename: str,
    bbox_vals: np.ndarray,
    filepath: str = "/global/scratch/users/ann_scheliga/CYGNSS_daily/",
    ID_pattern: re.Pattern | str = "",
) -> tuple[str, float] | float:
    """
    Calculate area of CYGNSS extent from filename and bounding box.

    Useful wrapper for parallelization.
    Meant to step through separate daily .nc files.
    """
    from codebase.load_data import load_CYGNSS_001_daily
    from codebase.utils import search_with_exception_handling

    subset_DA = load_CYGNSS_001_daily(filename, bbox_vals, filepath)
    output_area = CYGNSS_001_area_calculation(subset_DA, with_index=False)[0]
    if ID_pattern:
        ID = search_with_exception_handling(ID_pattern, filename)
        output = ID, output_area
    else:
        output = output_area
    return output


def CYGNSS_001_areal_aggregation(
    function: Any,
    cygnss_DA: xr.DataArray,
    x_dim: str = "x",
    y_dim: str = "y",
    with_index: str = "",
) -> np.ndarray | pd.Series:
    """
    Calculate a given function over the provided DataArray with reprojection.

    Long Description
    ----------------
    First reprojects to equal area or checks that the projection is equal area,
    then executes the function.

    Inputs
    ------
    cygnss_DA : xarray.DataArray
        all non-nan values in the DataArray will contribute to the average.
    with_index : str
        coordinate in DataArray to serve as Pandas index
        ex: 'time'
    """
    from pandas import Series

    cygnss_DA, x_dim, y_dim = reproject_to_equal_area(cygnss_DA, x_dim, y_dim)

    # Average across spatial dims
    _x_dim_idx = cygnss_DA.dims.index(x_dim)
    _y_dim_idx = cygnss_DA.dims.index(y_dim)
    aggregate = function(cygnss_DA.values, axis=(_x_dim_idx, _y_dim_idx))

    if with_index:
        aggregate = Series(data=aggregate, index=cygnss_DA[with_index])

    return aggregate
