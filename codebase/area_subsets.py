from typing import Any

import pandas as pd
from geopandas import GeoDataFrame
from xarray import DataArray


def check_for_multiple_dams(
    dam_name: str, res_shp: pd.DataFrame, idx: int = -1
) -> pd.DataFrame:
    """
    Subset reservoir dataset by dam name and
    select one reservoir if there are multiple of same name.

    Inputs
    ------
    dam_name : str
        Name of dam in reservoir dataset.
        Case insensitive
    res_shp : (Geo)DataFrame
        DataFrame of reservoirs to subset from.
        Looks for `dam_name` input in column named 'DAM_NAME'
    idx : int
        default = -1
        if `idx` < 0, then will not subset
        use to select a specific dam by index if the dam name appears more than once
        ex: the dam name 'Pelican Lake' appears 4 times in the data.
            use idx = 3 to get the last occurrence

    Outputs
    -------
    subset GeoDataFrame
    """
    dam_row = (res_shp["DAM_NAME"].str.lower()) == (dam_name.lower())
    n_rows = dam_row.sum()
    if n_rows == 0:
        print("Dam name not found")
    elif n_rows > 1:
        print("Dam name", dam_name, "is redundant.", n_rows, "entires found.")
    if idx >= 0:
        return res_shp[dam_row].iloc[[idx]]
    else:
        return res_shp[dam_row]


def reservoir_name_to_point(
    dam_name: str, res_shp: pd.DataFrame, idx: int = 0
) -> tuple[Any, ...]:
    """
    Must have already run: `res_shp = load_data.load_GRanD()`.

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
        ex: the dam name 'Pelican Lake' appears 4 times in the data.
            use idx = 3 to get the last occurrence

    Outputs
    -------
    coords_oi : tuple
        coordinates Of Interest
        form of (latitude , longitude)
        the given lat and lon values of the reservoir in the dataset
    """
    import numpy as np

    dam_row = (res_shp["DAM_NAME"].str.lower()) == (dam_name.lower())
    n_rows = dam_row.sum()
    coords_array = np.array([np.nan, np.nan])
    if n_rows == 0:
        print("Dam name not found")
    elif n_rows <= idx:
        print("idx input too large. idx =", idx, "for", n_rows, "total dam rows")
    elif n_rows > 1 and (n_rows > idx):
        print(
            "Dam name",
            dam_name,
            "is redundant.",
            n_rows,
            "entires found. Use idx input to help",
        )
        dam_row = dam_row[dam_row].index[idx]
        coords_array = np.array(res_shp.loc[dam_row, ["LAT_DD", "LONG_DD"]])
    else:
        coords_array = np.array(res_shp.loc[dam_row, ["LAT_DD", "LONG_DD"]])[0]
    return tuple(coords_array)


def grace_point_subset(
    coords_i: tuple[float, float], grace_dict: dict, buffer_val: float = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Must have already run: `grace_dict = load_data.load_GRACE()`.

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
    lat_max = (
        grace_dict["mascon"]["lat_center"]
        + grace_dict["mascon"]["lat_span"] / 2
        + buffer_val
    )
    lat_min = (
        grace_dict["mascon"]["lat_center"]
        - grace_dict["mascon"]["lat_span"] / 2
        - buffer_val
    )
    lat_range = (lat >= lat_min) * (lat <= lat_max)
    # Check if within latitude range
    lon_max = (
        grace_dict["mascon"]["lon_center"]
        + grace_dict["mascon"]["lon_span"] / 2
        + buffer_val
    )
    lon_min = (
        grace_dict["mascon"]["lon_center"]
        - grace_dict["mascon"]["lon_span"] / 2
        - buffer_val
    )
    lon_range = (lon >= lon_min) * (lon <= lon_max)

    range_bool = lat_range * lon_range

    mascon_i = grace_dict["mascon"].loc[range_bool]
    cmwe_i = grace_dict["cmwe"].loc[mascon_i.index].squeeze()
    if "geometry" in cmwe_i.index:
        cmwe_i.drop("geometry", axis="index", inplace=True)
    return cmwe_i, mascon_i


def precip_point_subset(coords_i: tuple[float, float], precip: DataArray) -> pd.Series:
    """
    Must have already run: `precip = load_data.load_IMERG()`.

    Inputs
    ------
    coords_i: tuple of (lat,lon)
    precip : xarray

    Outputs
    -------
    precip_ts : Pandas Series
        IMERG timeseries with datetime object index
    """
    import pandas as pd

    from . import time_series_calcs

    # Select data
    precip_xr = precip.sel(lat=coords_i[0], lon=coords_i[1], method="nearest")
    dates_precip = time_series_calcs.IMERG_timestep_to_pdTimestamp(precip_xr["time"])
    # Time = seconds since 1980 Jan 06 (UTC), per original HDF5 IMERG file units
    precip_ts = pd.Series(data=precip_xr, index=dates_precip)
    return precip_ts


def cygnss_point_subset(coords_i: tuple[float, float], fw: DataArray) -> pd.Series:
    """
    Subset the given DataArray to the time series at a single point.

    Inputs
    ------
    coords_i: tuple of (lat,lon)
    fw : xarray
        from fw = load_data.load_CYGNSS_05().

    Outputs
    -------
    precip_ts : Pandas Series
        CYGNSS timeseries with datetime object index
    """
    import pandas as pd

    from . import time_series_calcs

    # Select data
    fw_xr = fw.sel(lat=coords_i[0], lon=coords_i[1], method="nearest")
    dates_fw = time_series_calcs.CYGNSS_timestep_to_pdTimestamp(fw_xr["time"])
    fw_ts = pd.Series(data=fw_xr, index=dates_fw)
    return fw_ts


def grace_shape_subset(
    subset_gpd: GeoDataFrame, grace_dict: dict, buffer_val: float = 0
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Subset the input GRACE dictionary to the the input geodataframe.

    Inputs
    ------
    subset_gpd : GeoPandas GeoDataFrame
        row(s) of item(s) to subset the GRACE mascons
    grace_dict : dictionary of GRACE TWS info
        from `load_data.load_GRACE()`
        uses grace_dict 'mascon' and 'cmwe' keys
    buffer_val : float
        default = 0
        units of decimal degrees
        the extra length to extend the subset in a square from the central coordinate.

    Outputs
    -------
    subsetted_cmwe : pd.DataFrame
        GRACE cmwe solution of the mascons intersecting the input reservoir
    subsetted_mascon : pd.DataFrame
        GRACE mascon metadata of the mascons intersecting the input reservoir
    subsetted_cmwe_agg : pd.Series
        areal-weighted average of subsetted_cmwe
    """
    from . import area_calcs

    shape_poly = subset_gpd["geometry"].buffer(buffer_val).unary_union
    bool_series = grace_dict["mascon"].intersects(shape_poly)
    subsetted_mascon = grace_dict["mascon"][bool_series]
    subsetted_cmwe = grace_dict["cmwe"][bool_series]

    subsetted_cmwe_agg = area_calcs.GRACE_areal_average(
        subsetted_cmwe, subsetted_mascon
    )

    return subsetted_cmwe, subsetted_mascon, subsetted_cmwe_agg


def xr_shape_subset(
    subset_gpd: GeoDataFrame,
    input_xr: DataArray,
    buffer_val: float = 0,
    crs_code: int = 4326,
) -> DataArray:
    """
    Subset a generic xarray.DataArray to the provided geodataframe.

    Inputs
    ------
    subset_gpd : GeoPandas GeoDataFrame
        row(s) of item(s) to subset input_xr
    input_xr : xarray DataArray
        must have 'lat' and 'lon' substrings in coord names
    buffer_val : float
        default = 0
        units of decimal degrees
        the extra length to extend the subset in a square from the central coordinate
    crs_code : int
        default = 4326 (WGS84)
        EPSG code for coordinate reference system
        crs_code applied to input_xr.

    Outputs
    -------
    clip_rxr : xarray DataArray
        input_xr subset to the input reservoir
    """
    from codebase.area_calcs import grab_dims

    # Add crs to xr
    full_rxr = input_xr.rio.write_crs(crs_code)

    # Grab coordinate names
    x_name, y_name = grab_dims(input_xr)

    # Set spatial dimensions to xr
    full_rxr.rio.set_spatial_dims(x_name, y_name, inplace=True)

    # Apply shp subset
    clip_rxr = full_rxr.rio.clip(subset_gpd.geometry.buffer(buffer_val), subset_gpd.crs)
    return clip_rxr


def xr_shape_subset_from_filename(
    full_filename: str,
    subset_gpd: GeoDataFrame,
    buffer_val: float = 0,
    crs_code: int = 4326,
) -> DataArray:
    from xarray import open_dataarray

    full_DA = open_dataarray(full_filename)
    subset_DA = xr_shape_subset(subset_gpd, full_DA, buffer_val, crs_code)
    return subset_DA


def cygnss_shape_subset(
    subset_gpd: GeoDataFrame,
    input_xr: DataArray,
    buffer_val: float = 0,
    crs_code: int = 4326,
) -> tuple[DataArray, pd.Series]:
    """
    Subset CYGNSS DataArray to a reservoir.
    Calculate and format the average time series.
    """
    import pandas as pd

    from . import time_series_calcs

    fw_subset_xr = xr_shape_subset(subset_gpd, input_xr, buffer_val, crs_code)

    fw_dates = time_series_calcs.CYGNSS_timestep_to_pdTimestamp(fw_subset_xr["time"])
    fw_agg_series = pd.Series(
        data=fw_subset_xr.mean(dim=["lat", "lon"]), index=fw_dates
    )
    return fw_subset_xr, fw_agg_series


def precip_shape_subset(
    subset_gpd: GeoDataFrame,
    input_xr: DataArray,
    buffer_val: float = 0,
    crs_code: int = 4326,
) -> tuple[DataArray, pd.Series]:
    """Subset precip DataArray to a reservoir.
    Calculate and format the summed time series.
    """
    import pandas as pd

    from . import time_series_calcs

    precip_subset_xr = xr_shape_subset(subset_gpd, input_xr, buffer_val, crs_code)

    precip_dates = time_series_calcs.IMERG_timestep_to_pdTimestamp(
        precip_subset_xr["time"]
    )
    precip_agg_series = pd.Series(
        data=precip_subset_xr.sum(dim=["lat", "lon"]), index=precip_dates
    )
    return precip_subset_xr, precip_agg_series


def era5_shape_subset_and_concat(
    subset_gpd: GeoDataFrame,
    ordered_filenames: list[str],
    filepath: str,
    subset_dict: None | dict = None,
    concat_dict: None | dict = None,
) -> DataArray:
    if concat_dict is None:
        concat_dict = {}
    if subset_dict is None:
        subset_dict = {}
    from xarray import concat

    items_to_concat = [None] * len(ordered_filenames)
    for idx, filename in enumerate(ordered_filenames):
        full_filename = filepath + filename
        items_to_concat[idx] = xr_shape_subset_from_filename(
            full_filename, subset_gpd, **subset_dict
        )
    concat_DA = concat(items_to_concat, **concat_dict)
    return concat_DA


def era5_shape_subset_and_concat_from_file_pattern(
    filepath: str,
    input_pattern: str,
    subset_gpd: GeoDataFrame,
    concat_dict: dict | None = None,
    agg_function: Any = None,
) -> tuple[DataArray, pd.Series | None]:
    if concat_dict is None:
        concat_dict = {}
    from codebase.area_calcs import CYGNSS_001_areal_aggregation
    from codebase.utils import grab_matching_names_from_filepath

    filenames = grab_matching_names_from_filepath(filepath, input_pattern)
    xr_DA = era5_shape_subset_and_concat(
        ordered_filenames=filenames,
        filepath=filepath,
        concat_dict=concat_dict,
        subset_gpd=subset_gpd,
    )

    if agg_function:
        agg_series = CYGNSS_001_areal_aggregation(
            agg_function, xr_DA, with_index=concat_dict["dim"]
        )
    else:
        agg_series = None
    return xr_DA, agg_series


def combine_landsat_geotiffs(
    date_code: str = "",
    band: str = "",
    output_fn: str = "",
    dir_path: str = "/global/scratch/users/ann_scheliga/aux_dam_datasets/Landsat8/",
) -> None:
    """Combine landsat geotiffs by common band and date."""
    # Input handling
    date_code = str(date_code)
    band = str(band)

    # Create a useful default name for the output file
    if not output_fn:
        output_fn = "_".join(["landsat", date_code, band]) + ".tif"

    # Create regex for searching through files in directory
    search_criteria = "*" + date_code + "*" + band + ".tif"

    combine_geotiffs(search_criteria, output_fn, dir_path)
    # print("Searching by:", search_criteria)
    # # Output file path and name
    # out_fp = dir_path + output_fn
    # # Query search
    # q = os.path.join(dir_path, search_criteria)
    # # Gets all the geotiff file paths inside the folder
    # files_to_mosaic = glob.glob(q)
    # # Function to Merge all files
    # gdal.Warp(out_fp, files_to_mosaic, format="GTiff")
    # # Flush the file to local and close it from memory
    # print("Created:", output_fn)


def combine_geotiffs(
    search_criteria: str,
    output_fn: str,
    dir_path: str,
) -> None:
    """Combine geotiffs in directory matching search criteria."""
    import glob
    import os

    from osgeo import gdal

    print("Searching by:", search_criteria)
    # Output file path and name
    out_fp = dir_path + output_fn
    # Query search
    q = os.path.join(dir_path, search_criteria)
    # Gets all the geotiff file paths inside the folder
    files_to_mosaic = glob.glob(q)
    # Function to Merge all files
    gdal.Warp(out_fp, files_to_mosaic, format="GTiff")
    # Flush the file to local and close it from memory
    print("Created:", output_fn)


def hydroBASINS_by_PFAFID(
    PFAF_ID: str | int, hydroBASINS_gdf: GeoDataFrame
) -> GeoDataFrame:
    """Select one or more watersheds from hydroBASINS based on beginning of PFAF_ID."""
    PFAF_ID = str(PFAF_ID)
    subset_rows = hydroBASINS_gdf["PFAF_ID"].astype(str).str.startswith(PFAF_ID)
    return hydroBASINS_gdf[subset_rows]
