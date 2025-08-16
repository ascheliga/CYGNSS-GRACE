import h5py
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from xarray import DataArray


def load_CYGNSS_05(
    cygnss_filename: str = "CYGNSS_watermask_0_5_with_lakes.nc",
    cygnss_filepath: str = "/global/scratch/users/cgerlein/"
    "fc_ecohydrology_scratch/CYGNSS/Data/CYGNSS_L1_v3_1_data_products/"
    "Monthly_maps_watermasks_glob2_netCDF/WetCHARTs_size_0_5_deg/",
) -> DataArray:
    """
    Load the global, full time series CYGNSS watermask at 0.5deg resolution.

    Inputs
    ------
    cygnss_filename : str
        default = 'CYGNSS_watermask_0_5_with_lakes.nc'
    cygnss_filepath : str
        default = '/global/scratch/users/cgerlein/"\
        "fc_ecohydrology_scratch/CYGNSS/Data/CYGNSS_L1_v3_1_data_products/"\
        "Monthly_maps_watermasks_glob_netCDF/WetCHARTs_size_0_5_deg/'.

    Outputs
    -------
    fw : xarray.DataArray
    """
    import xarray as xr

    cygnss_raw = xr.open_dataset(
        cygnss_filepath + cygnss_filename, decode_coords="all", decode_times=False
    )
    fw = cygnss_raw["fw"]
    return fw


def load_CYGNSS_001_1month(
    filename: str,
    bbox_vals: np.ndarray,
    filepath: str = "/global/scratch/users/cgerlein/"
    "fc_ecohydrology_scratch/CYGNSS/Data/CYGNSS_L1_v3_1_data_products/"
    "Monthly_maps_watermasks_glob2_netCDF/Native_size_0_01_deg/With_lakes/",
) -> DataArray:
    """
    Load and subset a single month of data by filename.

    Inputs
    ------
    filename : str
        Name of one month of data
        Typical form of 'CYGNSS_watermask_2018_08.nc'
    bbox_vals : np.ndarray
        order of values: 'minx', 'miny', 'maxx', 'maxy'
        array of 4 values to feed to DataArray.rio.clip_box()
    filepath : str
        default: "/global/scratch/users/cgerlein/"\
        "fc_ecohydrology_scratch/CYGNSS/Data/CYGNSS_L1_v3_1_data_products/"\
        "Monthly_maps_watermasks_glob2_netCDF/Native_size_0_01_deg/With_lakes/"
        filepath to CYGNSS data

    Outputs
    -------
    clipped_rxr: xr.DataArray

    """
    import xarray as xr

    global_xrDS = xr.open_dataset(filepath + filename, decode_times=False)

    if "watermask" in global_xrDS.data_vars:
        global_xrDS = global_xrDS.rename_vars({"watermask": "Watermask"})

    dem_full_rxr = global_xrDS["Watermask"].rio.write_crs(4326)
    del global_xrDS
    dem_full_rxr.rio.set_spatial_dims("lon", "lat", inplace=True)
    clipped_rxr = dem_full_rxr.rio.clip_box(**bbox_vals)
    del dem_full_rxr
    return clipped_rxr


def load_CYGNSS_001_all_months(
    bbox_vals: pd.DataFrame,
    filepath: str = "/global/scratch/users/cgerlein/"
    "fc_ecohydrology_scratch/CYGNSS/Data/CYGNSS_L1_v3_1_data_products/"
    "Monthly_maps_watermasks_glob2_netCDF/Native_size_0_01_deg/With_lakes/",
) -> DataArray:
    """
    Load all available months of CYGNSS data for a subset area.

    Inputs
    ------
    bbox_vals : pd.DataFrame
        columns of 'minx', 'miny', 'maxx', 'maxy'
        single row of values
        default format from the gpd.bounds attribute
    filepath : str
        default: "/global/scratch/users/cgerlein/"\
        "fc_ecohydrology_scratch/CYGNSS/Data/CYGNSS_L1_v3_1_data_products/"\
        "Monthly_maps_watermasks_glob2_netCDF/Native_size_0_01_deg/With_lakes/"
        filepath to CYGNSS data

    Outputs
    -------
    cygnss_allmonths_xr: xr. DataArray

    """
    import os

    import numpy as np
    import pandas as pd
    import xarray as xr

    filenames = os.listdir(filepath)
    filenames.sort()
    list_of_xr = [
        load_CYGNSS_001_1month(filename, bbox_vals.values[0]) for filename in filenames
    ]
    time_idx = np.arange(len(filenames))
    cygnss_allmonths_xr = xr.concat(
        list_of_xr, pd.Index(time_idx, name="time"), combine_attrs="drop_conflicts"
    )
    return cygnss_allmonths_xr


def load_CYGNSS_001_daily(
    filename: str,
    bbox_vals: np.ndarray,
    filepath: str = "/global/scratch/users/ann_scheliga/CYGNSS_daily/powell/",
) -> DataArray:
    """
    Load and subset a single day of data by filename.

    Inputs
    ------
    filename : str
        Name of one month of data
        Typical form of 'cyg.ddmi.2023-12-26.l3.uc-berkeley-watermask-daily.a32.d33.nc'
    bbox_vals : np.ndarray
        order of values: 'minx', 'miny', 'maxx', 'maxy'
        array of 4 values to feed to DataArray.rio.clip_box()
    filepath : str
        default: "/global/scratch/users/ann_scheliga/CYGNSS_daily/powell/"
        filepath to CYGNSS data

    Outputs
    -------
    clipped_rxr: xr.DataArray

    """
    import xarray as xr

    global_xrDS = xr.open_dataset(filepath + filename, decode_times=False)
    dem_full_rxr = global_xrDS["watermask"].rio.write_crs(4326)
    del global_xrDS
    dem_full_rxr.rio.set_spatial_dims("lon", "lat", inplace=True)
    clipped_rxr = dem_full_rxr.rio.clip_box(**bbox_vals)
    del dem_full_rxr
    return clipped_rxr


def load_GRACE_uncertainty(f: h5py.File) -> pd.DataFrame:
    """
    Load GRACE uncertainty.

    Long Description
    ----------------
    Loads the leakage trend, leakage uncertainty, and noise uncertainty.
    Error of individual mascon: |leakage_trend| + leakage_2sigma + noise_2sigma
    Error of region of mascons: |spatial_av(leakage_trend)| +
        (spatial_av(leakage_2sigma) + spatial_av(noise_2sigma))/sqrt(N/Z)

    Inputs
    ------
    f : h5py.File
        GRACE h5 file object

    Outputs
    -------
    uncertainty_df : pd.DataFrame
    """
    import numpy as np
    import pandas as pd

    uncertainty_cols = list(f["uncertainty"])
    uncertainty_df = pd.DataFrame()
    for key in uncertainty_cols[:-1]:
        uncertainty_df[key] = np.array(f["uncertainty"][key]).T.squeeze()
    noise_df = pd.DataFrame(f["uncertainty"]["noise_2sigma"])
    uncertainty_df = pd.concat([uncertainty_df, noise_df], axis=1)
    return uncertainty_df


def load_GRACE_mascons(f: h5py.File) -> GeoDataFrame:
    """
    Load GRACE mascons.

    Inputs
    ------
    f : h5py.File
        GRACE h5 file object

    Outputs
    -------
    mascon_gdf : gpd.GeoDataFrame
    """
    import pandas as pd
    from geopandas import GeoDataFrame
    from shapely.geometry import Polygon

    mascon_cols = list(f["mascon"])  # grab dataset names in mascon group
    mascon_cols.remove("location_legend")  # remove unused dataset name
    mascon_df = pd.DataFrame()  # create empty pd.DataFrame
    for key in mascon_cols:  # fill df
        mascon_df[key] = np.array(f["mascon"][key]).T.squeeze()
    # Convert longitude from [0 to 360] to (-180 to 180]
    mascon_df.loc[mascon_df["lon_center"] > 180, "lon_center"] = (
        mascon_df.loc[mascon_df["lon_center"] > 180, "lon_center"] - 360
    )
    # Convert from lat/lon coordinates to polygons then to GeoDataFrame
    coord_corners = pd.DataFrame(columns=["NE", "SE", "SW", "NW", "close"])
    min_lon = mascon_df["lon_center"] - mascon_df["lon_span"] / 2
    min_lat = mascon_df["lat_center"] - mascon_df["lat_span"] / 2
    max_lon = mascon_df["lon_center"] + mascon_df["lon_span"] / 2
    max_lat = mascon_df["lat_center"] + mascon_df["lat_span"] / 2
    coord_corners["NE"] = list(zip(max_lon, max_lat, strict=True))
    coord_corners["SE"] = list(zip(max_lon, min_lat, strict=True))
    coord_corners["SW"] = list(zip(min_lon, min_lat, strict=True))
    coord_corners["NW"] = list(zip(min_lon, max_lat, strict=True))
    coord_corners["close"] = coord_corners["NE"]
    coord_geom = coord_corners.apply(Polygon, axis=1)
    mascon_gdf = GeoDataFrame(
        data=mascon_df, geometry=coord_geom.values, crs="EPSG:4326"
    )
    return mascon_gdf


def load_GRACE_dates(f: h5py.File) -> pd.DataFrame:
    """
    Load and format GRACE dates.

    Inputs
    ------
    f : h5py.File
        GRACE h5 file object

    Outputs
    -------
    date_df : pd.DataFrame
    """
    import numpy as np
    import pandas as pd

    start_date = pd.Timestamp("2001-12-31")
    time_cols = list(f["time"])  # grab dataset names in time group
    time_cols.remove("list_ref_days_solution")  # remove unused dataset name
    time_df = pd.DataFrame()  # create empty pd.DataFrame for reference dates
    for key in time_cols[2:-1]:  # fill df with days since reference day
        time_df[key] = np.array(f["time"][key]).T.squeeze()
    date_df = time_df.apply(
        lambda x: pd.to_datetime(x, unit="D", origin=start_date), axis=1
    )
    date_df.columns = ["date_first", "date_last", "date_middle"]
    date_df[["year_middle", "doy_middle", "frac_year_middle"]] = pd.DataFrame(
        f["time"]["yyyy_doy_yrplot_middle"]
    ).T
    return date_df


def load_GRACE_cmwe(f: h5py.File) -> pd.DataFrame:
    """Load GRACE solutions from h5 file."""
    import pandas as pd

    cmwe = pd.DataFrame(f["solution"]["cmwe"])
    return cmwe


def load_GRACE(
    grace_filename: str = "gsfc.glb_.200204_202211_rl06v2.0_obp-ice6gd.h5",
    grace_filepath: str = "/global/scratch/users/ann_scheliga/",
    land_subset: bool = True,
    uncertainty: bool = False,
    formatting: bool = True,
) -> dict:
    """

    Long description
    ----------------
    This is a placeholder description.

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
    grace_dict : dict
    """
    import h5py

    f = h5py.File(grace_filepath + grace_filename, "r")
    grace_dict = {}

    mascon_gdf = load_GRACE_mascons(f)
    date_df = load_GRACE_dates(f)
    cmwe = load_GRACE_cmwe(f)

    if land_subset:
        land_bool = mascon_gdf["location"] == 80
        mascon_gdf = mascon_gdf.loc[land_bool, :]
        cmwe = cmwe.loc[land_bool, :]
    if formatting:
        mascon_gdf.index = mascon_gdf["labels"].astype(int)
        cmwe.columns = date_df["date_middle"]
        cmwe.index = mascon_gdf["labels"].astype(int)

    grace_dict["mascon"] = mascon_gdf
    grace_dict["date"] = date_df
    grace_dict["cmwe"] = cmwe

    if uncertainty:
        uncertainty_df = load_GRACE_uncertainty(f)
        if land_subset:
            uncertainty_df = uncertainty_df.loc[land_bool, :]
        if formatting:
            uncertainty_df.index = mascon_gdf["labels"].astype(int)
        grace_dict["uncertainty"] = uncertainty_df

    return grace_dict


def load_GRanD(
    GRanD_filename: str = "GRanD_reservoirs_v1_3.shp",
    GRanD_filepath: str = "/global/scratch/users/ann_scheliga/dam_datasets/",
) -> GeoDataFrame:
    """
    Load the Global Reservoir and Dam Database (GRanD) shapefile.

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


def load_IMERG(
    imerg_filename: str = "IMERG_allmonths_201801_202304_xr.nc",
    imerg_filepath: str = "/global/scratch/users/ann_scheliga/IMERG_monthly_data/",
) -> DataArray:
    """
    Load global IMERG precipitation as an xarray.DataArray.

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

    imerg_raw = xr.open_dataarray(
        imerg_filepath + "IMERG_allmonths_201801_202304_xr.nc"
    )
    return imerg_raw


def load_DEM_full_as_nparray(
    dem_filepath: str = "/global/scratch/users/cgerlein/"
    "fc_ecohydrology_scratch/CYGNSS/Data/",
    dem_filename: str = "CYGNSS_0_01_deg_Map_DEM_Mask.npy",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read the global 0.01deg DEM file into memory.

    Inputs
    ------
    dem_filepath : str
    dem_filename : str

    Outputs
    ------
    dem: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    """
    import numpy as np

    dem = np.load(dem_filepath + dem_filename)
    # latitude ranges from -45 to 45, with num = 9001
    # longitude ranges from -180 to 180 with num = 36001
    # thus the coordinates can be set up as follow:
    lat = np.linspace(-45, 45, dem.shape[0])
    lon = np.linspace(-180, 180, dem.shape[1])

    return dem, lat, lon


def load_DEM_full_as_rxrDA(
    dem_filepath: str = "/global/scratch/users/cgerlein/"
    "fc_ecohydrology_scratch/CYGNSS/Data/",
    dem_filename: str = "CYGNSS_0_01_deg_Map_DEM_Mask.npy",
    _crs: int = 4326,
) -> DataArray:
    """
    Read the global 0.01deg DEM file into memory.

    Inputs
    ------
    dem_filepath : str
    dem_filename : str
    _crs : int
        default : 4326 (WGS84 lat/lon)
        Not used if reading a geospatial file (ie .tif)

    Outputs
    ------
    dem_full_rxr : xr.DataArray
        has rioxarray spatial reference
    """
    import rioxarray as rxr

    from codebase.dataprocessing import convert_from_np_to_rxr

    try:
        dem_full_rxr = rxr.open_rasterio(dem_filepath + dem_filename)
    except Exception:
        dem, lat, lon = load_DEM_full_as_nparray(dem_filepath, dem_filename)
        dem_full_rxr = convert_from_np_to_rxr(dem, lat=lat, lon=lon, _crs=_crs)
    return dem_full_rxr


def load_DEM_subset_as_nparray(
    bbox_vals: pd.DataFrame,
    dem_filepath: str = "/global/scratch/users/cgerlein/"
    "fc_ecohydrology_scratch/CYGNSS/Data/",
    dem_filename: str = "CYGNSS_0_01_deg_Map_DEM_Mask.npy",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read and subset the global 0.01deg DEM file.

    Inputs
    ------
    bbox_vals : pd.DataFrame
        columns of 'minx', 'miny', 'maxx', 'maxy'
        default format from the gpd.bounds attribute
    dem_filepath : str
    dem_filename : str

    Outputs
    ------
    dem_subset: np.ndarray
    lat_subset: np.ndarray
    lon_subset: np.ndarray
    """
    dem_full, lat, lon = load_DEM_full_as_nparray(dem_filepath, dem_filename)
    lat_bool = (lat >= bbox_vals["miny"].values[0]) & (
        lat <= bbox_vals["maxy"].values[0]
    )
    lon_bool = (lon >= bbox_vals["minx"].values[0]) & (
        lon <= bbox_vals["maxx"].values[0]
    )
    dem_subset = dem_full[np.ix_(lat_bool, lon_bool)]
    del dem_full
    lat_subset = lat[lat_bool]
    lon_subset = lon[lon_bool]
    return dem_subset, lat_subset, lon_subset


def load_DEM_subset_as_rxrDA(
    bbox_vals: pd.DataFrame,
    dem_filepath: str = "/global/scratch/users/cgerlein/"
    "fc_ecohydrology_scratch/CYGNSS/Data/",
    dem_filename: str = "CYGNSS_0_01_deg_Map_DEM_Mask.npy",
    _crs: int = 4326,
) -> DataArray:
    """
    Read and subset the global 0.01deg DEM file.

    Inputs
    ------
    bbox_vals : pd.DataFrame
        columns of 'minx', 'miny', 'maxx', 'maxy'
        default format from the gpd.bounds attribute
    dem_filepath : str
    dem_filename : str
    _crs: int
        default : 4326 (WGS84 lat/lon)
        Not used if reading a geospatial file (ie .tif)


    Outputs
    ------
    clipped_rxr: xr.DataArray
    """
    dem_full_rxr = load_DEM_full_as_rxrDA(dem_filepath, dem_filename, _crs)
    clipped_rxr = dem_full_rxr.rio.clip_box(*bbox_vals.values[0])
    return clipped_rxr


def load_usbr_data(
    name: str, file_dir: str = "/global/scratch/users/ann_scheliga/dam_datasets/"
) -> pd.DataFrame:
    import os

    try:
        # test if `name` input is a filename
        pd.read_csv(file_dir + name)
    except Exception:
        # if `name` input is a reservoir name, find the filename in the folder
        # list comp looks for reservoir name and "usbr" in file strings
        list_of_filenames = os.listdir(file_dir)
        filename = next(
            fname
            for fname in list_of_filenames
            if name.lower() in fname.lower() and "usbr" in fname.lower()
        )
    else:
        filename = name
    finally:
        raw_data = pd.read_csv(file_dir + filename, header=7)
    return raw_data


def load_formatted_usbr_data(
    name: str,
    file_dir: str = "/global/scratch/users/ann_scheliga/dam_datasets/",
    monthly: bool = False,
    agg_kwargs: dict | None = None,
) -> pd.DataFrame:
    """
    Load USBR data by reservoir name.

    Inputs
    ------
    name: str
        name of RESERVOIR
    file_dir: str
        directory the file is in
    monthly: bool
        whether to aggregate the data to monthly timesteps
        uses `time_series_calcs.resample_to_monthly` function
    agg_kwargs: dict
        passes to `create_dict_for_agg_function`
        accepted kwargs are 'default_agg' and 'custom_aggs'
        method (mean, sum, etc.) to aggregate data if aggregated monthly

    Outputs
    -------
    data: pd.DataFrame
        formatted dataframe with all columns
    """
    from codebase.dataprocessing import (
        create_dict_for_agg_function,
        usbr_dataprocessing,
    )
    from codebase.time_series_calcs import resample_to_monthly

    if agg_kwargs is None:
        agg_kwargs = {}

    raw_data = load_usbr_data(name, file_dir)
    data = usbr_dataprocessing(raw_data)

    if monthly:
        agg_dict = create_dict_for_agg_function(data, **agg_kwargs)
        data = resample_to_monthly(data, agg_dict)

    return data


def load_grealm_heights(
    url_str: str = "https://ipad.fas.usda.gov/lakes/images/lake000462.10d.2.smooth.txt",
    monthly: bool = False,
) -> pd.DataFrame:
    """Load G-REALM data from .txt url.
    Function contains several hardcoded processing steps.
    """
    import pandas as pd

    from codebase.time_series_calcs import resample_to_monthly

    grealm_raw = pd.read_csv(url_str, header=12, sep=" ", skipinitialspace=True)
    # Default column names are the nan values of the column
    grealm_raw.rename(
        columns={
            "99999999": "Date",
            "99": "Hour",
            "99.1": "Minute",
            "999.99": "height_var_JASON2",
            "9999.99": "height_mMSL",
        },
        inplace=True,
    )
    grealm_nanfiltered = grealm_raw.loc[
        (grealm_raw["Date"] != 99999999) & (grealm_raw["height_mMSL"] != 9999.99)
    ]
    grealm_nanfiltered.index = pd.to_datetime(
        grealm_nanfiltered["Date"], format="%Y%m%d"
    )
    grealm_all_heights = grealm_nanfiltered[["height_var_JASON2", "height_mMSL"]]
    if monthly:
        grealm_all_heights = resample_to_monthly(grealm_all_heights)
    return grealm_all_heights


def load_hydroBASINS(
    filepath: str = "/global/scratch/users/ann_scheliga/"
    "aux_dam_datasets/HydroBASINS_NorAmer/",
    continent: str = "na",
    lvl: int = 3,
) -> GeoDataFrame:
    """Load hydroBASINS .shp file with variable level and continent.
    Default is North America level 3.
    """
    from geopandas import read_file

    filename = "hybas_" + continent + "_lev" + str(f"{lvl:02}") + "_v1c.shp"
    hydroBASINS = read_file(filepath + filename)
    return hydroBASINS


def load_GRDC_timeseries(
    filename: str,
    filepath: str = "/global/scratch/users/ann_scheliga/aux_dam_datasets/GRDC_CRB/",
    start_year: int = 2000,
    stop_year: int = 2025,
) -> pd.DataFrame:
    """Stop year is exclusive."""
    import pandas as pd

    from codebase.dataprocessing import grdc_timeseries_data_processing

    try:
        full_df = pd.read_table(
            filepath + filename, header=36, encoding="unicode_escape", delimiter=";"
        )
    except FileNotFoundError:
        output_df = pd.DataFrame()
    else:
        output_df = grdc_timeseries_data_processing(full_df, start_year, stop_year)
    return output_df


def load_GRDC_station_metadata(
    id_no: int | float,
    filepath: str = "/global/scratch/users/ann_scheliga/aux_dam_datasets/GRDC_CRB/",
    basin_str: str = "",
) -> GeoDataFrame:
    from geopandas import read_file

    if (len(basin_str) > 0) & ~basin_str.endswith("_"):
        basin_str = basin_str + "_"

    all_stations_meta = read_file(filepath + basin_str + "stationbasins.geojson")
    station_meta = all_stations_meta.loc[all_stations_meta["grdc_no"] == id_no]
    return station_meta


def load_GRDC_station_data_by_ID(
    id_no: float | int,
    filepath: str,
    timeseries_dict: dict | None = None,
    basin_str: str = "",
    filename_str: str = "_Q_Day.Cmd.txt",
) -> tuple[GeoDataFrame, pd.DataFrame]:
    """
    Load timeseries data and watershed shapefile.

    Inputs
    ------
    timeseries_dict : dict
        kwargs to pass onto `load_data.load_GRDC_timeseries`
        accepted kwargs: 'start_year' (inclusive) and 'stop_year' (exclusive)
    """
    if timeseries_dict is None:
        timeseries_dict = {}
    station_gpd = load_GRDC_station_metadata(id_no, filepath, basin_str)
    filename = str(int(id_no)) + filename_str
    station_timeseries = load_GRDC_timeseries(filename, filepath, **timeseries_dict)
    return station_gpd, station_timeseries


def load_daily_reservoir_CYGNSS_area(
    dam_name: str,
    filepath: str = "/global/scratch/users/ann_scheliga/CYGNSS_daily/",
    formatting: bool = True,
) -> pd.Series:
    import pandas as pd

    from codebase.dataprocessing import daily_CYGNSS_area_data_processing

    sw_area = pd.read_csv(filepath + dam_name + "_area.csv", index_col=0)
    if formatting:
        sw_area = daily_CYGNSS_area_data_processing(sw_area)
    return sw_area


def add_era5_met_data_by_shp(
    input_gpd: GeoDataFrame,
    filepath: str,
    col_suffix: str = "",
    start_year: int = -1,
    stop_year_ex: int = -1,
) -> pd.DataFrame:
    """
    Load areal aggregated temp and precip based on provided gpd.

    Long Description
    ----------------
    Uses `area_subsets.era5_shape_subset_and_concat_from_file_pattern`
    for each variable.
    Searches for all instances of a substring (ex: 'daily_tempK'),
    and concatenates all found files according to hard-coded concat_dict dimensions.
    For precipitation, aggregates using np.nansum
    For temperature, aggregates using np.nanmean.
    start and stop year not used, but included in case useful in future edits.

    Inputs
    ------
    input_gpd : geopandas.GeoDataFrame
        geometry to subset data
    col_suffix : str
        default = "" (empty)
        added to column names in final output dataframe
        useful when using this function multiple times
    start_year, stop_year_ex : int
        default = -1
        not used, passed in case future edits need the bounds
        for pattern parsing or filtering.
    """
    from codebase.area_subsets import era5_shape_subset_and_concat_from_file_pattern

    concat_dict = {"dim": "valid_time"}

    __, max_tempK_1dim = era5_shape_subset_and_concat_from_file_pattern(
        filepath=filepath,
        input_pattern=r"daily_max_tempK",
        subset_gpd=input_gpd,
        concat_dict=concat_dict,
        agg_function=np.nanmean,
    )
    if max_tempK_1dim is None:
        max_tempK_1dim = pd.Series()
    max_tempK_1dim.rename("max_tempK", inplace=True)

    __, min_tempK_1dim = era5_shape_subset_and_concat_from_file_pattern(
        filepath=filepath,
        input_pattern=r"daily_min_tempK",
        subset_gpd=input_gpd,
        concat_dict=concat_dict,
        agg_function=np.nanmean,
    )
    if min_tempK_1dim is None:
        min_tempK_1dim = pd.Series()
    min_tempK_1dim.rename("min_tempK", inplace=True)

    __, precip_1dim = era5_shape_subset_and_concat_from_file_pattern(
        filepath=filepath,
        input_pattern=r"daily_tot_precip",
        subset_gpd=input_gpd,
        concat_dict=concat_dict,
        agg_function=np.nansum,
    )
    if precip_1dim is None:
        precip_1dim = pd.Series()
    precip_1dim.rename("precipm", inplace=True)

    __, dewpoint_1dim = era5_shape_subset_and_concat_from_file_pattern(
        filepath=filepath,
        input_pattern=r"daily_dewpoint_K",
        subset_gpd=input_gpd,
        concat_dict=concat_dict,
        agg_function=np.nanmean,
    )
    if dewpoint_1dim is None:
        dewpoint_1dim = pd.Series()
    dewpoint_1dim.rename("dewpointK", inplace=True)

    met_df = pd.concat(
        [max_tempK_1dim, min_tempK_1dim, precip_1dim, dewpoint_1dim], axis=1
    ).add_suffix(col_suffix)
    return met_df


if __name__ == "__main__":
    test = load_GRACE()
    print(test)
