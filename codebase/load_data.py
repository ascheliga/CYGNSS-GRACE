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
    dem_full_rxr = global_xrDS["Watermask"].rio.write_crs(4326)
    del global_xrDS
    dem_full_rxr.rio.set_spatial_dims("lon", "lat", inplace=True)
    clipped_rxr = dem_full_rxr.rio.clip_box(*bbox_vals)
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
    """
    import geopandas as gpd
    import h5py
    import numpy as np
    import pandas as pd
    from shapely.geometry import Polygon

    f = h5py.File(grace_filepath + grace_filename, "r")
    grace_dict = {}

    # MASCONS #
    mascon_cols = list(f["mascon"])  # grab dataset names in mascon group
    mascon_cols.remove("location_legend")  # remove unused dataset name
    mascon_df = pd.DataFrame()  # create empty pd.DataFrame
    for key in mascon_cols:  # fill df
        mascon_df[key] = np.array(f["mascon"][key]).T.squeeze()
    # Convert longitude from [0 to 360] to (-180 to 180]
    mascon_df.loc[mascon_df["lon_center"] > 180, "lon_center"] = (
        mascon_df.loc[mascon_df["lon_center"] > 180, "lon_center"] - 360
    )
    if land_subset:
        land_bool = mascon_df["location"] == 80
        mascon_df = mascon_df.loc[land_bool, :]
    if formatting:
        mascon_df.index = mascon_df["labels"].astype(int)
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
    mascon_gdf = gpd.GeoDataFrame(
        data=mascon_df, geometry=coord_geom.values, crs="EPSG:4326"
    )
    grace_dict["mascon"] = mascon_gdf

    # DATES #
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
    grace_dict["date"] = date_df

    # CMWE SOLUTIONS #
    cmwe = pd.DataFrame(f["solution"]["cmwe"])
    if land_subset:
        cmwe = cmwe.loc[land_bool, :]
    if formatting:
        cmwe.columns = date_df["date_middle"]
        cmwe.index = mascon_df["labels"].astype(int)
    # cmwe_gpd = gpd.GeoDataFrame(data=cmwe,geometry=coord_geom.values,crs="EPSG:4326")
    grace_dict["cmwe"] = cmwe

    # UNCERTAINTY #
    if uncertainty:
        uncertainty_cols = list(f["uncertainty"])
        uncertainty_df = pd.DataFrame()
        for key in uncertainty_cols[:-1]:
            uncertainty_df[key] = np.array(f["uncertainty"][key]).T.squeeze()
        noise_df = pd.DataFrame(f["uncertainty"]["noise_2sigma"])
        uncertainty_df = pd.concat([uncertainty_df, noise_df], axis=1)
        if land_subset:
            uncertainty_df = uncertainty_df.loc[land_bool, :]
        if formatting:
            mascon_df.index = mascon_df["labels"].astype(int)
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

    return dem , lat , lon

def load_DEM_full_as_rxrDA(
    dem_filepath: str = "/global/scratch/users/cgerlein/"
    "fc_ecohydrology_scratch/CYGNSS/Data/",
    dem_filename: str = "CYGNSS_0_01_deg_Map_DEM_Mask.npy",
    _crs :int = 4326
) -> DataArray:
    """
    Read the global 0.01deg DEM file into memory.

    Inputs
    ------
    dem_filepath : str
    dem_filename : str
    _crs : int
        default : 4326 (WGS84 lat/lon)

    Outputs
    ------
    dem_full_rxr : xr.DataArray
        has rioxarray spatial reference
    """
    import numpy as np
    import xarray as xr
    import rioxarray

    dem , lat , lon = load_DEM_full_as_nparray(dem_filepath, dem_filename)

    dem_full = xr.DataArray(data = dem , dims = ['lat','lon'], 
                       coords=dict(lat= (['lat'],lat),lon=(['lon'],lon)))
    dem_full_rxr = dem_full.rio.write_crs(_crs)
    dem_full_rxr.rio.set_spatial_dims("lon", "lat", inplace=True)
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
    dem_full = load_DEM_full_as_nparray(dem_filepath, dem_filename)
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
    _crs: int = 4326
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

    Outputs
    ------
    clipped_rxr: xr.DataArray
    """
    dem_full_rxr = load_DEM_full_as_rxrDA(dem_filepath, dem_filename, _crs)
    clipped_rxr = dem_full_rxr.rio.clip_box(*bbox_vals.values[0])
    return clipped_rxr

if __name__ == "__main__":
    test = load_IMERG()
    print(test)
