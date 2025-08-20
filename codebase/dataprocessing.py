import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


def usbr_dataprocessing(df: pd.DataFrame) -> pd.DataFrame:
    from codebase.utils import convert_to_num

    df = df.drop(columns="Location")
    df.dropna(axis=0, how="any", inplace=True)
    df = df[df["Timestep"] != "Timestep"]  # remove repeat header rows
    df["Variable"] = df["Parameter"] + " [" + df["Units"] + "]"
    df["Result"] = df["Result"].apply(convert_to_num)
    df_pivot = df.pivot(columns="Variable", index="Datetime (UTC)", values="Result")
    df_pivot.index = pd.to_datetime(df_pivot.index)
    return df_pivot


def convert_from_np_to_rxr(
    data_np: np.ndarray, lat: np.ndarray, lon: np.ndarray, _crs: int
) -> xr.DataArray:
    """Could be rewritten to take more general dims, not just lat/lon."""
    data_full = xr.DataArray(
        data=data_np,
        dims=["lat", "lon"],
        coords={"lat": (["lat"], lat), "lon": (["lon"], lon)},
    )
    data_full_rxr = data_full.rio.write_crs(_crs)
    data_full_rxr.rio.set_spatial_dims("lon", "lat", inplace=True)
    return data_full_rxr


def create_dict_for_agg_function(
    df_columns: pd.Index | list | pd.DataFrame,
    default_agg: str | Callable = "mean",
    custom_aggs: dict | None = None,
) -> dict:
    """
    Create a full dictionary of column names and agg functions.

    Long description
    ----------------
    First, create a dictionary with all column names and the default agg_function.
    Second, replace the agg_funciton for certain columns.
    This function is useful for a wide DataFrame,
    so you only have to name the columns that need to differ.

    Inputs
    ------
    df_columns: pd.Index | list | pd.DataFrame
        the column names of the DataFrame
        if a dataFrame is provided, will extract the column names
    default_agg: str | Callable
        the aggregation method applied for unspecified columns
    custom_aggs: dict
        a dictionary of {'column_name: 'agg_function'} to change from the default value.

    Outputs
    -------
    agg_dict: dict
        dictionary that contains all columns
    """
    if isinstance(df_columns, pd.DataFrame):
        col_names = df_columns.columns
    else:
        col_names = df_columns

    if custom_aggs is None:
        custom_aggs = {}

    default_aggs = [default_agg] * len(col_names)
    agg_dict = dict(zip(col_names, default_aggs, strict=True))
    for key, value in custom_aggs.items():
        agg_dict[key] = value
    return agg_dict


def grdc_timeseries_data_processing(
    df: pd.DataFrame, start_year: int, stop_year: int
) -> pd.DataFrame:
    """stop_year is exclusive."""
    from pandas import to_datetime

    df.rename(columns={"YYYY-MM-DD": "Date", " Value": "Q m3s"}, inplace=True)
    df["Date"] = to_datetime(df["Date"])
    df.drop(columns="hh:mm", inplace=True)
    df.set_index("Date", inplace=True)
    df = df[
        (df.index >= to_datetime(str(start_year) + "-01-01"))
        & (df.index < to_datetime(str(stop_year) + "-01-01"))
    ]
    return df


def daily_CYGNSS_area_data_processing(
    sw_df: pd.DataFrame, start_year: int = 2000, stop_year: int = 2025
) -> pd.Series:
    from pandas import to_datetime

    sw_df.index = to_datetime(sw_df.index)
    sw_area = sw_df.loc[
        (sw_df.index >= to_datetime(str(start_year) + "-01-01"))
        & (sw_df.index < to_datetime(str(stop_year) + "-01-01")),
        "Area m2",
    ]
    sw_area = (sw_area / 10**6).rename("Area km2", inplace=True)
    return sw_area


def write_topo_features(
    stations_meta: Path | str | pd.DataFrame, output_path: Path | str
) -> pd.DataFrame:
    from geopandas import read_file

    if isinstance(stations_meta, Path | str):
        stations_meta = read_file(stations_meta)
    topo_df = stations_meta[["grdc_no", "lat_pp", "long_pp", "area_calc"]]
    topo_df["grdc_no"] = topo_df["grdc_no"].astype(int).copy()
    topo_df.to_csv(
        output_path,
        mode="a",
        header=not os.path.exists(output_path),
        sep=";",
        index=False,
    )
    print("Saved to", output_path)
    return topo_df


def sort_csv_from_file(
    output_path: Path | str,
    col_name: str,
    read_kwargs: dict[str, Any] | None = None,
    write_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if write_kwargs is None:
        write_kwargs = {}
    if read_kwargs is None:
        read_kwargs = {}
    full_df = pd.read_csv(output_path, **read_kwargs)
    sorted_df = full_df.sort_values(by=col_name)
    sorted_df.to_csv(output_path, **write_kwargs)
    return sorted_df
