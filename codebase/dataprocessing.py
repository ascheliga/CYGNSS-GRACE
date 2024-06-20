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
