import numpy as np
import pandas as pd
from xarray import DataArray

from codebase.volume_pipeline import align_DEM_and_CYGNSS_coordinates


def create_sampleDA_2D(rand_seed) -> DataArray:
    np.random.seed(rand_seed)
    data = 12 * np.random.randn(2, 2)
    lon = np.array([[-99.83, -99.32], [-99.79, -99.23]]) + np.random.uniform() * 1e-8
    lat = np.array([[42.25, 42.21], [42.63, 42.59]]) + np.random.uniform() * 1e-8
    da = DataArray(
        data=data,
        dims=["x", "y"],
        coords={
            "lon": (["x", "y"], lon),
            "lat": (["x", "y"], lat),
        },
    )
    return da


def create_sampleDA_3D(rand_seed) -> DataArray:
    np.random.seed(rand_seed)
    data = 12 * np.random.randn(2, 2, 3)
    lon = np.array([[-99.83, -99.32], [-99.79, -99.23]]) + np.random.uniform() * 1e-8
    lat = np.array([[42.25, 42.21], [42.63, 42.59]]) + np.random.uniform() * 1e-8
    time = pd.date_range("2014-09-06", periods=3)
    reference_time = pd.Timestamp("2014-09-05")
    da = DataArray(
        data=data,
        dims=["x", "y", "time"],
        coords={
            "lon": (["x", "y"], lon),
            "lat": (["x", "y"], lat),
            "time": time,
            "reference_time": reference_time,
        },
    )
    return da


def test_close_3Darray() -> None:
    DA1 = create_sampleDA_3D(0)
    DA2 = create_sampleDA_3D(1)
    DA1_aligned, DA2_aligned = align_DEM_and_CYGNSS_coordinates(DA1, DA2)
    np.testing.assert_equal(
        DA1_aligned.coords["lat"].values, DA2_aligned.coords["lat"].values
    )
    np.testing.assert_equal(
        DA1_aligned.coords["lon"].values, DA2_aligned.coords["lon"].values
    )


def test_close_2Darray() -> None:
    DA1 = create_sampleDA_2D(0)
    DA2 = create_sampleDA_2D(1)
    DA1_aligned, DA2_aligned = align_DEM_and_CYGNSS_coordinates(DA1, DA2)
    np.testing.assert_equal(
        DA1_aligned.coords["lat"].values, DA2_aligned.coords["lat"].values
    )
    np.testing.assert_equal(
        DA1_aligned.coords["lon"].values, DA2_aligned.coords["lon"].values
    )


# def test_misaligned_3Darray() -> None:

# # def test_far_3Darray() -> None:
