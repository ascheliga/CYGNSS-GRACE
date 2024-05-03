import numpy as np
import pandas as pd
from xarray import DataArray

from codebase.area_calcs import cygnss_convert_to_binary


def create_sampleDA_3D(rand_seed: int) -> DataArray:
    np.random.seed(rand_seed)
    data = np.random.randint(0, 5, size=(2, 2, 3))
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


def test_string_input_coords() -> None:
    da = create_sampleDA_3D(0).astype(str)
    da_binary = cygnss_convert_to_binary(da, true_val="1")

    np.testing.assert_allclose(da.coords["lat"].values, da_binary.coords["lat"].values)
    np.testing.assert_allclose(da.coords["lon"].values, da_binary.coords["lon"].values)
    np.testing.assert_equal(
        da.coords["time"].values, da_binary.coords["time"].values
    )


def test_string_input_values() -> None:
    da = create_sampleDA_3D(0).astype(str)
    da_binary = cygnss_convert_to_binary(da, true_val="1")
    ref_values_seed0 = np.array([[[0, 1, 0], [1, 1, 0]], [[0, 0, 0], [0, 0, 0]]])
    np.testing.assert_array_equal(da_binary.values, ref_values_seed0)


def test_numeric_input_coords() -> None:
    da = create_sampleDA_3D(0)
    da_binary = cygnss_convert_to_binary(da, true_val=1)

    np.testing.assert_allclose(da.coords["lat"].values, da_binary.coords["lat"].values)
    np.testing.assert_allclose(da.coords["lon"].values, da_binary.coords["lon"].values)
    np.testing.assert_equal(
        da.coords["time"].values, da_binary.coords["time"].values
    )


def test_numeric_input_values() -> None:
    da = create_sampleDA_3D(0)
    da_binary = cygnss_convert_to_binary(da, true_val=1)
    ref_values_seed0 = np.array([[[0, 1, 0], [1, 1, 0]], [[0, 0, 0], [0, 0, 0]]])
    np.testing.assert_array_equal(da_binary.values, ref_values_seed0)


def test_true_val_0() -> None:
    da = create_sampleDA_3D(0)
    da_binary = cygnss_convert_to_binary(da, true_val=0)
    np.testing.assert_equal(0, da_binary.values.sum())
