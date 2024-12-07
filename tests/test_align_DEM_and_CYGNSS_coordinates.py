import numpy as np
import pandas as pd

from codebase import testing
from codebase.volume_pipeline import align_DEM_and_CYGNSS_coordinates


def test_close_3Darray() -> None:
    DA1 = testing.create_sampleDA_3D(0)
    DA2 = testing.create_sampleDA_3D(1)
    DA1_aligned, DA2_aligned = align_DEM_and_CYGNSS_coordinates(DA1, DA2)
    np.testing.assert_equal(
        DA1_aligned.coords["lat"].values, DA2_aligned.coords["lat"].values
    )
    np.testing.assert_equal(
        DA1_aligned.coords["lon"].values, DA2_aligned.coords["lon"].values
    )


def test_close_2Darray() -> None:
    DA1 = testing.create_sampleDA_2D(0)
    DA2 = testing.create_sampleDA_2D(1)
    DA1_aligned, DA2_aligned = align_DEM_and_CYGNSS_coordinates(DA1, DA2)
    np.testing.assert_equal(
        DA1_aligned.coords["lat"].values, DA2_aligned.coords["lat"].values
    )
    np.testing.assert_equal(
        DA1_aligned.coords["lon"].values, DA2_aligned.coords["lon"].values
    )


def test_misaligned_3Darray() -> None:
    DA1 = testing.create_sampleDA_3D(0)
    DA2 = testing.create_sampleDA_3D(1)
    DA2_shift = DA2.assign_coords(
        lat=(["x", "y"], np.array([[2.25, 2.21], [2.63, 2.59]]))
    )
    DA1_aligned, DA2_aligned = align_DEM_and_CYGNSS_coordinates(DA1, DA2_shift)
    np.testing.assert_allclose(
        DA1_aligned.coords["lat"].values, DA2_aligned.coords["lat"].values + 40
    )


def test_timeshift_3Darray() -> None:
    DA1 = testing.create_sampleDA_3D(0)
    DA2 = testing.create_sampleDA_3D(1)
    DA2_shift = DA2.assign_coords(time=pd.date_range("2014-09-07", periods=3))
    DA1_aligned, DA2_aligned = align_DEM_and_CYGNSS_coordinates(DA1, DA2_shift)
    np.testing.assert_equal(
        DA1_aligned.coords["lat"].values, DA2_aligned.coords["lat"].values
    )
    np.testing.assert_equal(
        DA1_aligned.coords["lon"].values, DA2_aligned.coords["lon"].values
    )
