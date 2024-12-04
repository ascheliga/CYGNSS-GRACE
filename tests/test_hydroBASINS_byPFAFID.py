import numpy as np
from geopandas import GeoDataFrame

from codebase.area_subsets import hydroBASINS_by_PFAFID
from codebase.load_data import load_hydroBASINS


def test_check_int_exact_match() -> None:
    expected_shape = (1, 14)
    expected_PFAFID = 7
    input_gdf = load_hydroBASINS(continent="na", lvl=1)
    output_gdf = hydroBASINS_by_PFAFID(7, input_gdf)
    output_shape = output_gdf.shape
    output_PFAFID = output_gdf.loc[0, "PFAF_ID"]
    np.testing.assert_equal(output_shape, expected_shape)
    np.testing.assert_equal(output_PFAFID, expected_PFAFID)
    assert isinstance(output_gdf, GeoDataFrame)


def test_check_int_multi_match() -> None:
    expected_shape = (8, 14)
    expected_PFAFID_mean = 74.5
    input_gdf = load_hydroBASINS(continent="na", lvl=2)
    output_gdf = hydroBASINS_by_PFAFID(7, input_gdf)
    output_shape = output_gdf.shape
    output_PFAFID_mean = output_gdf["PFAF_ID"].mean()
    np.testing.assert_equal(output_shape, expected_shape)
    np.testing.assert_approx_equal(output_PFAFID_mean, expected_PFAFID_mean)
    assert isinstance(output_gdf, GeoDataFrame)


def test_check_str_exact_match() -> None:
    expected_shape = (1, 14)
    expected_PFAFID = 7
    input_gdf = load_hydroBASINS(continent="na", lvl=1)
    output_gdf = hydroBASINS_by_PFAFID("7", input_gdf)
    output_shape = output_gdf.shape
    output_PFAFID = output_gdf.loc[0, "PFAF_ID"]
    np.testing.assert_equal(output_shape, expected_shape)
    np.testing.assert_equal(output_PFAFID, expected_PFAFID)
    assert isinstance(output_gdf, GeoDataFrame)


def test_check_str_multi_match() -> None:
    expected_shape = (8, 14)
    expected_PFAFID_mean = 74.5
    input_gdf = load_hydroBASINS(continent="na", lvl=2)
    output_gdf = hydroBASINS_by_PFAFID("7", input_gdf)
    output_shape = output_gdf.shape
    output_PFAFID_mean = output_gdf["PFAF_ID"].mean()
    np.testing.assert_equal(output_shape, expected_shape)
    np.testing.assert_approx_equal(output_PFAFID_mean, expected_PFAFID_mean)
    assert isinstance(output_gdf, GeoDataFrame)
