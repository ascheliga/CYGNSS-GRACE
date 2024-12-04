import numpy as np

from codebase.load_data import load_hydroBASINS


def test_lvl1_NorAm() -> None:
    expected_shape = (1, 14)
    expected_PFAFID = 7
    output_gdf = load_hydroBASINS(lvl=1, continent="na")
    output_shape = output_gdf.shape
    output_PFAFID = output_gdf.loc[0, "PFAF_ID"]
    np.testing.assert_equal(output_shape, expected_shape)
    np.testing.assert_equal(output_PFAFID, expected_PFAFID)


def test_lvl4_LCBR() -> None:
    expected_shape = (185, 14)
    expected_PFAFID = 7723
    output_gdf = load_hydroBASINS(lvl=4, continent="na")
    output_shape = output_gdf.shape
    output_PFAFID = output_gdf.loc[11, "PFAF_ID"]
    np.testing.assert_equal(output_shape, expected_shape)
    np.testing.assert_equal(output_PFAFID, expected_PFAFID)


def test_default_values() -> None:
    expected_shape = (29, 14)
    expected_PFAFID = 771
    output_gdf = load_hydroBASINS()
    output_shape = output_gdf.shape
    output_PFAFID = output_gdf.loc[0, "PFAF_ID"]
    np.testing.assert_equal(output_shape, expected_shape)
    np.testing.assert_equal(output_PFAFID, expected_PFAFID)
