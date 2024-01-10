from codebase.load_data import load_DEM_full
import numpy as np

def test_DEM_shape():
    expected_shape = (9001,36001)
    output_dem , output_lat ,output_lon = load_DEM_full()
    output_shape = output_dem.shape
    np.testing.assert_equal(output_shape, expected_shape)
    np.testing.assert_equal(len(output_lat),expected_shape[0])
    np.testing.assert_equal(len(output_lon),expected_shape[1])

def test_DEM_values():
    expected_max = 32767
    expected_min = -414.9602
    expected_mean = 24395.6754
    expected_stats = [expected_max , expected_min, expected_mean]
    output_dem , _ , _ = load_DEM_full()

    output_stats = [output_dem.max(), output_dem.min(), output_dem.mean()]

    np.testing.assert_allclose(output_stats,expected_stats,rtol = 1e-2)