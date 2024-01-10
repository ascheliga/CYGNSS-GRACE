import numpy as np
import pandas as pd

from codebase.load_data import load_DEM_subset


def test_DEM_shape():
    test_bbox = pd.DataFrame(
        data=np.array([10, 15, 30, 25]).reshape(1, -1),
        columns=["minx", "miny", "maxx", "maxy"],
    )
    expected_shape = (1001, 2001)
    output_dem, output_lat, output_lon = load_DEM_subset(test_bbox)
    output_shape = output_dem.shape
    np.testing.assert_equal(output_shape, expected_shape)
    np.testing.assert_equal(len(output_lat), expected_shape[0])
    np.testing.assert_equal(len(output_lon), expected_shape[1])


def test_DEM_values():
    test_bbox = pd.DataFrame(
        data=np.array([10, 15, 30, 25]).reshape(1, -1),
        columns=["minx", "miny", "maxx", "maxy"],
    )
    expected_max = 3304.5072
    expected_min = 151.6720
    expected_mean = 566.2700
    expected_stats = [expected_max, expected_min, expected_mean]
    output_dem, _, _ = load_DEM_subset(test_bbox)

    output_stats = [output_dem.max(), output_dem.min(), output_dem.mean()]

    np.testing.assert_allclose(output_stats, expected_stats, rtol=1e-2)
