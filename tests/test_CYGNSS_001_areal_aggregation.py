import numpy as np

from codebase import testing
from codebase.area_calcs import CYGNSS_001_areal_aggregation


def test_sum_3Darray() -> None:
    DA = testing.create_sampleDA_3D(0)
    CYGNSS_001_areal_aggregation(DA)
    np.testing.assert_allclose(output_val, expected_val)
    np.testing.assert_equal(
        DA1_aligned.coords["lon"].values, DA2_aligned.coords["lon"].values
    )
