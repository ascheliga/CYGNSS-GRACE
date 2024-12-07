import numpy as np
import pandas as pd
from xarray import DataArray

from codebase.area_calcs import CYGNSS_001_areal_aggregation
from codebase import testing

def test_sum_3Darray() -> None:
    DA = testing.create_sampleDA_3D(0)
    output_val = 
    DA1_aligned, DA2_aligned = CYGNSS_001_areal_aggregation(DA)
    np.testing.assert_equal(
        DA1_aligned.coords["lat"].values, DA2_aligned.coords["lat"].values
    )
    np.testing.assert_equal(
        DA1_aligned.coords["lon"].values, DA2_aligned.coords["lon"].values
    )
