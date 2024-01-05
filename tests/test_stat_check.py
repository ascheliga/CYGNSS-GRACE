from codebase.area_calcs import stat_check
import pandas as pd
import numpy as np
import pytest

def test_stat_check_single():
    np.random.seed(12)
    input_df = pd.DataFrame(data = np.random.uniform(size=(20,2)),
                            columns = ['slope', 'p_value'])
    output_vals , output_bool = stat_check(input_df,condition = 'pos',pcut=0.1)
    expected_vals = pd.DataFrame(data = [[0.90071485 , 0.03342143], [0.76456045 , 0.0208098 ]],
                                 index = [3,10],
                                 columns =['slope', 'p_value'])
    pd.testing.assert_frame_equal(output_vals,expected_vals)

@pytest.mark.parametrize(
    "input_array, expected_array",
    [
        (np.array([1, 2, 3, 4, 5]), np.array([0, 0.25, 0.5, 0.75, 1])),
        (np.array([5, 4, 3, 2, 1]), np.array([1, 0.75, 0.5, 0.25, 0])),
    ],
)
def test_stat_check_array(input_array, expected_array):
    """Test that rescale works correctly for multiple cases."""
    output_array = rescale(input_array)
    np.testing.assert_allclose(output_array, expected_array)    