import numpy as np
import pandas as pd

from codebase.area_calcs import stat_check


def test_stat_check_single():
    np.random.seed(12)
    input_df = pd.DataFrame(
        data=np.random.uniform(size=(20, 2)), columns=["slope", "p_value"]
    )
    output_vals, output_bool = stat_check(input_df, condition="pos", pcut=0.1)
    expected_vals = pd.DataFrame(
        data=[[0.90071485, 0.03342143], [0.76456045, 0.0208098]],
        index=[3, 10],
        columns=["slope", "p_value"],
    )
    pd.testing.assert_frame_equal(output_vals, expected_vals)


# Does not seem to work. I do not know how to make the input dataframes succinctly.
# @pytest.mark.parametrize(
#     "input_array, expected_array",
#     [
#         (pd.DataFrame(data = np.random.uniform(size=(20,2)),columns = ['slope', 'p_value']),
#          pd.DataFrame(data = [[0.90071485 , 0.03342143], [0.76456045 , 0.0208098 ]],
#                                  index = [3,10],
#                                  columns =['slope', 'p_value'])),
#         (pd.DataFrame(), pd.DataFrame()),
#     ],
# )
# def test_stat_check_array(input_array, expected_array):
#     """Test that rescale works correctly for multiple cases."""
#     output_array = stat_check(input_df, condition = 'pos' ,pcut=0.1)
#     np.testing.assert_allclose(output_array, expected_array)
