from numpy.testing import assert_equal

from codebase.utils import search_with_exception_handling


def test_single_match() -> None:
    input_pattern = "[0-9]{7}"
    input_string = "landsat_2020086_B03.tif"
    output_val = search_with_exception_handling(input_pattern, input_string)
    expected_val = "2020086"
    assert_equal(output_val, expected_val)


def test_single_match_re_input() -> None:
    input_pattern = r"[0-9]{7}"
    input_string = "landsat_2020086_B03.tif"
    output_val = search_with_exception_handling(input_pattern, input_string)
    expected_val = "2020086"
    assert_equal(output_val, expected_val)


def test_multi_match() -> None:
    input_pattern = "[0-9]{7}"
    input_string = "landsat_2020086_2020087_B03.tif"
    output_val = search_with_exception_handling(input_pattern, input_string)
    expected_val = "2020086"
    assert_equal(output_val, expected_val)


def test_no_match() -> None:
    input_pattern = "[0-9]{7}"
    input_string = "landsat_2020_B03.tif"
    output_val = search_with_exception_handling(input_pattern, input_string)
    expected_val = ""
    assert_equal(output_val, expected_val)


def test_group_handling() -> None:
    input_pattern = r"_([0-9]{7})_"
    input_string = "landsat_2020086_B03.tif"
    output_val = search_with_exception_handling(input_pattern, input_string)
    expected_val = "2020086"
    assert_equal(output_val, expected_val)
