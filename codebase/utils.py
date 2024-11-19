from re import Pattern
from typing import Any

import numpy as np


def convert_to_num(single_value: Any) -> float | int:
    if isinstance(single_value, float | int):
        return single_value
    try:
        float(single_value.replace(",", ""))
    except Exception:
        return np.nan
    else:
        return float(single_value.replace(",", ""))


def _object2float(*inputs: Any) -> Any:
    for each in inputs:
        s = each.select_dtypes(include=object).columns
        each[s] = each[s].astype(float)
    return inputs


def convert_from_m_to_ft(value_m: float | int) -> float:
    """Convert from meters to feet."""
    value_ft = value_m * 3.281
    return value_ft


def convert_from_ft_to_m(value_ft: float | int) -> float:
    """Convert from feet to meters."""
    value_m = value_ft / 3.281
    return value_m


def convert_from_af_to_m3(value_af: float | int) -> float:
    """Convert from acre-feet to cubic meters."""
    value_m3 = value_af * 1233.48
    return value_m3


def convert_from_m3_to_af(value_m3: float | int) -> float:
    """Convert from cubic meters to acre-feet."""
    value_af = value_m3 / 1233.48
    return value_af


def convert_from_cfs_to_m3s(value_cfs: float | int) -> float:
    """Convert from cubic feet per second to cubis meters per second."""
    value_m3s = value_cfs / 35.315
    return value_m3s


def convert_from_m3s_to_cfs(value_m3s: float | int) -> float:
    """Convert from cubic meters per second to cubis feet per second."""
    value_cfs = value_m3s * 35.315
    return value_cfs


def convert_from_ac_to_m2(value_ac: float | int) -> float:
    """Convert from acres to square meters."""
    value_m2 = value_ac * 4046.856
    return value_m2


def convert_from_m2_to_ac(value_m2: float | int) -> float:
    """Convert from acres to square meters."""
    value_ac = value_m2 / 4046.856
    return value_ac


def search_with_exception_handling(r_pattern: str | Pattern, item: str) -> str:
    """Return regex search as a string, else return a blank string."""
    import re

    found = re.search(r_pattern, item)

    if found:
        try:
            output_str = found.group(1)
        except Exception:
            output_str = found.group(0)
    else:
        output_str = ""
    return output_str


def search_list_unique_values(regex_str: Pattern, input_list: list) -> list:
    """
    Find regex substrings in list.

    Long Description
    ----------------
    Use regex search to find the first occurrence of the pattern in each list item.
    Return the set() of all matches as a list.

    Inputs
    ------
    regex_str : re.Pattern
        the pattern to search for
        must compile beforehand (`re.compile()`) to a re.Pattern
    input_list : list
        list to search through

    Outputs
    -------
    unique_matches : list
        list of strings that match the pattern
    """
    all_matches = [
        search_with_exception_handling(regex_str, list_item) for list_item in input_list
    ]
    unique_matches = list(set(all_matches))
    return unique_matches
