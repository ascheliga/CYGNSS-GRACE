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