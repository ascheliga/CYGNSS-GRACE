from typing import Any

import numpy as np


def convert_to_num(single_value: Any) -> float|int:
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
