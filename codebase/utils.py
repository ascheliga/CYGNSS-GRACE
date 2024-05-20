import numpy as np


def convert_to_num(single_value):
    if isinstance(single_value, float | int):
        return single_value
    try:
        float(single_value.replace(",", ""))
    except Exception:
        return np.nan
    else:
        return float(single_value.replace(",", ""))
