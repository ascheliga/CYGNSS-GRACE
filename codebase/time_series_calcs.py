from __future__ import annotations

import calendar
from datetime import date
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats
from sklearn.metrics import r2_score
from xarray import DataArray


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Take the z-score normalization of input dataframe."""
    return (df - df.mean()) / df.std()


def toYearFraction(input_date: date) -> float:
    """
    Convert input_date-time objects to decimal year.

    Inputs
    ------
    input_date : date object

    Outputs
    -------
    decYear : float
    """
    import time
    from datetime import datetime as dt

    def sinceEpoch(input_date: date) -> float:  # returns seconds since epoch
        return time.mktime(input_date.timetuple())

    s = sinceEpoch

    year = input_date.year
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year + 1, month=1, day=1)

    yearElapsed = s(input_date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed / yearDuration

    decYear = input_date.year + fraction
    return decYear


def IMERG_timestep_to_pdTimestamp(input_xrcoord: DataArray) -> ArrayLike:
    """
    Convert xr array of IMERG timestep numbers to an array Pandas Timestamp objects.

    Timestep  = seconds since 1980 Jan 06 (UTC), per original HDF5 IMERG file units
    """
    import numpy as np
    import pandas as pd

    dates_precip = np.array(
        [
            pd.Timestamp("1980-01-06") + pd.DateOffset(seconds=x)
            for x in input_xrcoord.values
        ]
    )
    return dates_precip


def CYGNSS_timestep_to_pdTimestamp(input_xrcoord: DataArray) -> np.ndarray:
    """
    Convert xr array of CYGNSS timestep numbers to an array Pandas Timestamp objects.

    Long Description
    ----------------
    Timestep  = months since 2018 Aug 01 (UTC)

    Inputs
    ------
    input_xrcoord : xr.DataArray
        1-dimensional array of ints

    Outputs
    -------
    dates_fw : np.ndarray
        1-dimensional array of pd.Timestamp objects
    """
    import numpy as np
    import pandas as pd

    dates_fw = np.array(
        [pd.Timestamp("2018-08-01") + pd.DateOffset(months=x) for x in input_xrcoord]
    )
    return dates_fw


def linregress_wrap(x_input: ArrayLike, y_input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run linear regression on each pixel/mascon time series
    and return metrics of interest.

    Long description
    ----------------
    Perform column-wise linear regression using scipy.linregress.
    Creates dataframe to store lin_regress metrics.
    Metrics stored are 'slope', 'intercept', 'r_value', 'p_value', 'std_err'.

    Inputs
    ------
    x_input : array_like
        the x-component of the linear regression
        typically is a Pandas Series with decimal year
        must be same length as y_input_df
    y_input_df : Pandas DataFrame
        the y-component of the linear regression
        linregress functions needs an array_like input,
        the wrapper needs a df input to extract columns
        size: [n_timesteps x n_pixels]
        must be same length as x_input

    Outputs
    -------
    output_df : Pandas DataFrame
        size: [n_pixels x 5 attributes], n_pixels = columns in y_input_df
        column names in order: 'slope', 'intercept', 'r_value', 'p_value', 'std_err'

    """
    output_df = pd.DataFrame(
        columns=["slope", "intercept", "r_value", "p_value", "std_err"],
        index=y_input_df.columns,
    )
    for mascon in y_input_df.columns:
        output_df.loc[mascon] = stats.linregress(x_input, y_input_df[mascon])
    return output_df


def intersecting_timeframes(*series: pd.Series, buffer: int = 1) -> list:
    """
    Slice multiple time series down to a shared timespan.

    Long description
    ----------------
    Uses pandas.DateOffset to apply buffer
    Finds the start and stop input_date (with buffer) of each input series,
    then subsets to the maximum start and the minimum stop.
    First made for res_time_series.ipynb

    Inputs
    ------
    *series : variable number of pd.Series
        time series to subset
        must have Timestamp indices
    buffer : int
        default = 1
        number of months to extend each time series

    Outputs
    -------
    cropped_series_list : list
        all *series cropped to shared timeframe
        in the same order as the inputs
    """
    series_start_list = [np.nan] * len(series)
    series_stop_list = [np.nan] * len(series)
    for idx, ts in enumerate(series):
        series_start_list[idx] = ts.index.min() + pd.DateOffset(months=-buffer)
        series_stop_list[idx] = ts.index.max() + pd.DateOffset(months=buffer)

    combined_start = max(series_start_list)
    combined_stop = min(series_stop_list)

    cropped_series_list = [None] * len(series)
    for idx, ts in enumerate(series):
        cropped_series_list[idx] = ts.loc[
            (ts.index >= combined_start) & (ts.index <= combined_stop)
        ]
    return cropped_series_list


class TimeSeriesMetrics:
    """Calculate various metrics on a single time series input as a Pandas Series."""

    def __init__(
        self,
        pd_series: pd.Series,
        dataset_name: str,
        remove_seasonality: bool = False,
        zero_start: bool = True,
        start_month: int = 1,
    ):
        self.ts = pd_series
        self.ts_raw = pd_series  # maintain the input time series
        # allowed_datasets = ['FO' , # GRACE-FO data
        #                     'DA' , # data assimilation output
        #                     'OL']  # open-loop or model-only
        # assert dataset_name in allowed_datasets
        self.name = dataset_name

        self.detrend()
        if zero_start:
            self.ts_zero_start = self.zero_start()
        if remove_seasonality:
            self.remove_seasonality(start_month=start_month)

    def zero_start(self) -> pd.DataFrame:
        """
        Vertical shift of time series to have all time series start at zero.
        Subtracts the first row from the dataset.
        """
        _ts_zero_start = self.ts - self.ts.iloc[0]
        return _ts_zero_start

    def detrend(self) -> None:
        """
        Remove linear trend from time series.

        Inputs
        ------
        None.

        Outputs
        -------

        """

        def detrend_timeseries(
            df_actuals: pd.DataFrame,
        ) -> tuple[ArrayLike, pd.DataFrame]:
            """Caluclate and remove linear trend from time series."""
            x_values = df_actuals.index.values
            x_mask = ~pd.isnull(x_values)

            lintrend_metrics = linregress_wrap(
                x_values[x_mask].astype(float), df_actuals[x_mask]
            )
            m = lintrend_metrics["slope"].values
            b = lintrend_metrics["intercept"].values

            y_trend = np.array(
                [np.nan if pd.isnull(x) else float(m * x + b) for x in x_values]
            ).reshape(-1, 1)
            y_detrend = df_actuals.values - y_trend
            return y_detrend, lintrend_metrics

        _detrended_ts, _ts_linmetrics = detrend_timeseries(self.ts.to_frame())
        self.ts_detrend = pd.Series(
            np.squeeze(_detrended_ts.astype(float)), index=self.ts.index.fillna(0)
        )
        self.lintrend_metrics = _ts_linmetrics.iloc[0]

    def cross_corr(
        self,
        comparison_ts: TimeSeriesMetrics | ArrayLike,
        ax: plt.Axes,
        ts_type: str = "detrend",
        plot_on: bool = True,
    ) -> tuple[ArrayLike, ArrayLike]:
        """Plot cross-correlation time lag plots against comparison input."""
        if "detrend" in ts_type:
            if "TimeSeriesMetrics" in str(type(comparison_ts)):
                _y = comparison_ts.ts_detrend
            else:
                _y = comparison_ts
            _x = self.ts_detrend
        elif "season" in ts_type:
            if "TimeSeriesMetrics" in str(type(comparison_ts)):
                _y = comparison_ts.seasonality
            else:
                _y = comparison_ts
            _x = self.seasonality

        assert len(_x) == len(_y), "Mismatch in length of time series"
        x_mask = ~np.isnan(_x)
        _x = _x[x_mask]
        _y = _y[x_mask.values]

        try:
            lag_time, lag_corr, _, _ = ax.xcorr(_x, _y)
        except Exception:
            lag_time, lag_corr, _, _ = plt.xcorr(_x, _y)

        # Print results
        ytrue_name = getattr(self, "name", "Base (true) time series")
        ypred_name = getattr(comparison_ts, "name", "Comparison (pred) time series")
        print("---cross correlation----")
        print("Between", ytrue_name, "and", ypred_name)
        y_max = np.max(lag_corr)
        idx_y_max = np.argmax(lag_corr)
        print(
            "Max correlation of",
            np.round(y_max, 3),
            "occurs with time shift of ",
            lag_time[idx_y_max],
        )
        if plot_on:
            plt.show()
        return lag_time, lag_corr

    def coef_determination(
        self, comparison_ts: TimeSeriesMetrics | ArrayLike, **kwargs: dict[str, Any]
    ) -> float:
        """
        Calculate and print coefficient of determination
        against comparison time series.
        """
        y_true = self.ts_detrend
        if "TimeSeriesMetrics" in str(type(comparison_ts)):
            y_pred = comparison_ts.ts_detrend
        else:
            y_pred = comparison_ts

        # Create x_mask to deal with nan's
        if "x_mask" in kwargs:
            x_mask = kwargs.get("x_mask")
        elif y_true.isnull().values.any() or y_pred.isnull().values.any():
            x_mask = ~(np.isnan(y_true).values + np.isnan(y_pred).values)
        else:
            x_mask = np.ones(len(y_true), dtype=bool)

        # Calculate coefficient of determination
        coef_det = r2_score(y_true[x_mask], y_pred[x_mask])

        # Print results
        ytrue_name = getattr(self, "name", "Base (true) time series")
        ypred_name = getattr(comparison_ts, "name", "Comparison (pred) time series")
        print("---coef of determination----")
        print("Between", ytrue_name, "and", ypred_name)
        print(coef_det)
        return coef_det

    def remove_seasonality(
        self, reps: int = 12, overwrite: bool = False, start_month: int = 1
    ) -> None:
        """
        Use for time series that haven't had seasonality removed
        (typically non-TWS time series).

        Inputs
        ------
        self.ts = matrix to remove climatology on
            first dimension must be time
            must be 2-D matrix
            must not have missing time steps
        reps = the number of time steps in 1 cycle
            reps = 12 for monthly
            reps = 0 for no cycle
        Outputs
        -------
        output = matrix with climatology removed
            same size as input
        means = the climatology values removed from the input.
        """
        if overwrite:
            del self.seasonality
            print("Seasonality already calculated. Overwriting previous calculation.")
        try:
            _test_exist = self.seasonality
            print("Seasonality already calculated. Add overwrite=True to overwrite")
        except Exception:
            print("Calculating seasonality.")
            l_input = self.ts_detrend.shape[0]
            # time length
            # if len(self.ts_detrend.shape)>1:
            #     # number of independent tiles/iterations/components
            #     w_input = self.ts_detrend.shape[1]
            # else:
            #     self.ts_detrend = self.ts_detrend.reshape(-1,1)
            w_input = 1
            m_add = reps - (l_input % reps)
            n_cycles = (l_input + m_add) / reps
            filler = np.empty([m_add, w_input])
            filler[:] = np.nan
            in_div = np.concatenate(
                (self.ts_detrend.values.reshape(-1, 1), filler), axis=0
            )  # adds nans to end to make divisible
            in_mat = in_div.reshape(
                -1, reps, w_input
            )  # size[n_cycles (axis to be averaged), reps, w_input]
            m_mean = np.nanmean(in_mat, 0)  # mean of each month
            m_mean_mat = np.repeat(m_mean[np.newaxis, :, :], n_cycles, axis=0)
            out_mat = in_mat - m_mean_mat
            out_div = np.reshape(
                out_mat, (int(reps * n_cycles), w_input)
            )  # contains nans to make divisible
            output = out_div[:-m_add, :]
            mean = m_mean

            self.ts_detrend = pd.DataFrame(output, index=self.ts_detrend.index)[0]
            months_list = list(calendar.month_name[start_month:]) + list(
                calendar.month_name[1 : (start_month - 13)]
            )
            _seasonality = pd.DataFrame(mean, index=months_list)[0]
            self.seasonality: pd.DataFrame = _seasonality.reindex(
                calendar.month_name[1:13]
            )

    def plot_anomalies(
        self,
        ax: plt.Axes,
        norm: bool = True,
        x_mask: ArrayLike | None = None,
        **plot_kwargs: dict[str, Any],
    ) -> None:
        """Plot detrended time series."""
        y = self.ts_detrend
        if x_mask is None:
            x_mask = np.ones(len(y), dtype=bool)
        y.loc[~x_mask] = np.nan
        if norm:
            y = normalize(y)
        ax.plot(y, **plot_kwargs)

    def plot_seasonality(
        self,
        ax: plt.Axes,
        norm: bool = True,
        x_mask: ArrayLike | None = None,
        **plot_kwargs: dict[str, Any],
    ) -> None:
        """Plot seasonality attribute."""
        y = self.seasonality
        if x_mask is None:
            x_mask = np.ones(len(y), dtype=bool)
        y.loc[~x_mask] = np.nan
        if norm:
            y = normalize(y)
        ax.plot(y, **plot_kwargs)


def resample_to_monthly(data: pd.DataFrame) -> pd.DataFrame:
    from pandas import Timedelta

    data_M = data.resample("M", label="left", closed="left").mean()
    # resampling makes the index one day too early, corrects to first of the month
    data_M.index = data_M.index + Timedelta("1D")
    return data_M
