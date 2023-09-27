import pandas as pd
import numpy as np
from scipy import stats
import calendar
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def _object2float(*inputs):
    for each in inputs:
        s = each.select_dtypes(include=object).columns
        each[s] = each[s].astype(float)
    return inputs

def normalize(df):
    return (df-df.mean())/df.std()

def toYearFraction(date):
    """
    Convert date-time objects to deciml year

    Inputs
    ------
    date : date object

    Outputs
    -------
    decYear : float
    """
    from datetime import datetime as dt
    import time
    def sinceEpoch(date): # returns seconds since epoch
        return time.mktime(date.timetuple())
    s = sinceEpoch

    year = date.year
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year+1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed/yearDuration

    decYear = date.year + fraction
    return decYear

def linregress_wrap(x_input,y_input_df):
    """
    Run linear regression on each pixel/mascon time series and return metrics of interest.
    ----------
    
    Long description
    ----------------
    Perform column-wise linear regression using scipy.linregress. Creates dataframe to store lin_regress metrics.
    Metrics stored are 'slope', 'intercept', 'r_value', 'p_value', 'std_err'.
    
    Inputs
    ------
    x_input : array_like
        the x-component of the linear regression
        typically is a Pandas Series with decimal year
        must be same length as y_input_df
    y_input_df : Pandas DataFrame
        the y-component of the linear regression
        linregress functions needs an array_like input, the wrapper needs a df input to extract columns
        size: [n_timesteps x n_pixels]
        must be same length as x_input
        
    Outputs
    -------
    output_df : Pandas DataFrame
        size: [n_pixels x 5 attributes], n_pixels = columns in y_input_df
        column names in order: 'slope', 'intercept', 'r_value', 'p_value', 'std_err'
        
    """
    output_df = pd.DataFrame(columns=['slope', 'intercept', 'r_value', 'p_value', 'std_err'],index=y_input_df.columns)
    for mascon in y_input_df.columns:
        output_df.loc[mascon] = stats.linregress(x_input,y_input_df[mascon])
    return output_df    

def intersecting_timeframes(*series,buffer=1):
    """
    Slice multiple time series down to a shared timespan.

    Long description
    ----------------
    Uses pandas.DateOffset to apply buffer
    Finds the start and stop date (with buffer) of each input series,
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
    series_start_list = [None] *  len(series)
    series_stop_list = [None] *  len(series)
    for idx , ts in enumerate(series):
        series_start_list[idx] = ts.index.min() + pd.DateOffset(months=-buffer)
        series_stop_list[idx] = ts.index.max() + pd.DateOffset(months=buffer)

    combined_start = max(series_start_list)
    combined_stop = min(series_stop_list)

    cropped_series_list = [None] * len(series)
    for idx , ts in enumerate(series):
        cropped_series_list[idx] = ts.loc[(ts.index >= combined_start) & (ts.index <= combined_stop)]
    return cropped_series_list
class TimeSeriesMetrics:
    """
    Calculate various metrics on a single time series input as a Pandas Series
    """
    def __init__(self,pd_series,dataset_name,remove_seasonality=False):
        self.ts = pd_series
        self.ts_raw = pd_series # maintain the input time series
        # allowed_datasets = ['FO' , # GRACE-FO data
        #                     'DA' , # data assimilation output
        #                     'OL']  # open-loop or model-only
        # assert dataset_name in allowed_datasets
        self.name = dataset_name
        self.zero_start()
        self.detrend()
        if remove_seasonality:
            self.remove_seasonality()
    def zero_start(self):
        """
        Vertical shift of time series to have all time series start at zero.
        Subtracts the first row from the dataset.
        """
        self.ts = self.ts - self.ts.iloc[0]
    def detrend(self):
        def detrend_timeseries(df_actuals):
            x_values = df_actuals.index.values
            x_mask = ~pd.isnull(x_values)
            
            lintrend_metrics = linregress_wrap(x_values[x_mask].astype(float),df_actuals[x_mask])
            m = lintrend_metrics['slope'].values
            b = lintrend_metrics['intercept'].values
            
            y_trend = np.array([np.nan if pd.isnull(x) else float(m*x + b) for x in x_values]).reshape(-1,1)
            y_detrend = df_actuals.values - y_trend
            return y_detrend , lintrend_metrics
        _detrended_ts , _ts_linmetrics = detrend_timeseries(self.ts.to_frame())
        self.ts_detrend = pd.Series(np.squeeze(_detrended_ts.astype(float)),index = self.ts.index.fillna(0))
        self.lintrend_metrics = _ts_linmetrics.iloc[0]
    def cross_corr(self,comparison_ts):
        if 'TimeSeriesMetrics' in str(type(comparison_ts)):
            _y = comparison_ts.ts_detrend
        else:
            _y = comparison_ts
        assert len(self.ts_detrend) == len(_y) , 'Mismatch in length of time series'
        x_mask = ~np.isnan(self.ts_detrend)
        _x = self.ts_detrend[x_mask]
        _y = comparison_ts[x_mask.values]
        print(plt.xcorr(_x,_y))
    def coef_determination(self,comparison_ts,**kwargs):
        y_true = self.ts_detrend
        if 'TimeSeriesMetrics' in str(type(comparison_ts)):
            y_pred = comparison_ts.ts_detrend
        else:
            y_pred = comparison_ts

        # Create x_mask to deal with nan's
        if 'x_mask' in kwargs:
            x_mask = kwargs.get('x_mask')
        elif y_true.isnull().values.any() or y_pred.isnull().values.any():
            x_mask = ~(np.isnan(y_true).values + np.isnan(y_pred).values)
        else:
            x_mask = np.ones(len(y_true), dtype=bool)

        # Calculate coefficient of determination
        coef_det = r2_score(y_true[x_mask],y_pred[x_mask])

        # Print results
        ytrue_name = getattr(self, 'name', 'Base (true) time series')
        ypred_name = getattr(comparison_ts, 'name', 'Comparison (pred) time series')
        print('---coef of determination----')
        print('Between',ytrue_name,'and',ypred_name)
        print(coef_det)
        return coef_det
    def remove_seasonality(self,reps=12):
        """
        Use for time series that haven't had seasonality removed (typically non-TWS time series)
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
        means = the climatology values removed from the input
        """
        l_input = self.ts_detrend.shape[0]; # time length
        # if len(self.ts_detrend.shape)>1:
        #     w_input = self.ts_detrend.shape[1] # number of independent tiles/iterations/components
        # else:
        #     self.ts_detrend = self.ts_detrend.reshape(-1,1)
        w_input = 1  
        m_add = reps - (l_input % reps)
        n_cycles = (l_input + m_add)/reps
        filler = np.empty([m_add,w_input])
        filler[:]=np.nan
        in_div = np.concatenate((self.ts_detrend.values.reshape(-1,1),filler),axis=0)      # adds nans to end to make divisible
        in_mat = in_div.reshape(-1,reps,w_input)                 # size[n_cycles (axis to be averaged), reps, w_input]
        m_mean = np.nanmean(in_mat,0)                            # mean of each month
        m_mean_mat = np.repeat(m_mean[np.newaxis,:,:],n_cycles,axis=0)   
        out_mat = in_mat - m_mean_mat
        out_div = np.reshape(out_mat,(int(reps*n_cycles),w_input))    # contains nans to make divisible
        output = out_div[:-m_add,:]
        mean= m_mean

        self.ts_detrend = pd.DataFrame(output,index=self.ts_detrend.index)[0]
        months_list = calendar.month_name[1:13]
        self.seasonality = pd.DataFrame(mean,index=months_list)[0]
    def plot_anomalies(self,ax,norm=True,x_mask=None,**plot_kwargs):
        y = self.ts_detrend
        if x_mask is None:
            x_mask = np.ones(len(y), dtype=bool)
        y.loc[~x_mask] = np.nan
        if norm:
            y = normalize(y)
        # y = y.zero_start()
        ax.plot(y,**plot_kwargs)