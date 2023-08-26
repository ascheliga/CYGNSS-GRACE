# In[1]:


## Define directories

helpdir = '/global/home/users/ann_scheliga/time_series_metrics_pipeline/'


# In[2]:


# Import packages
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd

# In[3]:

def stat_check(input_df, condition, pcut):
    """
    Mask dataframe by slope sign (positive or negative) and p-value cutoff.

    Long Description
    ----------------

    Inputs
    ------
    input_df : Pandas DataFrame
        must have 'slope' and 'p-value' columns
    condition : str
        must have 'pos' or 'neg' for positive or negative slope
    pcut : float
        p-value cutoff to determine statistically-significant slopes

    Outputs
    -------
    ouput_df : Pandas DataFrame
        input_df sliced to only rows that meet the input condition
    bool_vec : Pandas Series
        boolean series with True for dataframe rows that meet the input condition
    """
    if   'neg' in condition.lower():
        bool_vec = (input_df['slope']<0) & (input_df['p_value']<pcut)
    elif 'pos' in condition.lower():
        bool_vec = (input_df['slope']>0) & (input_df['p_value']<pcut)
    else:
        bool_vec = []
        print('Invalid condition given')
    output_df = input_df[bool_vec]
    return output_df, bool_vec


# In[4]:

def pos_neg_area_calc(input_df,pcut):
    """
    Provide the total area (km^2) that has a significant positive trend and a significant negative trend.

    Long Description
    ----------------

    Inputs
    ------
    input_df : Pandas DataFrame
        dataframe to perform area calculations on
        must have 'slope', 'p-value', and 'area' columns
    p_cut : float
        p-value cutoff to determine statistically-significant slopes

    Outputs
    -------
    pos_area_km2, neg_area_km2 : float
        area in km^2 that has statistically positive and negative (respectively) slope
    """
    neg_df , _ = stat_check(input_df,'neg',pcut)
    pos_df , _ = stat_check(input_df,'pos',pcut)
    
    areacol_list = [input_df.columns.get_loc(col) for col in input_df.columns if 'area' in col]
    areacol = areacol_list[0]
    neg_area_km2 = neg_df.iloc[:,areacol].sum()
    pos_area_km2 = pos_df.iloc[:,areacol].sum()
    return pos_area_km2 , neg_area_km2


# In[55]:


def area_frac_calc(metrics_3dfs,pcut,label_type=['pos','neg'],idx_labels=[0,1,2]):
    """
    Wrapper function
    Provide fractional and total land area with a statisticaly significant trend for three input datasets.

    Long Description
    ----------------

    Inputs
    ------
    metrics_3dfs : list of DataFrames
        must have 3 dataframes
    pcut : float
        p-value cutoff to determine statistically-significant slopes
    label_type : list
        default = 'pos_neg'
        column labels
    """
    if 'wet_dry' in label_type:
        col_labels = ['wet','dry']
        idx_labels = ['FO','DA','OL']
    elif 'pos_neg' in label_type:
        col_labels = ['pos','neg']
        idx_labels = ['FOOL','DAOL','FODA']
    FO_poskm2 , FO_negkm2 = pos_neg_area_calc(metrics_3dfs[0],pcut)
    DA_poskm2 , DA_negkm2 = pos_neg_area_calc(metrics_3dfs[1],pcut)
    OL_poskm2 , OL_negkm2 = pos_neg_area_calc(metrics_3dfs[2],pcut)


    km2_data = [[FO_poskm2 , FO_negkm2],
                [DA_poskm2 , DA_negkm2],
                [OL_poskm2 , OL_negkm2]]
    km2_df = pd.DataFrame(km2_data,
                          columns = col_labels,
                          index = idx_labels).astype(int)
    print('\nLand area in km^2\n---\n',km2_df)
    
    # get indices of the columns that contain pixel areas
    area_cols = [[df.columns.get_loc(col) for col in df.columns if 'area' in col][0] for df in metrics_3dfs]
    
    # divide km2 values by total area
    frac_data = [[FO_poskm2 , FO_negkm2]/(metrics_3dfs[0].iloc[:,area_cols[0]].sum()),
            [DA_poskm2 , DA_negkm2]/metrics_3dfs[1].iloc[:,area_cols[1]].sum(),
            [OL_poskm2 , OL_negkm2]/metrics_3dfs[2].iloc[:,area_cols[2]].sum()]
    # Add in non-significant fraction
    full_frac_data = [np.append(arr , 1-np.sum(arr)) for arr in frac_data]
    
    # convert to dataframe
    col_labels.append('non')
    frac_df = pd.DataFrame(full_frac_data,
                          columns = col_labels,
                          index = idx_labels)

    print('\nFraction of total land\n---\n',frac_df)
    return frac_df, km2_df