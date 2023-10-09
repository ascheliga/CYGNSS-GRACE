# In[2]:


# Import packages
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

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


def area_frac_calc(metrics_3dfs,pcut,col_labels=['pos','neg'],idx_labels=[0,1,2]):
    """
    Wrapper function
    Provides fractional and total land area with a statisticaly significant trend for three input datasets.

    Long Description
    ----------------

    Inputs
    ------
    metrics_3dfs : list of DataFrames
        must have 3 dataframes in list
        each dataframe must contain a 'slope', 'p-value', and 'area' column
    pcut : float
        p-value cutoff to determine statistically-significant slopes
    col_labels : list
        default = ['pos','neg']
        first two column labels of output dataframe
    idx_labels : list
        default = [0,1,2]

    Outputs
    -------
    frac_df : Pandas DataFrame
        3x3 dataframe containing positive, negative, and non-significant trend areas as fraction of total area
    frac_df, km2_df : Pandas DataFrame
        3x3 dataframe containing positive, negative, and non-significant trend areas in units of square kilometers 
    """
    km2_data = [pos_neg_area_calc(df,pcut) for df in metrics_3dfs]
    
    # Back-up version of km2_data calc
    # pos_area_0 , neg_area_0 = pos_neg_area_calc(metrics_3dfs[0],pcut)
    # pos_area_1 , neg_area_1 = pos_neg_area_calc(metrics_3dfs[1],pcut)
    # pos_area_2 , neg_area_2 = pos_neg_area_calc(metrics_3dfs[2],pcut)
    # km2_data = [[pos_area_0 , neg_area_0],
    #             [pos_area_1 , neg_area_1],
    #             [pos_area_2 , neg_area_2]]
    km2_df = pd.DataFrame(km2_data,
                          columns = col_labels,
                          index = idx_labels).astype(int)
    print('\nLand area in km^2\n---\n',km2_df)
    
    # get indices of the columns that contain pixel areas
    area_cols = [[df.columns.get_loc(col) for col in df.columns if 'area' in col][0] for df in metrics_3dfs]
    
    # divide km2 values by total area
    frac_data = [km2_data.iloc[0,:]/(metrics_3dfs[0].iloc[:,area_cols[0]].sum()),
            km2_data.iloc[1,:]/metrics_3dfs[1].iloc[:,area_cols[1]].sum(),
            km2_data.iloc[2,:]/metrics_3dfs[2].iloc[:,area_cols[2]].sum()]
    # Add in non-significant fraction
    full_frac_data = [np.append(arr , 1-np.sum(arr)) for arr in frac_data]
    
    # convert to dataframe
    col_labels.append('non')
    frac_df = pd.DataFrame(full_frac_data,
                          columns = col_labels,
                          index = idx_labels)

    print('\nFraction of total land\n---\n',frac_df)
    return frac_df, km2_df

def GRACE_areal_average(input_cmwe,input_mascon):
    """
    Calculate the weighted average from given cmwe and mascons

    Inputs
    ------
    input_cmwe : pd.DataFrame
        GRACE cmwe solutions
        each row is a mascon, each column is a timestep
    input_mascon : pd.DataFrame
        GRACE mascon metadata with a column called 'area_km2'
    """
    areal_average = input_cmwe.mul(input_mascon['area_km2'],axis='index').sum(axis=0)/(input_mascon['area_km2'].sum())
    return areal_average