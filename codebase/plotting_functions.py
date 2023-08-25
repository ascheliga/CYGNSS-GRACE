
# Define directories

calcdir = '/global/home/users/ann_scheliga/globalTWStrends/GRACEFO/'

# In[49]:

# Import packages
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd

os.chdir(calcdir)
import dry_wet_areas as dw_funcs


# In[3]:

def pie_from_series(row,axi,cmaps="BrBG"):
    """
    Plots a three-wedge pie chart on an existing axis object

    Long Description
    ----------------
    Function copied from GRACE_SLR project
    first row value corresponds to 90th percentiles of the colormap
    second row value corresponds to 10th percentiles of the colormap
    third row value is hard-coded to white

    Inputs
    ------
    row : 1-D array-like
        length = 3
        wedge sizes of pie chart
        typically, row is a pd.DataFrame row sliced to a pd.Series
    axi : axis-object
        existing axis to plot the pie chart on
    cmaps : Colormap name
        default = 'BrBG'
        name of a Matplotlib-accepted Colormap name
        will use the 90th and 10th percentiles of the colormap

    Outputs
    -------
    None
    """
    cmap = mpl.cm.get_cmap(cmaps)
    
    axi.pie(row, autopct='%2.0f',pctdistance=1.4,#     labels = row.index,
            colors = [cmap(0.9) , cmap(0.1),'white'],
            wedgeprops = {"edgecolor" : "black",
                          "linewidth": 1,
                          'antialiased': True})


# In[124]:


def statsig_map(input_gdf,ax,count,cmaps="BrBG",title = '',pie_row = [],cbar_flag='', pcut = 0.01,**plot_params):
    """

    Long Description
    ----------------

    Inputs
    ------
    input_gdf
    ax
    count
    cmaps="BrBG"
    title = ''
    pie_row : array
        default = []
        data for pie chart using pie_from_series function
        if empty array, then will not plot a pie chart
    cbar_flag : str
        default = ''
        determines if colorbar on map is vertical or horizontal
        looks for 'ver' or 'hor' in the string, respectively
    pcut = 0.01
        Leftover from SLR project, not used
    **plot_params

    Outputs
    -------

    """
    # Left over from SLR project. I don't think it is necessary?
    # dry_df , dry_bool = dw_funcs.stat_check(input_gdf,'dry',pcut)
    # wet_df , wet_bool = dw_funcs.stat_check(input_gdf,'wet',pcut)
    # plot_bool = dry_bool + wet_bool

    if 'hor' in cbar_flag.lower():
        input_gdf.plot('slope',cmaps,vmin=commin,vmax=commax,ax=ax,legend=True,
                        legend_kwds={'label': plot_params['legend_label'],'orientation': "horizontal"})
    elif 'ver' in cbar_flag.lower():
        input_gdf.plot('slope',cmaps,vmin=commin,vmax=commax,ax=ax,legend=True,
                        legend_kwds={'label': plot_params['legend_label'],'orientation': "vertical"})
    else:
        input_gdf.plot('slope',cmaps,vmin=commin,vmax=commax,ax=ax)

    ax.set_title(title)
    ax.set_ylabel(plot_params['y_label'][count])
    ax.set_xlabel(plot_params['x_label'][count])
    ax.set_facecolor('grey')
    
    if pie_row.any():
        small = ax.inset_axes([0.05 , 0.1 , 0.13 , 0.26])
        pie_from_series(pie_row,small,cmaps)

# In[131]:


def tri_figuremap(input_3gdfs,cmaps="BrBG", n_rows = 3, n_cols = 1, cbar_flag = 'hor',pcut = 0.01, **plot_params):
    """
    
    Long Description
    ----------------

    Inputs
    ------
    input_3gdfs
    cmaps="BrBG"
    n_rows = 3
    n_cols = 1
    cbar_flag = 'hor'
    pcut = 0.01
    **plot_params
    
    Outputs
    -------
    axs
    """
    if 'titles' not in plot_params: # if no titles provided, create blank variable
        plot_params['titles'] = ['','','']
        
    if 'piechart' in plot_params and plot_params['piechart']:
        # if piechart is a plot param and is true calculates total trend areas of each input gdf
        # pcut value must be defined earlier in script
        if   cmaps == "BrBG":
            area_calc_type = 'wet_dry'
        elif cmaps == "RdBu":
            area_calc_type = 'pos_neg'
        frac_df, __ = dw_funcs.area_frac_calc(input_3gdfs,
                   pcut,
                   area_calc_type)
    else:
        # if no piechart or 'piechart' is false, creates empty pd.Series to keep later inputs from breaking
        frac_df = pd.Series(index=['','',''])

    if n_rows == 3 and 'hor' in cbar_flag: # if statement to not break the subplots if only one row
        fig, axs = plt.subplots(n_rows,n_cols,gridspec_kw={'height_ratios': [1, 1, 1.5]},figsize=[10,18],facecolor='white')
        plot_params['x_label'] = ['','','Longitude (\N{DEGREE SIGN})']
        plot_params['y_label'] = ['Latitude (\N{DEGREE SIGN})','Latitude (\N{DEGREE SIGN})','Latitude (\N{DEGREE SIGN})']
    elif n_cols == 3 and 'ver' in cbar_flag:
        fig, axs = plt.subplots(n_rows,n_cols,gridspec_kw={'width_ratios': [1, 1, 1.2]},figsize=[9,2],facecolor='white')
        plot_params['x_label'] = ['Longitude (\N{DEGREE SIGN})','Longitude (\N{DEGREE SIGN})','Longitude (\N{DEGREE SIGN})']
        plot_params['y_label'] = ['Latitude (\N{DEGREE SIGN})','','']
    else:
        fig, axs = plt.subplots(n_rows,n_cols,figsize=[10,18],facecolor='white')

    for count, axi, gdf, title, x_label, y_label, pie_idx in zip(range(len(axs)), axs,input_3gdfs,plot_params['titles'],plot_params['x_label'],plot_params['y_label'],frac_df.index):
        # create trend map for each input_gdf. Will add a colorbar to the last plot unless plot_params['cbar_off'] is True
        if axi == axs[-1] and cbar_flag:
            statsig_map(gdf,axi,count,cmaps,title=title,pie_row=frac_df.loc[pie_idx],
                        cbar_flag=cbar_flag,**plot_params)
        else:
            statsig_map(gdf,axi,count,cmaps,title=title,pie_row=frac_df.loc[pie_idx],**plot_params)
        
    return axs