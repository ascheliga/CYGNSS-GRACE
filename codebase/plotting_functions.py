
# Define directories

func_dir = '/global/home/users/ann_scheliga/CYGNSS-GRACE/codebase/'

# Import packages
import os
import sys
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import geopandas as gpd

import area_calcs

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

def statsig_map(input_gdf,ax,count,cmaps="BrBG",pie_row = [],cbar_flag='', **plot_params):
    """
    Plots a map of slope values with an option for a pie chart inset.

    Long Description
    ----------------

    Inputs
    ------
    input_gdf : GeoDataFrame
        plots the 'slope' column from this input gdf
    ax : axis-object
        existing axis to plot the map on
    count : int
        index/count of the map within a for loop
        used for indexing plot_params
        if statsig_map not used within a for loop, try setting count to 0
    cmaps : Colormap name or Colomap object
        default = 'BrBG'
        name of a Matplotlib-accepted Colormap name
        will use the 90th and 10th percentiles of the colormap
    pie_row : array
        default = []
        data for pie chart using pie_from_series function
        if empty array, then will not plot a pie chart
    cbar_flag : str
        default = ''
        determines if colorbar on map is vertical or horizontal
        looks for 'ver' or 'hor' in the string, respectively
    **plot_params : dict
        dictionary of plot formatting options and labels
        Keys used:
            'titles' : list of strings
            'x_labels' : list of strings
            'y_labels' : list of strings
            'legend_label' : str
            'vmin' : float
            'vmax' : float
        for dictionary values formatted as lists, the count parameter selects from the list

    Outputs
    -------
    None
    """
    # Create colormap
    if isinstance(cmaps, str):
        cmap = mpl.cm.get_cmap(cmaps)

    # Pull colormap bounds
    if 'vmin' in plot_params:
        commin = plot_params['vmin']
    else:
        commin = None
    if 'vmax' in plot_params:
        commax = plot_params['vmax']
    else:
        commax = None

    # Plot
    if 'hor' in cbar_flag.lower():
        input_gdf.plot('slope',cmap,vmin=commin,vmax=commax,ax=ax,legend=True,
                        legend_kwds={'label': plot_params['legend_label'],'orientation': "horizontal"})
    elif 'ver' in cbar_flag.lower():
        input_gdf.plot('slope',cmap,vmin=commin,vmax=commax,ax=ax,legend=True,
                        legend_kwds={'label': plot_params['legend_label'],'orientation': "vertical"})
    else:
        input_gdf.plot('slope',cmap,vmin=commin,vmax=commax,ax=ax)

    # Go through plotting parameters
    if 'titles' in plot_params:
        ax.set_title(plot_params['titles'][count])
    if 'x_labels' in plot_params:
        ax.set_xlabel(plot_params['x_labels'][count])
    if 'y_labels' in plot_params:
        ax.set_ylabel(plot_params['y_labels'][count])
    ax.set_facecolor('grey')
    
    if len(pie_row) != 0:
        small = ax.inset_axes([0.05 , 0.1 , 0.13 , 0.26])
        pie_from_series(pie_row,small,cmaps)

def tri_figuremap(input_3gdfs,cmaps="BrBG", n_rows = 3, n_cols = 1, cbar_flag = 'hor',pcut = 0.01, **plot_params):
    """
    
    Long Description
    ----------------
    When subplot grid is 1x3 or 3x1 and cbar_flag given, extra plot_params and formatting are built-in

    Inputs
    ------
    input_3gdfs : array-like of 3 GeoDataFrames 
    cmaps : Colormap name
        default = 'BrBG'
        name of a Matplotlib-accepted Colormap name
        will use the 90th and 10th percentiles of the colormap
    n_rows : int
        default = 3
        number of rows of the subplot grid
    n_cols : int
        default = 1
        number of columns of the subplot grid
    cbar_flag : str
        default = 'hor'
        determines if colorbar on map is vertical or horizontal
        looks for 'ver' or 'hor' in the string, respectively
    pcut : float
        default = 0.01
        p-value cutoff to determine statistically-significant slopes
    **plot_params : dict
        dictionary of plot formatting options and labels
        Keys used: 
            'piechart' : boolean
            'titles' : list of strings
            'x_labels' : list of strings
            'y_labels' : listof strings
            'legend_label' : str
        each dictionary value formatted as a list of strings
        count parameter selects string from each dictionary value
    
    Outputs
    -------
    axs : axes object or array of axes
        formed from plt.subplots()
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
        frac_df, __ = area_calcs.area_frac_calc(input_3gdfs,
                   pcut,
                   area_calc_type)
    else:
        # if no piechart or 'piechart' is false, creates empty pd.Series to keep later inputs from breaking
        frac_df = pd.Series(index=['','',''])

    if n_rows == 3 and 'hor' in cbar_flag: # if statement to not break the subplots if only one row
        fig, axs = plt.subplots(n_rows,n_cols,gridspec_kw={'height_ratios': [1, 1, 1.5]},figsize=[10,18],facecolor='white')
        plot_params['x_labels'] = ['','','Longitude (\N{DEGREE SIGN})']
        plot_params['y_labels'] = ['Latitude (\N{DEGREE SIGN})','Latitude (\N{DEGREE SIGN})','Latitude (\N{DEGREE SIGN})']
    elif n_cols == 3 and 'ver' in cbar_flag:
        fig, axs = plt.subplots(n_rows,n_cols,gridspec_kw={'width_ratios': [1, 1, 1.2]},figsize=[9,2],facecolor='white')
        plot_params['x_labels'] = ['Longitude (\N{DEGREE SIGN})','Longitude (\N{DEGREE SIGN})','Longitude (\N{DEGREE SIGN})']
        plot_params['y_labels'] = ['Latitude (\N{DEGREE SIGN})','','']
    else:
        fig, axs = plt.subplots(n_rows,n_cols,figsize=[10,18],facecolor='white')

    for count, axi, gdf, pie_idx in zip(range(len(axs)), axs,input_3gdfs,frac_df.index):
        # create trend map for each input_gdf. Will add a colorbar to the last plot unless plot_params['cbar_off'] is True
        if axi == axs[-1] and cbar_flag:
            statsig_map(gdf,axi,count,cmaps,pie_row=frac_df.loc[pie_idx],
                        cbar_flag=cbar_flag,**plot_params)
        else:
            statsig_map(gdf,axi,count,cmaps,pie_row=frac_df.loc[pie_idx],**plot_params)
        
    return axs

def three_part_timeseries(input3dfs,**plot_params):
    """
    Plot three time series with one shared x-axis and three separate y-axes.

    Inputs
    ------
    input3dfs : list of Pandas DataFrames
        Three Pandas DataFrames or Series with x-axis in the index
    **plot_params : dict
        dictionary of plot formatting options and labels
            Keys used:
                'line_fmt' : list of strings
                'title' : string
                'x_label' : string
                'y_labels' : list of strings, axis labels
                'data_labels' : list of strings, legend labels

    Outputs
    -------
    ax : axis object
    """
    fig, ax = plt.subplots(facecolor='white',figsize=plot_params['figsize'])
    fig.subplots_adjust(right=0.75)

    if len(input3dfs) == 2:
        print('Only plotting two timeseries')
        drop_twin2 = True
    else: drop_twin2=False
        
    if 'grid' in plot_params:
        plt.grid()
    
    twin1 = ax.twinx()
    if not drop_twin2:
        twin2 = ax.twinx()

        # Offset the right spine of twin2.  The ticks and label have already been
        # placed on the right by twinx above.
        twin2.spines.right.set_position(("axes", 1.1))
        
    p1, = ax.plot(input3dfs[0],plot_params['line_fmt'][0],label=plot_params['data_labels'][0]) 
    p2, = twin1.plot(input3dfs[1],plot_params['line_fmt'][1],label=plot_params['data_labels'][1])
    if not drop_twin2:
        p3, = twin2.plot(input3dfs[2],plot_params['line_fmt'][2],label=plot_params['data_labels'][2])

    if 'x_ticks' in plot_params and 'year' in plot_params['x_ticks']:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        myFmt = mdates.DateFormatter('%Y')
        ax.xaxis.set_major_formatter(myFmt)
    
    ax.set_title(plot_params['title'])
    ax.set_xlabel(plot_params['x_label'])
    ax.set_ylabel(plot_params['y_labels'][0])
    twin1.set_ylabel(plot_params['y_labels'][1])
    if not drop_twin2:
        twin2.set_ylabel(plot_params['y_labels'][2])

    ax.yaxis.label.set_color(p1.get_color())
    twin1.yaxis.label.set_color(p2.get_color())
    if not drop_twin2:
        twin2.yaxis.label.set_color(p3.get_color())

    tkw = dict(size=4, width=1.5)
    ax.tick_params(axis='x', **tkw)
    ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
    twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    if not drop_twin2:
        twin2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    

    if drop_twin2:
        ax.legend(handles=[p1, p2])
    else:
        ax.legend(handles=[p1, p2, p3])
    

    plt.show()
    return ax