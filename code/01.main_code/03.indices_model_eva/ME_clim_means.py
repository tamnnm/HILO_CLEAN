import os
import numpy as np
import xarray as xr
import json
import pandas as pd
import scipy.stats as sst
import multiprocessing

# Plotting and visualization
import matplotlib
import matplotlib.style
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm
from matplotlib.text import TextPath
# from matplotlib.tri import Triangulation
# from matplotlib.animation import FuncAnimation
# from matplotlib.image import imread
from matplotlib.patches import FancyBboxPatch, PathPatch
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
# from matplotlib.ticker import MultipleLocator
import seaborn as sns
import proplot as pplt

from matplotlib.offsetbox import AnchoredText, TextArea, VPacker, HPacker, AnnotationBbox, PaddedBox
import skill_metrics as sm

# Geospatial libraries
# import geopandas as gpd
import shapefile as shp
# import cartopy.feature as cfeature
import cartopy.crs as ccrs


# Import other pythons
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ME_extreme_indices import check_non_exist, MetricExtreme, city_ds_pts
from ME_man_kandell import list_colors
from constant import * # Import constants like Data_csv, Data_ind, Data_json, Data_shp, Data_obs, Data_csv, img_wd
from my_junk import *

from math import pi
from typing import Union, Optional, List, Tuple, Dict, Literal, Callable

print("Importing modules")

params = {
    'axes.titlesize': 20,
    'axes.labelsize': 15,
    'axes.labelpad': 10,
    'axes.linewidth': 1.5,
    'axes.edgecolor': 'gray',
    'font.size': 15,
    'font.family': 'cmss10',
    'mathtext.fontset': 'stixsans',
    'legend.fontsize': 15,
    'legend.title_fontsize': 20,
    'legend.loc': 'lower left',
    'legend.labelspacing': 0.25,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    # 'lines.linewidth': 3,
    # 'text.usetex': True,
    # 'figure.autolayout': True,
    'ytick.right': False,
    'xtick.top': False,

    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.minor.size': 5,
    'ytick.minor.size': 5,

    'xtick.major.width': 3,
    'ytick.major.width': 3,
    'xtick.minor.width': 3,
    'ytick.minor.width': 3,

    'xtick.major.pad': 10,
    'ytick.major.pad': 12,
    # 'xtick.minor.pad': 14,
    # 'ytick.minor.pad': 14,

    'figure.facecolor': 'none',
    'axes.facecolor': 'none',

    'savefig.transparent': True,
    'savefig.bbox': 'tight',
    'savefig.dpi': 300,

    'xtick.direction': 'inout',
    'ytick.direction': 'inout',
    'axes.unicode_minus': False,

}
plt.clf()
matplotlib.rcParams.update(params)

with open(json_path, 'r') as json_file:
    data_dict = json.load(json_file)
    data_dict['dataset'] = ['cera','era','era5','noaa']
with open(json_path, 'w') as json_file:
    json.dump(data_dict, json_file)

station_data = pd.read_csv("/work/users/tamnnm/code/01.main_code/01.city_list_obs/city/rain_city.txt", names=[
                      'name_station', 'first_year', 'last_year', 'lon', 'lat', 'elev', 'appendix'])
map_pro = ccrs.PlateCarree()

img_wd = os.path.join(os.getenv("img"), "03")


Cache = DataCache()

"""
Step 1: Get the mean temperature => Annual + seasonal MEAN
Step 2: Produce needed metrics
Step 3: Plot
"""
#*DONE

print("Metric Extreme run")
metricRun = MetricExtreme()
# metricRun(var='T2m', season_option='all', list_ind_option='Tm')()
# metricRun(var='R', season_option='all', rerun="all", list_ind_option='Rm')()
# metricRun(var='T2m', season_option='seasonal', list_ind_option='Tm_mon')()
# metricRun(var='R', season_option='seasonal', rerun = "all", list_ind_option='Rm_mon')()

# Configuration
seasons = ['annual', 'DJF', 'MAM', 'JJA', 'SON']
datasets = ['cera', 'era', 'era5', 'noaa']
datasets_name = {'cera': 'CERA-20C', 'era': 'ERA-20C', 'era5': 'ERA5', 'noaa': '20CR'}
markers = ['o', 's', 'd', '^', 'v']
metrics_compare = ['R','rmse', 'mae', 'mpe', 'mape']

def plot_correlation_maps(variable,metric):
    fig = plt.figure(figsize=(9, 20))

    # Setup figure
    gs = gridspec.GridSpec(1,2, width_ratios=[2.2,1.3])

    gs_main = gs[0].subgridspec(len(seasons), len(datasets) - 1,\
                                    width_ratios=[1,1,1], height_ratios=[1]*len(seasons), hspace = -0.1, wspace = -0.6)
    gs_last_column = gs[1].subgridspec(len(seasons), 1, height_ratios=[1]*len(seasons), hspace = -0.1)

    gs.update(wspace = -0.33)

    # Create all subplots first
    axes = []
    for i in range(5):  # 5 rows for seasons
        for j in range(3):  # 4 columns for datasets
            ax = fig.add_subplot(gs_main[i, j], projection=ccrs.PlateCarree())
            axes.append(ax)
        ax_1 = fig.add_subplot(gs_last_column[i, 0], projection=ccrs.PlateCarree())
        axes.append(ax_1)


    # Red: d77e78, circle: c84943/943631
    # Blue: 70a5cd, circle: 214b6c

     # Define size thresholds and legend elements
    size_thresholds = {
        'R': [0.3, 0.5, 0.7],  # R thresholds: 0-0.3, 0.3-0.5, 0.5-0.7, >0.7
        'default': [0.25, 0.5, 0.75]  # Default thresholds for other metrics
    }

    # Size mapping based on thresholds (smallest to largest)
    sizes = [20, 40, 80, 120]   # Circle sizes

    legend_elements = []

    # Define colormaps PROPERLY
    n_levels = 6
    if variable == "R":
        # Precipitation - use cubehelix
        cmap = sns.cubehelix_palette(n_levels, start=.5, rot=-.5)
        discrete_cmap = mcolors.ListedColormap(cmap)
        unit = 'mm'
    else:
        # Temperature - use your custom palette
        cmap = sns.color_palette("ch:s=-.2,r=.6", n_colors=n_levels)
        discrete_cmap = mcolors.ListedColormap(cmap)
        unit = r'$^\circ C$'

    #  cor_map = sns.diverging_palette(220, 20, as_cmap=True)
    # cor_map = pplt.Colormap('Blues4_r', 'Reds3', name='Diverging', save=True)
    cor_map = sns.color_palette("coolwarm", n_colors=10, as_cmap=True)
    # Process each season and dataset
    for i, season in enumerate(seasons):

        #* GENERAL: DATASET SETUP
        # Load data
        filename_basename = f"{variable}_{metric}_sup_{season}_merge"
        filepath = os.path.join(Data_csv, 'ensemble', filename_basename)

        if os.path.exists(filepath + '.nc'):
            filepath += '.nc'
        else:
            filepath += '.csv'

        try:

            df = Cache.get_data(filepath)
        except FileNotFoundError:
            raise KeyError(f"File not found: {filepath}")

        meta_df = station_data[['name_station', 'lat', 'lon']]

        if isinstance(df, pd.DataFrame):
            merged_df = pd.merge(df, meta_df,
                                on='name_station', how='left')
        else:
            merged_df = df

        for j, dataset in enumerate(datasets):

            #* GENERAL: PLOT SETUP
            ax = axes[i*4+j]

            # Set transparent background
            ax.set_facecolor('none')

            # Plot island
            if ax == axes[i*4+3]:
                plot_shp(ax,dem=False, island_full = True, font_scale = 3 if metric == 'R' else 2.5)
            else:
                plot_shp(ax,dem=False, extend = [101, 111, 7, 25], font_scale =3 if metric == 'R' else 2.5)


            # Calculate size bins, facecolor, and edgecolor for all points at once
            thresholds = size_thresholds['R'] if metric == 'R' else size_thresholds['default']

            #* PLOT STEP: Plot scatter points
            #? Correlation coefficient
            if metric == 'R':
                continue
                plot_df = pd.merge(
                    merged_df.sel(dataset=dataset).to_dataframe(),
                    meta_df,
                    on='name_station', how='left')

                #? Significance
                plot_df['sig'] = plot_df['pvalue'] < 0.05
                plot_df['bin_size'] = np.digitize(np.abs(plot_df['R']), thresholds)
                plot_df['size'] = plot_df['bin_size'].map(lambda x: sizes[x])

                # Set fixed edgecolor and facecolor based on the dataset
                # plot_df['edgecolor'] = np.where(plot_df['sig'],np.where(plot_df['R'] >= 0, '#943631', '#214b6c'),'gray')  # Red for positive, blue for negative
                plot_df['edgecolor'] = np.where(plot_df['sig'], 'k', 'gray')  # Black for significant, gray for non-significant
                plot_df['hatch'] = np.where(plot_df['sig'], '', '/')  # Hatching for non-significant points
                plot_df['linesize'] = np.where(plot_df['sig'], 2, 0.5)  # Thicker lines for significant points
                # plot_df['facecolor'] = np.where(plot_df['sig'],np.where(plot_df['R'] >= 0, '#c84943', '#70a5cd'),"none") #Red for positive, blue for negative, none for non-significant

                #! KEEP THE LOOP
                #? There is no other ways to use directly plot_df['facecolor']. I tried
                # Plot scatter points colored by correlation coefficient

                # If plot by SIZE
                # for k in range(len(plot_df)):
                    # ax.scatter(
                    #     plot_df['lon'].iloc[k],
                    #     plot_df['lat'].iloc[k],
                    #     s=plot_df['size'].iloc[k],
                    #     facecolor=plot_df['facecolor'].iloc[k],
                    #     edgecolor='none',
                    #     linewidth=1.5,
                    #     alpha = 0.8,
                    #     transform=map_pro,
                    #     marker='o',  # Explicitly set circle marker
                    # )

                # Plot edgecolor online
                scatter = ax.scatter(
                    plot_df['lon'],
                    plot_df['lat'],
                    # s = plot_df['size'],
                    facecolor='none',
                    # edgecolor = 'k',
                    edgecolor=plot_df['edgecolor'],
                    linewidth= plot_df['linesize'],
                    s = 120,
                    transform=map_pro,
                    marker='o',  # Explicitly set circle marker
                    linestyle='-',  # Solid line for edge
                )

                # Filled inside
                scatter = ax.scatter(
                    plot_df['lon'],
                    plot_df['lat'],
                    c=plot_df['R'],
                    cmap = cor_map,
                    vmin = -1,
                    vmax = 1,
                    # norm = norm,
                    s=100,
                    alpha=0.9,
                    linewidth=1,
                    transform=map_pro,
                    marker = 'o',
                    # hatch = plot_df['hatch'],  # Hatching for non-significant points
                    # Explicitly set circle marker
                )


            else:
                #* PLOT STEP: Specify configuration
                plot_df = merged_df.copy()

                scatter = ax.scatter(
                    plot_df['lon'],
                    plot_df['lat'],
                    c=plot_df[dataset],
                    cmap = discrete_cmap,
                    facecolor = "none",
                    edgecolor = "none",
                    # norm = norm,
                    s=60,
                    alpha=0.8,
                    linewidth=3,
                    transform=map_pro,
                    marker = 'o',  # Explicitly set circle marker

                )
                # # Plot scatter points colored by value
                # scatter_2 = ax.scatter(
                #     plot_df['lon'],
                #     plot_df['lat'],
                #     # c=plot_df[dataset],
                #     facecolor = "none",
                #     edgecolor = discrete_cmap(plot_df[dataset]),
                #     # norm = norm,
                #     s=60,
                #     linewidth=1,
                #     transform=map_pro,
                #     marker = 'o',
                #     linestyle='-',  # Solid line for edge
                # )


            #* PLOT STEP: Add title elements
            if i == 0:
                if j !=3:
                    # Remove default title and add custom text instead
                    ax.set_title("")  # Clear any default title

                    # Create title text at custom position
                    ax.text(0.4, 0.98, datasets_name[dataset],
                                    transform=ax.transAxes,
                                    fontsize=15,
                                    ha='center',
                                    va='center',
                                    fontweight='bold',
                                    )

                    # Create grey box for title
                    box = FancyBboxPatch((0.15, 0.95),  # x, y (bottom left corner)
                                    0.5,        # width (full subplot width)
                                    0.075,        # height
                                    transform=ax.transAxes,
                                    facecolor='lightgrey',
                                    edgecolor='none',
                                    alpha=0.7,
                                    lw=1.5,
                                    clip_on = False,
                                    boxstyle="round,pad=0.01",
                                    )   # put behind text
                    ax.add_patch(box)
                else:
                    ax.text(0.16, 0.98, datasets_name[dataset],
                                    transform=ax.transAxes,
                                    fontsize=15,
                                    ha='left',
                                    va='center',
                                    fontweight='bold',
                                    )
                    box = FancyBboxPatch((0.1, 0.95),  # x, y (bottom left corner)
                                    0.35,        # width (full subplot width)
                                    0.075,        # height
                                    transform=ax.transAxes,
                                    facecolor='lightgrey',
                                    edgecolor='none',
                                    alpha=0.7,
                                    lw=1.5,
                                    clip_on = False,
                                    boxstyle="round,pad=0.01",
                                    )   # put behind text
                    ax.add_patch(box)

            if j == 0:
                box = FancyBboxPatch((-0.1, 0.1),  # x, y (bottom left corner)
                                0.15,         # width
                                0.8,         # height (full subplot height)
                                transform=ax.transAxes,
                                facecolor='lightgrey',
                                edgecolor='none',
                                alpha=0.7,
                                lw=1.5,
                                clip_on = False,
                                boxstyle="round,pad=0.01",
                                )    # put behind text
                ax.text(-0.075, 0.5, season.upper(),
                        transform=ax.transAxes,
                        fontsize=15,
                        rotation=90,
                        va='center',
                        fontweight='bold',
                        )
                ax.add_patch(box)

    #* PLOT STEP: Add common labels

    # TODO: Limit of the colorbar

    #? Legend: Circle sizes
    if metric == 'R':
        return
        cbar_ax = fig.add_axes([0.98, 0.15, 0.02, 0.7])
        cbar_label = "Correlation Coefficient"
        # Size legend
        # for i in range(len(size_thresholds)+1):
        #     if i == 0: label = f"0 - {size_thresholds['R'][0]}"
        #     elif i == len(size_thresholds): label = f"{size_thresholds['R'][-1]} - 1.0"
        #     else: label = f"{size_thresholds['R'][i-1]} - {size_thresholds['R'][i]}"

        #     legend_elements.append(
        #         mlines.Line2D([], [], marker='o', color='k',
        #                        markerfacecolor='w', markeredgecolor='k',
        #                        markersize=sizes[i]*0.6, label=label, linestyle='None')
        #     )

        # cbar_ax.clear()
        # cbar_ax.set_axis_off()  # Hide the axis
        # # cbar_ax.set_visible(False)
        # legend = cbar_ax.legend(handles=legend_elements,# title = cbar_label,
        #                           loc='center', ncol=1,labelspacing=4,  # Vertical spacing
        #                             handletextpad=2,    # Space between symbol and text
        #                             borderpad=1, frameon=False)

        scatter.set_clim(-1, 1)  # Force full range

        cbar = fig.colorbar(scatter, cax=cbar_ax,
                            spacing='proportional',
                            ticks=np.linspace(-1, 1, 11),
                            boundaries=np.linspace(-1, 1, 11)
                            )
        # Add dividing lines between color levels
        cbar.outline.set_edgecolor('black')
        cbar.outline.set_linewidth(3)

        cbar.dividers.set_color('black')
        cbar.dividers.set_linewidth(3)

        cbar.ax.xaxis.set_tick_params(width=2, length=6, color='black')
        # cbar.ax.set_yticklabels([])
        cbar.ax.set_yticklabels([f'{x:.1f}' for x in np.linspace(-1, 1, 11)])
        # Refresh colorbar to enforce limits
        cbar.update_normal(scatter)

    else:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        vmin, vmax = scatter.get_clim()
        if variable == "R":
            vmin = 0
            vmax = 15
        else:
            vmin = 0
            vmax = 6
        cbar = fig.colorbar(scatter, cax=cbar_ax,
                            spacing='proportional',
                            # ticks=np.linspace(vmin, vmax, n_levels+1),
                            boundaries=np.linspace(vmin, vmax, n_levels+1), extend = "both")
        # Add dividing lines between color levels
        cbar.outline.set_edgecolor('black')
        cbar.outline.set_linewidth(3)
        cbar.dividers.set_color('black')
        cbar.dividers.set_linewidth(3)
        # Optional: Round the tick labels for better readability
        # cbar.ax.set_yticklabels([])
        cbar.set_ticks(np.linspace(vmin, vmax, n_levels+1))
        cbar.set_ticklabels([f'{x:.1f}' for x in np.linspace(vmin, vmax, n_levels+1)])

        if 'p' in metric:
            cbar_label = 'Percentage Error (%)'
        elif 'n' in metric:
            cbar_label = rf"Normalized Bias ({unit})"
        else:
            cbar_label = rf"Bias ({unit})"

        cbar.set_label(cbar_label)
        print(cbar_label)
        # Adjust layout
    # Save figure

    output_path = os.path.join(img_wd, f"{variable}_{metric}_maps.png")
    fig.savefig(output_path)

    print(f"Figure saved to {output_path}")
    plt.close(fig)
# TODO: plot the scatter plot
# - Produce annual + seasonal clim for observation and reanalysis
# - Plot the scatter plot for each season and each dataset
# - Add title for each season and each dataset
# - Add colorbar
# - Save figure

def plot_scatter(variable, ind = None):

    list_name_city = data_dict[f'name_{variable}_city']
    list_no_city = data_dict[f'no_{variable}_city']

    if ind is None:
        ind = "Tm" if variable == "T2m" else "Rm"

    # fig = plt.figure(figsize=(40, 70))
    # # Setup figure
    # gs_main = gridspec.GridSpec(len(seasons), len(datasets),\
    #                                 width_ratios=[1,1,1,1], height_ratios=[1]*len(seasons))
    # gs_main.update(hspace=0.05)


    # # Create all subplots first
    # axes = []
    # for i in range(5):  # 5 rows for seasons
    #     for j in range(4):  # 4 columns for datasets
    #         ax = fig.add_subplot(gs_main[i, j], projection=ccrs.PlateCarree())
    #         axes.append(ax)

    fig, axes = plt.subplots(5, 4, figsize=(15, 20))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    limit_R = {"annual": [0, 25,5], "DJF": [0, 15,5], "MAM": [0, 15,5], "JJA": [0, 40,10], "SON": [0, 40,10]}
    limit_T = {"annual": [10, 30,5], "DJF": [10, 30,5], "MAM": [10, 30,5], "JJA": [10, 35,5], "SON": [10, 30,5]}
    limit = limit_T if variable == "T2m" else limit_R
    # Process each season and dataset
    for j, dataset in enumerate(datasets):
        # Load observation
        ob_filename_ss = f"{ind}_obs_{dataset}_seasonal_clim.nc"
        ob_filename_ann = f"{ind}_obs_{dataset}_annual_clim.nc"
        try:
            obs_ss_ds = Cache.get_data(os.path.join(Data_ind, ob_filename_ss))
            obs_ann_ds = Cache.get_data(os.path.join(Data_ind, ob_filename_ann))
        except:
            print(f"File not found: {ob_filename_ss} or {ob_filename_ann}")

        # Load data
        filename_ss = f"{ind}_{dataset}_seasonal_clim.nc"
        filename_ann = f"{ind}_{dataset}_annual_clim.nc"
        try:
            ss_full_ds = Cache.get_data(os.path.join(Data_ind, filename_ss))
            ann_full_ds = Cache.get_data(os.path.join(Data_ind, filename_ann))
        except:
            print(f"File not found: {filename_ss} or {filename_ann}")
            sys.exit()

        pts_dict = city_ds_pts(dataset)

        final_data = []

        # Extract all observation data at once
        obs_ss_all = obs_ss_ds.sel(no_station=list_no_city)
        obs_ann_all = obs_ann_ds.sel(no_station=list_no_city)
        for k, name_city in enumerate(list_name_city):
            ss_ind = cut_mlp(dts=cut_points(ss_full_ds, pts_dict[name_city]), dim_mean=['geo'], var=variable, val_only=True)
            ann_data_ind = cut_mlp(dts=cut_points(ann_full_ds, pts_dict[name_city]), dim_mean=['geo'], var=variable, val_only=True)

            # Extract observation data for this city (already pre-selected)
            obs_ss_ind = cut_var(obs_ss_all.sel(no_station=list_no_city[k]), var=variable, val_only=True)
            ann_obs_ind = cut_var(obs_ann_all.sel(no_station=list_no_city[k]), var=variable, val_only=True)

            # sys.exit()
            ind_data = list(ann_data_ind) + list(ss_ind)
            obs_data = list(ann_obs_ind) + list(obs_ss_ind)

            final_data.append([np.array(ind_data), np.array(obs_data)])

        # Convert to array
        final_data = np.array(final_data)

        # Plot all data
        for i, season in enumerate(seasons):

            ax = axes[i, j]
            # Set transparent background
            # ax.set_facecolor('none')

            ind_data = final_data[:,0, i]
            obs_data = final_data[:,1, i]

            # min_val = min(min(ind_data), min(obs_data))
            # max_val = max(max(ind_data), max(obs_data))
            min_val = limit[season][0]
            max_val = limit[season][1]
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)
            tick_step = limit[season][2]  # Custom step size
            ticks = np.arange(min_val, max_val + tick_step, tick_step)

            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', linewidth=3)

            #TODO: Facecolor is none but in plot, it's filled
            marker_style = dict(marker=markers[j], s=250, edgecolor=list_colors[j], facecolor='none', linewidth=3, alpha = 0.75)
            # SEASONAL PLOT
            scatter = ax.scatter(
            final_data[:,0, i], final_data[:,1, i],
            # c = list_colors[j],
            **marker_style
            )
#* PLOT STEP: Add title elements
            if i == 0:
                # Remove default title and add custom text instead
                ax.set_title("")  # Clear any default title

                # Create title text at custom position
                ax.text(0.5, 1.05, datasets_name[dataset],
                                transform=ax.transAxes,
                                fontsize=20,
                                ha='center',
                                va='center',
                                fontweight='bold',
                                )

                # Create grey box for title
                box = FancyBboxPatch((0.15, 1.03),  # x, y (bottom left corner)
                                0.7,        # width (full subplot width)
                                0.09,        # height
                                transform=ax.transAxes,
                                facecolor='lightgrey',
                                edgecolor='none',
                                alpha=0.7,
                                lw=1.5,
                                clip_on = False,
                                boxstyle="round,pad=0.01",
                                )   # put behind text
                ax.add_patch(box)

            if j == 0:
                box = FancyBboxPatch((-0.45, 0.05),  # x, y (bottom left corner)
                                0.15,         # width
                                0.9,         # height (full subplot height)
                                transform=ax.transAxes,
                                facecolor='lightgrey',
                                edgecolor='none',
                                alpha=0.7,
                                lw=1.5,
                                clip_on = False,
                                boxstyle="round,pad=0.01",
                                )    # put behind text
                ax.text(-0.43, 0.5, season.upper(),
                        transform=ax.transAxes,
                        fontsize=20,
                        rotation=90,
                        va='center',
                        fontweight='bold',
                        )
                ax.add_patch(box)


    # fig.supxlabel('Model', fontsize=50)
    # fig.supylabel('Observation', fontsize=50)

    # Adjust layout
    # fig.tight_layout(rect=[0, 0, 0.9, 1])


    fig.supylabel("Observation", fontsize=32, x=0.01)
    fig.supxlabel("Reanalysis", fontsize=32, y=0.06)

    # Save figure
    output_path = os.path.join(img_wd, f"{variable}_{ind}_scatter.png")
    fig.savefig(output_path)
    print(f"Figure saved to {output_path}")
    plt.close(fig)

    # Process each season and dataset

def check_station(variable):

    ind = "Tm" if variable == "T2m" else "Rm"

    list_name_city = data_dict[f'name_{variable}_city']
    list_no_city = data_dict[f'no_{variable}_city']

    # obs_cache = {}
    # ind_cache = {}

    from functools import lru_cache
    from scipy.stats import pearsonr


    for k, name_city in enumerate(list_name_city):
        fig, ax = plt.subplots(4,2, figsize=(20, 45))

        for i, season in enumerate(seasons[1:]):
            obs_filename = f"{ind}_obs_{season}.nc"
            obs_mon_filename = f"{ind}_mon_obs_{season}.nc"
            # obs_full_filename = f"{ind}_full_obs_{season}.nc"

            obs_ds = Cache.get_data(os.path.join(Data_ind, obs_filename))
            obs_mon_ds = Cache.get_data(os.path.join(Data_ind, obs_mon_filename))
            # obs_full_ds = load_dataset(os.path.join(Data_ind, obs_full_filename))

            for j, dataset in enumerate(datasets):
                pts_dict = city_ds_pts(dataset)

                ind_filename = f"{ind}_{dataset}_{season}.nc"
                ind_mon_filename = f"{ind}_mon_{dataset}_{season}.nc"
                # ind_full_filename = f"{ind}_full_{dataset}_{season}.nc"

                # # Use cache for index datasets
                # if ind_filename not in ind_cache:
                #     ind_cache[ind_filename] = Cache.get_data(os.path.join(Data_ind, ind_filename)))
                # ind_ds = ind_cache[ind_filename]

                # if ind_full_filename not in ind_cache:
                #     ind_cache[ind_full_filename] = Cache.get_data(os.path.join(Data_ind, ind_full_filename)))
                # ind_full_ds = ind_cache[ind_full_filename]

                ind_ds = Cache.get_data(os.path.join(Data_ind, ind_filename))
                ind_mon_ds = Cache.get_data(os.path.join(Data_ind, ind_mon_filename))
                # ind_full_ds = load_dataset(os.path.join(Data_ind, ind_full_filename))

                time_y = ind_ds.time.dt.strftime('%Y')
                time_my = ind_mon_ds.time.dt.strftime('%m-%Y')
                # time_dmy = ind_full_ds.time.dt.strftime('%d-%m-%Y')

                obs_ind = cut_var(obs_ds.sel(no_station = list_no_city[k]), var = variable, val_only=True)
                ind_ind = cut_mlp(dts=cut_points(ind_ds, pts_dict[name_city]), dim_mean=['geo'], var=variable, val_only=True)
                obs_ind = obs_ind[:ind_ind.shape[0]]

                if season == "DJF":
                    obs_ind = obs_ind[1:-1]
                    ind_ind = ind_ind[1:-1]
                    time_y = time_y[1:-1]

                # Pearson correlation
                pearson_corr = pearson_correlation(obs_ind, ind_ind)

                obs_mon_ind = cut_var(obs_mon_ds.sel(no_station = list_no_city[k]), var = variable, val_only=True)
                ind_mon_ind = cut_mlp(dts=cut_points(ind_mon_ds, pts_dict[name_city]), dim_mean=['geo'], var=variable, val_only=True)
                obs_mon_ind = obs_mon_ind[:ind_mon_ind.shape[0]]

                pearson_mon_corr = pearson_correlation(obs_mon_ind, ind_mon_ind)
                # obs_full_ind = cut_var(obs_full_ds.sel(no_station = list_no_city[k]), var = variable, val_only=True)
                # ind_full_ind = cut_mlp(dts=cut_points(ind_full_ds, pts_dict[name_city]), dim_mean=['geo'], var=variable, val_only=True)
                # obs_full_ind = obs_full_ind[:ind_full_ind.shape[0]]

                print(f"Plotting timeseries for {name_city} in {season} for {dataset}")
                # Plot timeseries
                ax[i,0].scatter(obs_ind,ind_ind, marker = markers[j], color = list_colors[j], label = dataset, s = 500)
                ax[i,0].scatter(obs_ind, obs_ind, color = 'black', linestyle = '--', linewidth = 3)
                # ax[j,0].plot(time_y, ind_ind, marker = markers[i], color = list_colors[i], label = dataset, linewidth = 3)
                # ax[j,0].plot(time_y, obs_ind, color = 'black', linestyle = '--', linewidth = 3)
                ax[i,0].text(0.05 + i*0.2, 0.95, f"{dataset}:R={pearson_corr:.2f} ", transform=ax[i,0].transAxes, fontsize=50, va='top')

                ax[i,1].scatter(obs_mon_ind, ind_mon_ind, marker = markers[j], color = list_colors[j], label = dataset, s = 500)
                ax[i,1].scatter(obs_mon_ind, obs_mon_ind, color = 'black', linestyle = '--', linewidth = 3)
                # ax[j,1].plot(time_my, ind_mon_ind, marker = markers[i], color = list_colors[i], label = dataset, linewidth = 3)
                # ax[j,1].plot(time_my, obs_mon_ind, color = 'black', linestyle = '--', linewidth = 3)
                ax[i,1].text(0.05 + i*0.23, 0.95, f"{dataset}:R={pearson_mon_corr:.2f} ", transform=ax[i,1].transAxes, fontsize=50, va='top')

                # ax[j,2].plot(time_dmy, ind_full_ind, marker = markers[i], color = list_colors[i], label = dataset)
                # ax[j,2].plot(time_dmy, obs_full_ind, marker = markers[i], color = list_colors[i], label = dataset)

        ax[0,0].set_title(f"Interannual")
        ax[0,1].set_title(f"Intraseasonal - Monthly")

        for row in ax:
            for subplot in row:
                subplot.tick_params(axis='x', rotation=90)  # Rotate x-axis ticks by 90 degrees

        fig.savefig(os.path.join(img_wd, f"{name_city}_{variable}_timeseries_scatter.png"))
        plt.close(fig)

def text_marker(text, size=10):
    path = TextPath((0, 0), text, size=size)
    return PathPatch(path, color='k', lw=0)


def taylor_diagram(variable, type="inter", normalized=True):
    """
        Create a Taylor diagram comparing multiple datasets across seasons.

        Args:
            variable: 't2m' or 'pre' (temperature or precipitation)
            type: "intra" or "inter" (intraseasonal or interannual)
            normalized: bool, whether to normalize the data by the standard deviation of the observational data
        Vars:
            indice: Tm or Rm
    Step 1: Calculate the spatial average (output a time series for both reanalysis and observational)
    Step 2: Calculate the std, ccoef, crmsd for each season
    Step 3: Plot the Taylor diagram

    """

    # Number of panels
    # nP = 2

    # Create figure with extra space for title boxes
    fig = plt.figure(figsize=(12, 10))

    # Configure datasets based on type
    # Colors for each dataset
    # Markers for each season
    markers = ['o', 's', 'd', '^', 'v']

    # TASK: Declare figure
    # Create main axes for Taylor diagram
    ax = fig.add_subplot(111)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    # TASK: Calculate the std, ccoef, crmsd for each season

    # Legend elements
    dataset_legend = []
    season_legend = []

    suffix = "_mon" if type == "intra" else ""

    indice="Tm" if variable == "T2m" else "Rm"
    list_no_city = data_dict[f'no_{variable}_city']
    list_name_city = data_dict[f'name_{variable}_city']

    for d_idx, dataset in enumerate(datasets):
        sdev = []
        crmsd = []
        ccoef = []
        obs_spatial_clim = []
        obs_annual_clim = []
        ds_spatial_clim = []
        ds_annual_clim = []
        pts_dict = city_ds_pts(dataset)  # Use the first dataset to get points


        # print(f"Processing {dataset} for {season}")
        # Load observational data
        # e.g. Tm_mon_obs_DJF_spa_avg.nc
        # print(f"Loading observation data for {dataset} in {season}")
        obs_clim = Cache.get_data(os.path.join(Data_ind, f"{indice}{suffix}_obs_{dataset}_seasonal_clim.nc"))
        obs_annual_clim = Cache.get_data(os.path.join(Data_ind, f"{indice}{suffix}_obs_{dataset}_annual_clim.nc"))

        ds_clim = Cache.get_data(os.path.join(Data_ind, f"{indice}{suffix}_{dataset}_seasonal_clim.nc"))
        ds_annual_clim = Cache.get_data(os.path.join(Data_ind, f"{indice}{suffix}_{dataset}_annual_clim.nc"))

        for k, name_city in enumerate(list_name_city):
            ds_spatial_point = cut_mlp(dts=cut_points(ds_clim, pts_dict[name_city]), dim_mean=['geo'], var=variable, val_only=True)
            ds_annual_point = cut_mlp(dts=cut_points(ds_annual_clim, pts_dict[name_city]), dim_mean=['geo'], var=variable, val_only=True)
            # Extract observation data for this city (already pre-selected)
            obs_spatial_point = cut_var(obs_clim.sel(no_station=list_no_city[k]), var=variable, val_only=True)
            obs_annual_point = cut_var(obs_annual_clim.sel(no_station=list_no_city[k]), var=variable, val_only=True)

            ds_data = list(ds_annual_point) + list(ds_spatial_point)
            obs_data = list(obs_annual_point) + list(obs_spatial_point)

            ds_spatial_clim.append(np.array(ds_data))
            obs_spatial_clim.append(np.array(obs_data))

        ds_spatial_clim = np.array(ds_spatial_clim)
        obs_spatial_clim = np.array(obs_spatial_clim)
            #? Call the class without init function
            # instance = object.__new__(MetricExtreme)
            # instance.dataset=dataset
            # instance.season=season
            # instance._setup_year()
            # (ds_spatial_clim, _), (obs_spatial_clim, _) = instance._subset_year(ds_spatial_clim, variable, mode='rean'), instance._subset_year(obs_spatial_clim, variable)
        for s_idx, season in enumerate(seasons):
            # Calculation for metrics
            obs_spatial_ss = obs_spatial_clim[:, s_idx]
            ds_spatial_ss = ds_spatial_clim[:, s_idx]

            if normalized:
                std_obs = evaluate_single(obs_spatial_ss, metrics='std')
                # Normalization
                ds_spatial_ss = ds_spatial_ss/std_obs
                obs_spatial_ss = obs_spatial_ss/std_obs

                std_ds = evaluate_single(ds_spatial_ss, metrics='std')
                std_obs = evaluate_single(obs_spatial_ss, metrics='std')

                crmsd_ind = sm.centered_rms_dev(ds_spatial_ss, obs_spatial_ss)
                print("test", std_ds, std_obs)

            else:
                std_ds = evaluate_single(ds_spatial_ss, metrics='std')
                std_obs = evaluate_single(obs_spatial_ss, metrics='std')
                crmsd_ind = np.sqrt(std_ds**2 + std_obs**2 - 2*std_ds*std_obs*ccoef_ind)

            # Calculate centered root mean square deviation (CRMSD) and correlation coefficient (CCoef)
            ccoef_ind = evaluate_compare(obs_spatial_ss,ds_spatial_ss, metrics='R')['R']

        # Add to list
            sdev.append(std_ds)
            crmsd.append(crmsd_ind)
            ccoef.append(ccoef_ind)

        # Turn the metrics list to an array
        sdev = np.array([std_obs]+sdev)
        crmsd = np.array([0]+crmsd)
        ccoef = np.array([1]+ccoef)
        list_label = [""]+list(np.arange(len(seasons))+1)
        print(f"Dataset: {dataset}, Seasons: {seasons}, Standard Deviations: {sdev}, CRMSD: {crmsd}, CCoef: {ccoef}")
        # Save metrics
        if d_idx == 0:
            # Initialize data storage for each dataset
            sm.taylor_diagram(sdev, crmsd, ccoef,
                            markerSize=15,
                            markerColor=list_colors[d_idx],
                            markerLegend='off',
                            markerSymbol=markers[d_idx],
                            markerLabel= list_label,
                            markerLabelColor='m',
                            numberPanels=1 if variable == "T2m" else 2,
                            # Custom the root mean square line
                            tickRMSangle=80.0 if variable == "T2m" else 85.0,
                            colRMS='k',
                            # rmsLabelFormat='0:1f',
                            styleRMS=':',
                            widthRMS=1.5,
                            tickRMS=np.array([0.25, 0.5, 0.75, 1]) if variable == "T2m" else np.array([0.5, 1, 1.5, 2]),
                            titleRMS='off',
                            # Custom the observation points
                            styleOBS='-',
                            colOBS='r',
                            markerobs='o',
                            # widthOBS=1.5,
                            titleOBS="Reference",
                            # Custom the standard deviation line
                            colSTD='k',
                            styleSTD='-.',
                            widthSTD=2.0,
                            # Custom the correlation coefficient curve
                            colCOR='k',
                            styleCOR='--',
                            widthCOR=1.0,
                            axismax = 2 if variable == "T2m" else 3,
                            )
        else:
            sm.taylor_diagram(sdev, crmsd, ccoef,
                            markerSize=15,
                            markerColor=list_colors[d_idx],
                            markerSymbol=markers[d_idx],
                            markerLabelColor='m',
                            markerLegend='off',
                            markerLabel=list_label,
                            overlay='on'
                            )

        ds_legend = mlines.Line2D([], [], color=list_colors[d_idx], marker=markers[d_idx],
                            markersize=10, label=datasets_name[datasets[d_idx]], linestyle='none')
        dataset_legend.append(ds_legend)

    # Create legend elements for seasons
    for s_idx, season in enumerate(seasons, start=1):
        # Create a custom marker using text
        season_leg = plt.Line2D([], [],
                            color='k',
                            marker='$'+str(s_idx)+'$',  # Math text mode
                            markersize=13,
                            label=season.upper(),
                            linestyle='none',
                            markeredgewidth=0.5)
        season_legend.append(season_leg)

    if variable == "T2m":
        cbar_ax = fig.add_axes([0.5, 0.55, 0.4, 0.7])  # Adjust position and size of colorbar axis
        cbar_ax_2 = fig.add_axes([0.65, 0.55, 0.4, 0.7])  # Adjust position and size of colorbar axis
        cbar_ax.clear()
        cbar_ax.set_axis_off()  # Hide the axis
        cbar_ax_2.clear()
        cbar_ax_2.set_axis_off()  # Hide the axis
        # cbar_ax.set_visible(False)

        legend = cbar_ax.legend(handles=dataset_legend,# title = cbar_label,
                                    loc='center', ncol=1,labelspacing=1,  # Vertical spacing
                                    handletextpad=0.5,    # Space between symbol and text
                                    borderpad=1, frameon=False)

        legend_2 = cbar_ax_2.legend(handles=season_legend,# title = cbar_label,
                                    loc='center', ncol=1,labelspacing=1,  # Vertical spacing
                                    handletextpad=0.5,    # Space between symbol and text
                                        borderpad=1, frameon=False)

    # Remove unnecessary ticks
    plt.tick_params(axis='x', which='both', top=False)
    plt.tick_params(axis='y', which='both', right=False)

    # Create custom legend box at right side

    # # Create a grey box around the dataset title
    # bbox = dataset_box.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # rect = mpatches.FancyBboxPatch((bbox.x0-0.05, bbox.y0-0.02),
    #                         bbox.width+0.1, bbox.height+0.04,
    #                         facecolor='lightgrey', alpha=0.7,
    #                         transform=fig.transFigure, zorder=0)
    # fig.patches.append(rect)


    # Save figure
    plt.tight_layout(rect=[0, 0.08, 1, 0.9])  # Adjust layout to make room for titles
    save_path = os.path.join(img_wd, f'{indice}_{type}_taylor_diagram.png')
    plt.savefig(save_path, format="png", dpi=300, bbox_inches='tight')
    print(f"Saved Taylor diagram to {save_path}")
    plt.close()

    # Generate all four Taylor diagrams
    # taylor_plot("t2m", "grid")
    # taylor_plot("t2m", "rean")
    # taylor_plot("pre", "grid")
    # taylor_plot("pre", "rean")

if __name__ == "__main__":

    import concurrent.futures as con_fu
    from joblib import Parallel, delayed


    print("Starting the analysis...")

    for variable in ['R','T2m']:

        # TASK: Create correlation map
        # Parallel processing for each metric
        # with con_fu.ProcessPoolExecutor() as executor:
        #     executor.map(plot_correlation_maps, [variable]*len(metrics_compare), metrics_compare)

        # plot_correlation_maps(variable, 'R')
        # Serial processing for debugging
        for metric in metrics_compare:
            plot_correlation_maps(variable, metric)
        #     sys.exit(0)

        # plot_correlation_maps(variable, 'R')
        # TASK: Create scatter plot for seasonal mean
        # plot_scatter(variable)

        # TASK: Check manually the station
        # check_station(variable)
            # sys.exit(0)

        # TASK: Create Taylor diagram
        # taylor_diagram(variable)
        # # sys.exit(0)
