# Standard library imports
# import PtitPrince as pt
from my_junk import *
from cdo import *
from cf_units import decode_time
from tkinter import ttk
import os
import subprocess
import shlex
import concurrent.futures as con_fu
import time
import math
import datetime as dt
import warnings
import json
from typing import Union, Optional, List, Tuple, Dict, Literal, Callable

# Data manipulation and analysis
import pandas as pd
import numpy as np
import scipy.stats as sst
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.utils import check_random_state
import xarray as xr

# Plotting and visualization
import matplotlib
import matplotlib.style
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# import matplotlib.dates as mdates
# import matplotlib.cm as cm
# import matplotlib.lines as mlines
# import matplotlib.ticker as ticker
# from matplotlib.colors import ListedColormap
# from matplotlib.tri import Triangulation
# from matplotlib.animation import FuncAnimation
# from matplotlib.image import imread
# from matplotlib.patches import Wedge, PathPatch
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
from matplotlib.ticker import MultipleLocator
import seaborn as sns

# Geospatial libraries
import geopandas as gpd
import shapefile as shp
# from shapely.geometry import MultiPolygon
# from cartopy.feature import ShapelyFeature
# from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
# import cartopy.feature as cfeature
import cartopy.crs as ccrs

# ------------------------------------------------------------------ #

import sys
sys.path.append("/home/tamnnm/.conda/envs/tamnnm/lib/python3.9/site-packages/ptitprince")
import PtitPrince as pt
# Other libraries


sns.set_style('ticks')
sns.set_context("paper")

params = {
    'axes.titlesize': 40,
    'axes.labelsize': 10,
    'font.size': 10,
    'font.family': 'cmss10',
    'legend.fontsize': 5,
    'legend.loc': 'upper right',
    'legend.labelspacing': 0.25,
    'xtick.labelsize': 60,
    'ytick.labelsize': 60,
    'lines.linewidth': 2,
    'text.usetex': False,
    # 'figure.autolayout': True,
    'ytick.right': False,
    'xtick.top': False,

    'figure.figsize': [15, 20],  # instead of 4.5, 4.5
    'axes.linewidth': 3,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.minor.size': 1,
    'ytick.minor.size': 1,

    'xtick.major.width': 1,
    'ytick.major.width': 1,
    'xtick.minor.width': 1,
    'ytick.minor.width': 1,

    'xtick.major.pad': 10,
    'ytick.major.pad': 10,

    'xtick.direction': 'in',
    'ytick.direction': 'in',
    
    'axes.unicode_minus': False,
    'figure.constrained_layout.use': True,
    'figure.dpi': 300
}
plt.clf()
matplotlib.rcParams.update(params)

warnings.filterwarnings("ignore")

Data_wd = os.getenv("data")
Code_wd = os.path.join(os.getenv("mcode"), "01.city_list_obs/")
Data_csv = os.path.join(Data_wd, "wrf_data/netcdf/para/csv_file/")
Data_nc = os.path.join(Data_wd, "wrf_data/netcdf/para/")
Data_obs_list = os.path.join(Code_wd, "city/")
Data_obs_pts = os.path.join(Code_wd, "city_pts/")
img_wd = os.getenv("img")

script_path = script_path = os.path.dirname(os.path.abspath(__file__))
json_path = script_path+"/constant.json"

with open(json_path, 'r') as json_file:
    data_dict = json.load(json_file)
    data_dict['dataset'] = ['cera','era','era5','noaa']
with open(json_path, 'w') as json_file:
    json.dump(data_dict, json_file)

rain_csv = data_dict['rain_csv']
temp_csv = data_dict['temp_csv']
temp_tuple = data_dict['temp_tuple']
rain_tuple = data_dict['rain_tuple']
no_T_station = data_dict['no_T2m_city']
no_R_station = data_dict['no_R_city']
name_T_station = data_dict['name_T2m_city']
name_R_station = data_dict['name_R_city']
dataset_list = data_dict['dataset']

map_pro = ccrs.PlateCarree()
shp_path = os.getenv("vnm_sp")
vnmap = shp.Reader(shp_path)
column_df = ['name_station', 'first_year', 'last_year', 'lon', 'lat', 'elev', 'appendix']

file_t = xr.open_dataset(Data_csv+"ensemble/T2m_manndall_kendall_merge.nc")
file_r = xr.open_dataset(Data_csv+"ensemble/R_manndall_kendall_merge.nc")
file_obs_t = pd.read_csv(Data_csv+"single/obs_T2m.csv")
file_obs_r = pd.read_csv(Data_csv+"single/obs_R.csv")
rain_df = pd.read_csv(rain_csv, names=column_df)
temp_df = pd.read_csv(temp_csv, names=column_df)
var = ['slope', 'p']

def point_map(par:Optional[Literal['R','T2m']] = 'R',dataset:Optional[Literal['rean', 'obs']]= 'rean', direction = 'vertical'):
    
    if par == "R":
        var_tuple = rain_tuple
        main_dts = file_r[var] if dataset == 'rean' else file_obs_r
        name_list = name_R_station
        main_df = rain_df
    else:
        var_tuple = list(temp_tuple)
        var_tuple.remove('SU35')
        main_dts = file_t[var] if dataset == 'rean' else file_obs_t
        main_df = temp_df
        name_list = name_T_station
    
    # Direction = verticle
    if direction == "vertical":
        no_row, no_col = 4,4
        figsize=(50, 70)
        
        gs = gridspec.GridSpec(1, 2, width_ratios=[no_row,0.5])
        gs_main = gridspec.GridSpecFromSubplotSpec(no_row, no_col,\
                                                    width_ratios=[1]*(no_col), height_ratios=[1]*no_row,\
                                                    subplot_spec=gs[0],\
                                                    wspace= -0.8, hspace=-0.05)
        gs_island = gridspec.GridSpecFromSubplotSpec(no_row, 1,\
                                                    height_ratios=[1]*no_row,\
                                                    subplot_spec=gs[1],\
                                                    hspace= 0)
        gs.update(wspace = -0.43)
    
    # Direction = horizontal
    else:
        no_row, no_col = 3, math.ceil(len(var_tuple)/3)
        figsize=(50, 80)
    
        gs = gridspec.GridSpec(1, 2, width_ratios=[no_row,0.5])
        gs_main = gridspec.GridSpecFromSubplotSpec(no_row, no_col,\
                                                    width_ratios=[1]*(no_col), height_ratios=[1]*no_row,\
                                                    subplot_spec=gs[0],\
                                                    wspace= -0.4, hspace=-0.05)
        gs_island = gridspec.GridSpecFromSubplotSpec(no_row, 1,\
                                                    height_ratios=[1]*no_row,\
                                                    subplot_spec=gs[1],\
                                                    hspace= 0)
        gs.update(wspace = -0.2)
    
    # Create the figure
    
    fig = plt.figure(figsize=figsize)

    # Create the subplots
    axs = []
    for i in range(no_row):
        row = []
        for j in range(no_col):
            ax = fig.add_subplot(gs_main[i, j], projection=map_pro)
            row.append(ax)
        # append the last subplot of each row
        #? Create equal wspace; if last plot = 1.5x width => wspace = 1.5x
        ax = fig.add_subplot(gs_island[i, 0], projection=map_pro)
        row.append(ax)
        axs.append(row)
    # Flatten the list of subplots
    axs = [ax for row in axs for ax in row]
    # START PLOTTING
    ind_count = 0
    for ax in axs:
        
        ax.axis('off')
        #! Do this for every subplot to avoid blank plot has different dimension
        ax.set_extent([101, 110, 8, 25])
        
        
        ## Clear extra subplots
        #? e.g. 14 ind, 16 suplots => Clear the first 2 subplots in last row
        # get_subplotspec return position in the nested subplot
        if ax.get_subplotspec().rowspan.start == no_row - 1 \
            and ax.get_subplotspec().colspan.start < no_row * no_col - len(var_tuple) \
            and ax.get_subplotspec().get_geometry()[1] != 1:
            continue
        
        #? Condition: Take the last row => Get the abundant position => Check it's the main plot
        
        # plot vietnam and the station point
        # Get all the plotting point in the sahep
        txt_shapes = []
        for vnmapshape in vnmap.shapeRecords():
            listx = []
            listy = []
            # parts contains end index of each shape part
            parts_endidx = vnmapshape.shape.parts.tolist()
            parts_endidx.append(len(vnmapshape.shape.points) - 1)
            for i in range(len(vnmapshape.shape.points)):
                x, y = vnmapshape.shape.points[i]
                if i in parts_endidx:
                    # we reached end of part/start new part
                    txt_shapes.append([listx, listy])
                    listx = [x]
                    listy = [y]
                else:
                    # not end of part
                    listx.append(x)
                    listy.append(y)
        # Plot all the point
        for zone in txt_shapes:
            x, y = zone
            # Plot only the border
            ax.plot(x, y, color="k", markersize=10e-6,
                    linewidth=0.4)
        
        ## PLOT ISLAND
        #? get_geometry() => (rows_in_subplot, cols_in_subplot,start, stop)
        if ax.get_subplotspec().get_geometry()[1] == 1:
            ax.set_extent([110.5, 116, 8, 25])
            ax.text(112, 10, "Spratly Islands", fontsize=35, rotation=45, alpha = 0.75)
            ax.text(111.5, 14.5, "Paracel Islands", fontsize=35, rotation=45, alpha = 0.75)
            ax.text(110.5, 12, "E  a  s  t   S  e  a", fontsize=45, rotation = 90, fontstretch = 'expanded', alpha = 0.75) # ax.text(102,20 , "Laos", color="w", fontsize=16)
            # ax.grid(linewidth=1, color='gray', alpha=0.75,
            #         linestyle='--')
            continue
       
        print(ax.get_subplotspec().get_geometry(), ind_count)
        ## EXTEND OF MAIN PLOT
    
        ## DATASET
        ind = var_tuple[ind_count]
        ind_count += 1
        ind_dts = main_dts.sel(ind=ind) if dataset == 'rean' else main_dts[main_dts['ind']==ind]

        ## TICK PARAMETERS
        # ax.coastlines(color='black', linewidth=1, resolution='10m', alpha=1)
        # ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle='-')
        ax.set_xticks(np.linspace(101, 110, 5), crs=map_pro)
        ax.set_yticks(np.linspace(8, 25, 5), crs=map_pro)
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)

        # Ensure tick labels are visible
        # ax.xlabels_top = False
        # ax.ylabels_right = False
        # ax.tick_params(axis='both', which='major',
        #                labelsize=10, direction='in', length=10)
        # ax.tick_params(axis='both', which='minor',
        #                labelsize=8, direction='in', length=4)
        # ax.xaxis.set_tick_params(labelbottom=False)
        # ax.yaxis.set_tick_params(labelleft=False)
        # ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        # ax.yaxis.set_minor_locator(MultipleLocator(0.5))

        # ax.set_title(ind, fontsize=20, pad=20, loc='left')
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])

        # ax.set_xlabel('Longitude')
        # ax.set_ylabel('Latitude')

            
        # TITLE
        ax.text(0.05, 0.95, ind, fontsize=60, transform=ax.transAxes, ha='left', va='top')
        
        # Creat legend
        # configure color and border of plot
        fig.patch.set_facecolor('none')
        fig.patch.set_alpha(0)
        ax.patch.set_facecolor('none')
        #ax.patch.set(lw=2, ec='k', alpha=0.5) #?Create thin border of axis

        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        
        if dataset == 'rean':
            ratios = [0.25, 0.25, 0.25, 0.25]
            for station in name_list:
                # Use city_list in mcode 01
                position = main_df[main_df['name_station']
                                == station][['lon', 'lat']].values[0]
                # Filter the DataFrame once
                station_dts = ind_dts.sel(name_list=station)

                # Extract the values for 'trend' and 'p' columns
                man_value = station_dts['slope'].values
                p_value = station_dts['p'].values

                colors_markers_edge = [
                    ['#d7191c', 'v', '#d7191c'] if man_value_i < 0 and p_value_i < 0.05 else
                    [None, 'v', '#d7191c'] if man_value_i < 0 and p_value_i >= 0.05 else
                    ['#2c7bb6', '^', '#2c7bb6'] if man_value_i > 0 and p_value_i < 0.05 else
                    [None, '^','#2c7bb6']
                    for man_value_i, p_value_i in zip(man_value, p_value)
                ]
                
                colors = colors_markers_edge[:, 0]
                markers = colors_markers_edge[:, 1]
                edge_colors = colors_markers_edge[:, 2]
                
                # Calculate the angles for the pie chart
                #! Pie chart is not a good idea
                # start_angle = 0
                # for ratio, color in zip(ratios, colors):
                #     end_angle = start_angle + ratio * 360
                #     wedge = Wedge(center=position, r=0.2, theta1=start_angle, theta2=end_angle,
                #                   facecolor=color, edgecolor='black', linewidth=0.2, transform=ccrs.PlateCarree())
                #     ax.add_patch(wedge)
                #     start_angle = end_angle
        if dataset == 'obs':
            for station in name_list:
                position = main_df[main_df['name_station']
                                == station][['lon', 'lat']].values[0]
                # Filter the DataFrame once
                station_dts = ind_dts[ind_dts['name_station']==station]
                
                # Extract the values for 'trend' and 'p' columns
                man_value = station_dts['slope'].values
                p_value = station_dts['p'].values
                
                if man
                if (man_value < 0 and par == 'R') or (man_value > 0 and par == 'T2m'):
                    markers = 'v'
                    edge_colors = '#d7191c'
                    colors = '#d7191c' if p_value < 0.05 else 'w'
                else:
                    markers = '^'
                    edge_colors = '#2c7bb6'
                    colors = '#2c7bb6' if p_value < 0.05 else 'w'
                
                ax.scatter(position[0], position[1], s=150, facecolor=colors, marker=markers, edgecolors=edge_colors, alpha = 0.75, linewidth=2, transform=ccrs.PlateCarree())
    
    # Create legend
    # points_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    # points, labels = [sum(lol, []) for lol in zip(*points_labels)]
    # fig.legend(points, labels)
    
    # print(points, labels)
    
    fig.tight_layout()
    fig.savefig(os.path.join(img_wd,f'{par}_MK_{dataset}_{direction}.jpg'),
                format="jpg", bbox_inches='tight', dpi=200, transparent=True)
    # break
    return

point_map('R','obs')
point_map('T2m','obs')
# point_map('T','obs','horizontal')
def violin(par):
    
    
    params ={
        'xtick.labelsize': 70,
        'ytick.labelsize': 70,
        'axes.labelpad': 10
    }
    matplotlib.rcParams.update(params)
    
    if par == "R":
        var_tuple = rain_tuple
        main_dts = file_r[var]
        obs_main_df = file_obs_r
        # name_list = no_R_station
        no_row, no_col = len(var_tuple)//3,3
        figsize=(70, 80)
    else:
        var_tuple = list(temp_tuple)
        var_tuple.remove('SU35')
        main_dts = file_t[var]
        obs_main_df = file_obs_t
        # name_list = no_T_station
        no_row, no_col = (len(var_tuple)-1)//4+1,4
        figsize=(100, 100)
    fig, axs = plt.subplots(no_row, no_col, figsize=figsize)
    fig.subplots_adjust(hspace = 0.8, wspace = 0.7)
    
    # Nx,Ny=1,1000
    #imgArr = np.tile(np.linspace(0,1,Ny), (Nx,1)).T
    #cmap = sns.color_palette("RdBu", as_cmap=True)
    colors = ['#374b75','#feec8e','#8dc9ba','#dc3f51','#c4c8ce']
    
    box_props = dict(alpha = 1, color='black', linewidth=5)
    box_main = dict(facecolor='black', edgecolor='black', linewidth=5)
    median_propis = dict(color='white', linewidth=5)
    # positions = np.arange(len(dataset_list))
    ind_count = 0
    for ax in axs.flatten():
        if ax.get_subplotspec().rowspan.start == no_row - 1 and ax.get_subplotspec().colspan.start < no_row * no_col - len(var_tuple):
            ax.axis('off')
            ax.set_xlim(-0.75, len(dataset_list)+0.5)
            continue
        ind = var_tuple[ind_count]
        ind_count += 1
        
        # Filter the DataArray
        rean_dts = main_dts.sel(ind=ind)
        # Convert the DataArray to a DataFrame
        rean_df = rean_dts.to_dataframe().reset_index()
        obs_df = obs_main_df[obs_main_df['ind']==ind]
        obs_df['dataset'] = 'obs'
        ind_df = pd.concat([rean_df, obs_df], ignore_index=True)

        # Example plot using seaborn violin plot
        
        sns.despine(offset=0.2, trim=False)
        for i, dataset in enumerate(dataset_list+['obs']):
            data_df = ind_df[ind_df['dataset'] == dataset]
            y = data_df['slope']
            x = i + np.random.uniform(-0.1, 0.3, size=len(y))
            if i == len(dataset_list) -1:
                final_point = x.max()
            edge_colors = ['black' if p < 0.05 else 'grey' for p in data_df['p']]
            ax.boxplot(y, positions=[i], widths=0.1, patch_artist=True, boxprops=box_main,
                        manage_ticks=False,
                        showfliers = False,
                        medianprops=median_propis, whiskerprops=box_props, capprops= box_props)
        sns.violinplot(x='dataset', y='slope', data=ind_df, ax=ax, inner = None, palette = colors, orient = 'v',scale = 'count')
        ax.axvline(final_point+0.2, color='black', linestyle='--', linewidth=2)
        ax.set_xlabel('') #Jitter the points
        ax.set_ylabel('')
            
        # create a numpy image to use as a gradient

        #? If you want to create the color gradient for these
        #? https://i.sstatic.net/KDf0l.png
        #violins = sns.violinplot(x='dataset', y='slope', data=ind_df, ax=ax, split=True)
        # for violin in violins['bodies']:
        #     path = Path(violin.get_paths()[0].vertices)
        #     patch = PathPatch(path, facecolor='none', edgecolor='none')
        #     ax.add_patch(patch)
        #     img = ax.imshow(imgArr, origin="lower", extent=[xmin,xmax,ymin,ymax], aspect="auto",
        #             cmap=cmap,
        #             clip_path=patch)
        
        ax.set_xlim(-0.75, len(dataset_list)+0.5)
        ax.text(0.02, 0.97, ind, transform=ax.transAxes,
                fontsize=80, va='top', ha='left')
        # Clear any remaining axes
    fig.tight_layout(pad = 0.2)
    fig.savefig(os.path.join(img_wd,f'violin_{par}.jpg'),
                format="jpg", bbox_inches='tight', dpi=200, transparent=True)
    return

# violin('R')
# violin('T2m')
