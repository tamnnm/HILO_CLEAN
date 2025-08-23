# ----------------------- Import plot module ----------------------- #
import time
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize, from_levels_and_colors
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.axes as maxes
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.dates as mdates
from typing import Optional, List
from my_junk import *
from geopy.distance import distance as geodistance
from matplotlib.ticker import FormatStrFormatter
import geopandas as gpd
from shapely.geometry import Point
import xarray as xr

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
# import metpy.calc as mpcalc
# from metpy.units import units
import shapefile as shp
import matplotlib
from scipy.ndimage import gaussian_filter
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.lines as mlines
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.cm import ScalarMappable
import geopandas as gpd
from shapely.geometry import MultiPolygon
import pandas as pd
import matplotlib
import numpy as np
import os
from matplotlib import font_manager

# -------------------- set up background border -------------------- #


def ini_plot(dlat, ulat, dlon, ulon, pad, figsize = [12, 8], label = False, label_size = 10, ax_num=111,title="", coastline = True, border = True, grid = False, land = False, ocean = False):
    params = {
    'axes.titlesize': 25,
    'axes.labelsize': 0,
    'font.size': 20,
    'font.family': 'Source Code Pro',
    # 'font.monospace':['Lucida Console','Courier New','Consolas','Source Code Pro'],
    # 'font.family': 'serif',
    # 'font.sans-serif':['Hershey'],
    'legend.fontsize': 20,
    'legend.loc': 'upper right',
    'legend.labelspacing': 0.25,
    # 'xtick.labelsize': 20,
    # 'ytick.labelsize': 20,
    'lines.linewidth': 3,
    'text.usetex': False,
    # 'figure.autolayout': True,
    'ytick.right': True,
    'xtick.top': True,

    'figure.figsize': figsize,  # instead of 4.5, 4.5
    'axes.linewidth': 1.5,
    'xtick.major.size': 10,
    'ytick.major.size': 10,
    'xtick.minor.size': 5,
    'ytick.minor.size': 5,

    'xtick.major.width': 5,
    'ytick.major.width': 5,
    'xtick.minor.width': 3,
    'ytick.minor.width': 3,

    'xtick.major.pad': 3,
    'ytick.major.pad': 3,
    # 'xtick.minor.pad': 14,
    # 'ytick.minor.pad': 14,

    'xtick.direction': 'out',
    'ytick.direction': 'out',
    }
    plt.clf()
    matplotlib.rcParams.update(params)
    
    map_pro = ccrs.PlateCarree()
    fig = plt.figure()
    ax = plt.subplot(ax_num, projection=map_pro)
    
    # Set the extent (bounding box) for the map
    ax.set_extent([dlon, ulon, dlat, ulat], crs=map_pro)
    ax.set_xticks(np.arange(dlon, ulon + pad, pad), crs=map_pro)
    ax.set_yticks(np.arange(dlat, ulat + pad, pad), crs=map_pro)
    
    lon_formatter = LongitudeFormatter(zero_direction_label=False)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    
        
    # Tick for axis
    ax.tick_params(axis='both', bottom=False, left=False, top=False, right=False, labelsize=label_size)
    # change fontsize of tick when have major
    
    ax.set_xticks(ax.get_xticks())
    ax.set_yticks(ax.get_yticks())
    
    if label:
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
    
    # Add grid lines
    if grid: ax.grid(True, which='both', linestyle='--', linewidth=1, color = 'black', zorder=2)
    
    # Add title
    ax.set_title(title, pad = 30)
    
    # Draw border, coasral line,...
    if coastline: ax.coastlines(resolution='10m',zorder=3)
    if border: ax.add_feature(cfeature.BORDERS.with_scale('10m'),zorder=3)
    if land: ax.add_feature(cfeature.LAND,)#, facecolor=cfeature.COLORS["land_alt1"])
    if ocean: ax.add_feature(cfeature.OCEAN,facecolor=cfeature.COLORS['water'])

    return fig, ax

def make_cbar(ax, cmap, sm=None, norm=None, ticks=[], label=[], custom_cbar=True):

    divider = make_axes_locatable(ax)
    ax.set_aspect('equal')
    # Extract the projection from the existing axis
    # projection = ax.projection if hasattr(ax, 'projection') else None
    if sm is None:
        if norm is not None:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        else:
            sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([])

    if custom_cbar == True:
        cax = divider.append_axes(
            "bottom", size='5%', pad=0.6, axes_class=maxes.Axes)
        cbar = plt.colorbar(
            sm, ax=ax, cax=cax, orientation='horizontal', ticks=ticks, drawedges=True)

        cbar.ax.set_xticks([tick + 0.5 for tick in ticks])
        cbar.ax.set_xticklabels(label)

        cbar.outline.set_edgecolor('white')
        cbar.outline.set_linewidth(2.5)

        cbar.dividers.set_color('white')
        cbar.dividers.set_linewidth(2.5)
        cbar.ax.tick_params(size=0,  labelsize=8, pad=-10)
        cbar.ax.xaxis.set_ticks_position('bottom')
    else:
        cax = divider.append_axes(
            "bottom", size='5%', pad=0.6, axes_class=maxes.Axes)

        if len(ticks) == 0:
            cbar = plt.colorbar(sm, ax=ax, cax=cax, orientation='horizontal')
        else:
            cbar = plt.colorbar(
                sm, ax=ax, cax=cax, orientation='horizontal', format='%.3f', ticks=ticks)

        if len(label) != 0:
            cbar.set_ticklabels(label)

        cbar.ax.tick_params(size=5,  labelsize=8)
        cbar.ax.xaxis.set_ticks_position('bottom')
