# ----------------------- Import plot module ----------------------- #
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
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.cm import ScalarMappable
import geopandas as gpd
from shapely.geometry import MultiPolygon
import pandas as pd
import matplotlib
import numpy as np
import os

# -------------------- set up background border -------------------- #

def ini_plot(dlat, ulat, dlon, ulon, pad, figsize = [12, 8], label = False, label_size = 15, ax_num=111,title="", coastline = True, border = True, grid = False, land = False, ocean = False):
    params = {
    'axes.titlesize': 25,
    'axes.labelsize': 0,
    'font.size': 20,
    'font.family': 'monospace',
    'font.monospace':['Lucida Console','Courier New','Consolas','Source Code Pro'],
    # 'font.family': 'sans-serif',
    'font.sans-serif':['Myriad Pro','Calibri','DejaVu Sans','Lucida Grande', 'Verdana'],
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

