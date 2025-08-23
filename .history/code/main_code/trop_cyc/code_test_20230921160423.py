# ----------------------------------------- Import module ---------------------------------------- #
from ast import Continue, Pass
from re import I
from selectors import EpollSelector
from tkinter import ttk
from cf_units import decode_time
from matplotlib.font_manager import ttfFontProperty
# from matplotlib.lines import _LineStyle
import pandas as pd
import matplotlib
import numpy as np
import os
from netCDF4 import Dataset
import scipy.stats as sst
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.utils import check_random_state
import matplotlib.dates as mdates
import datetime as dt
import csv
import xarray as xr
from matplotlib.image import imread
import cfgrib
import itertools
from wrf import *
from xtci.entropy_deficit import entropy_deficit as ent_def
from xtci.potential_intensity_tcpypi import potential_intensity as pi
from xtci.wind_shear import wind_shear as w_shear
from xtci.absolute_vorticity import absolute_vorticity as abs_vor
from my_junk.cut_dataset import *
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
from scipy.ndimage import gaussian_filter
from matplotlib.animation import FuncAnimation
import matplotlib.lines as mlines
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.cm import ScalarMappable
import geopandas as gpd
from shapely.geometry import MultiPolygon

params = {
    'axes.titlesize': 25,
    'axes.labelsize': 0,
    'font.size': 20,
    'font.family': 'serif',
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

    'figure.figsize': [12, 10],  # instead of 4.5, 4.5
    'axes.linewidth': 1.5,
    'xtick.major.size': 15,
    'ytick.major.size': 15,
    'xtick.minor.size': 5,
    'ytick.minor.size': 5,

    'xtick.major.width': 5,
    'ytick.major.width': 5,
    'xtick.minor.width': 3,
    'ytick.minor.width': 3,

    'xtick.major.pad': 10,
    'ytick.major.pad': 10,
    # 'xtick.minor.pad': 14,
    # 'ytick.minor.pad': 14,

    'xtick.direction': 'inout',
    'ytick.direction': 'inout',
}
plt.clf()
matplotlib.rcParams.update(params)

# -------------------- set up background border -------------------- #


def set_up_plot(dlat, ulat, dlon, ulon, pad):
    map_pro = ccrs.PlateCarree()
    fig = plt.figure()
    ax = plt.subplot(111, projection=map_pro)
    ax.set_xticks(np.arange(dlon, ulon, pad), crs=map_pro)
    ax.set_yticks(np.arange(dlat, ulat, pad), crs=map_pro)
    lon_formatter = LongitudeFormatter(zero_direction_label=False)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    # Tick for axis
    ax.tick_params(axis='both', which="major", labelsize=15)
    # change fontsize of tick when have major
    # """
    ax.set_xticks(ax.get_xticks())
    ax.set_yticks(ax.get_yticks())
    # Draw border, coasral line,...
    ax.coastlines(resolution='10m')
    # , linewidth=borders)#resolution='10m')
    ax.add_feature(cfeature.BORDERS.with_scale('10m'))
    # axins.add_feature(cfeature.LAND,)#, facecolor=cfeature.COLORS["land_alt1"])
    # axins.add_feature(cfeature.OCEAN,facecolor=cfeature.COLORS['water'])
    return fig, ax

# ---------------- function the geopotential height --------------- #

def func_hgt(data_test, threshold, range, convert=None):
    convert_fac = 9.80616
    # print(f"You have choose the range is {range}")
    print(type(data_test))
    if "DataArray" in str(type(data_test)):
        data_test = data_test
    else:
        data_test = data_test.to_array()
    break_factor = 0
    while range <= 50 and break_factor <= 0:
        # Only accept values with Dataarray, so be careful
        try:
            if convert == None:
                data_plot = data_test.where(
                    abs(data_test.values - threshold) <= float(range))
            else:
                data_plot = data_test.where(
                    abs(data_test.values/convert_fac-threshold) <= float(range))/convert_fac
                # print(data_test.values/convert_fact)
                print(abs(data_test.values-convert_fac*threshold))
            break_factor = 1
            # print("Success")
        except Exception as e:
            print(e)
            range = range+1
    return data_plot


# ---------------------- file in grib_original --------------------- #
Data_path_grib = "/work/users/tamnnm/Data/wrf_data/gribfile/era5_6h_1940_2023/"
filename = "geopotential_1964_b_even.grib"
Dataset = xr.open_dataset(f'{Data_path_grib+filename}',
                          engine="cfgrib")

# ----------------------- file in wrf_output ----------------------- #
"""
#Data_path_wrf = "/work/users/tamnnm/wrf/WRF/test/RUN_ERA5_196411_9/"
# for filename in (os.listdir(f'{Data_path}/wrfout*')):
filename = "wrfout_d02_1964-11-29_00:00:00"
dataset = Dataset(Data_path+filename)
# Variable
slp = getvar(dataset, "slp")
p = getvar(dataset, "z")
ua = getvar(dataset, "ua", units="m s-1")
va = getvar(dataset, "va", units="m s-1")
wspd = getvar(dataset, "wspd_wdir", units="m s-1")[0, :]
wspd10 = getvar(dataset, "wspd_wdir10", units="m s-1")[0, :]
temp = getvar(dataset, "temp", units="degC")
T2 = getvar(dataset, "T2")  # K
rh = getvar(dataset, "rh")
rh2 = getvar(dataset, "rh2")
eth = getvar(dataset, "eth", units="degC")
ctt = getvar(dataset, "ctt", units="degC")
dbz = getvar(dataset, "dbz")
z = getvar(dataset, "z")
avo = getvar(dataset, "avo")
landsea = getvar(dataset, "LANDMASK")
sst = getvar(dataset, "SST")  # K

# Level variable
# hgt_500 = interplevel(z, p, 500)
u_200 = interplevel(ua, p, 200)
v_200 = interplevel(va, p, 200)
u_850 = interplevel(ua, p, 850)
v_850 = interplevel(va, p, 850)
"""

# ------------------ calculate geopotential height ----------------- #

p_test = [200, 500, 850]
z_threshold = [12400, 5850, 1520]
# """
for i, z_p in enumerate(p_test):
    break_factor = 0
    level_co = cut_level(Dataset, level=z_p)
    latlon_co = cut_co(Dataset, dlat=-5, ulat=40, dlon=80, ulon=150)
    # Extract level from wrfoutput
    # data_test = interplevel(z, p, z_p)

    # Extracct level from gribfile
    for day in np.arange(1, 2):
        time_co = cut_time(Dataset, yr=1964, mon=11, day=day)
        data_test = cut_mlp(Dataset, time_data=time_co,
                            geo_data=latlon_co, level_data=level_co, dim_mean=['time', 'level'], dim_plot=['geo'])  # even though there is one level butdo this so that the data array is 2 dimension
        # print(data_test.values)
        # Computation can only be executed with dataarray so change dataset->dataarray
        data_hgt = func_hgt(data_test=data_test, threshold=z_threshold[i],
                            range=5, convert="yes")
        (fig, ax) = set_up_plot(-5, 40, 80, 150, 5)
        for i in np.arange(1, 2):
            data_plot = data_hgt
        # print(data_hgt.values)
            pl = data_plot.plot.contour(
                ax=ax, colors="r", linewidths=2, levels=1)
        # Set
            ax.clabel(pl, inline=True, fontsize=10)
        # If the variable dimension is not important for the plot, then you can use the ... placeholder.
            ax.set_title(f'{z_p}-11/1964', pad=25)
            plt.savefig(os.path.abspath(
                rf'/work/users/tamnnm/code/main_code/trop_cyc/img/era5_{z_p}hPa_{day}_11.jpg'), format="jpg", dpi=300)
            plt.close()


# ----------------------- calculate humidity ----------------------- #

Caculate