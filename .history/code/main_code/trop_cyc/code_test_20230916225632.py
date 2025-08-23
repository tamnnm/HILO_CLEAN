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
    'axes.labelsize': 25,
    'font.size': 20,
    'font.family': 'serif',
    'legend.fontsize': 20,
    'legend.loc': 'upper right',
    'legend.labelspacing': 0.25,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
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
    'ytick.major.pad': 12,
    # 'xtick.minor.pad': 14,
    # 'ytick.minor.pad': 14,

    'xtick.direction': 'inout',
    'ytick.direction': 'inout',
}
plt.clf()
matplotlib.rcParams.update(params)

def func_hgt(data_test,range):
    print(f"You have choose the range is {range}")
    while range <= 5 and break_factor <= 0:
        try:
            dset_plot = data_test.where(abs(data_test.values - z_p) <= range)
            break_factor = 1
        except:
            range = range+1
    return dset_plot


# ---------------------- file in grib_original --------------------- #
Data_path_grib =  "/work/users/tamnnm/Data/wrf_data/gribfile/era5_6h_1940_2023/"
filename="geopotential_1964_b_even.grib"
Dataset= xr.open_dataset(f'{Data_path_grib+filename}',
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


#
p_test = [200, 500, 850]
z_threshold = [100, 5850, 5880]
# """
for i, z_p in enumerate(p_test):
    range = 5
    break_factor = 0

    # Extract level from wrfoutput
    #data_test = interplevel(z, p, z_p)

    #Extracct level from gribfile
    for day in np.arange(1,31):
        time_co=cut_time(datayr=1964,mon=11,day=day)

        data_plot=func_hgt(data_test,range=5)
        map_pro = ccrs.PlateCarree()
        fig = plt.figure()
        ax = plt.subplot(111, projection=map_pro)
        # for i in range(0, 31):
        pl = data_plot.plot.contour(
            ax=ax, colors="red", linewidths=2, levels=1)
        ax.set_title(f'{z_p}-11/1964', pad=25)
        plt.savefig(os.path.abspath(
            rf'/work/users/tamnnm/code/main_code/trop_cyc/img/test_{z_threshold[i]}wrf.jpg'), format="jpg", dpi=300)
# """
