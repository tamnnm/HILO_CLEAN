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

Data_path = "/work/users/tamnnm/wrf/WRF/test/RUN_ERA5_196411_9/"
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
ht_500 = interplevel(z, p, 500)
u_200 = interplevel(ua, p, 200)
v_200 = interplevel(va, p, 200)
u_850 = interplevel(ua, p, 850)
v_850 = interplevel(va, p, 850)

p_test=[200,500,850]
time_co=cut_time(z,1964,11)
data_test=cut_mlp(z,time_co)
range=5
while range<=5 and break_factor<=0:
                try:
                    dset_plot=dset_test.where(abs(dset_test.values - z)<=range)
                    base_plot=dset_base.where(abs(dset_base.values - z)<=range+3)
                    break_factor=1
                except:
                    range=range+1

