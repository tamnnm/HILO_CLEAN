

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
import pandas as pd
import os
import scipy.stats as sst
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pandas import read_csv
import datetime
import csv
import xarray as xr
from matplotlib.image import imread
import numpy as np
import pandas as pd
import shapefile as shp
import cfgrib
import itertools


# ----------------------- Import plot module ----------------------- #
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
# import metpy.calc as mpcalc
# from metpy.units import units
import numpy as np
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

# --------------------- Data path and constant --------------------- #
Data_wd = "/work/users/tamnnm/Data/"
Data_grid = Data_wd+"wrf_data/gribfile/synop_file/"
Data_2019 = Data_wd+"dat_daily_1961_2019/"
suffix_2019 = '_1961_2019.txt'
Data_2023 = Data_wd+"dat_daily_1961_2021"
param_no = [89, 165, 166, 167, 168]

# --------------------- Call for array and list -------------------- #
# 4 points closest
distance = []
arr_pts_val = []


# 89: Total water column
# 165: u wind
# 166: v wind
# 167: 2m temperature
# 168: 2m dewpoint temperature

# -------------------- Open grib and netcdf file ------------------- #
# for file in os.listdir(Data_grid)
# for para in param_no:
#    for file in os.listdir(Data_grid):
ds = xr.open_dataset(f'{Data_grid+"era5_167.grib"}',
                     engine="cfgrib")
# If you want to add more hours,you should create a new time dimension
reference_date = '1940-01-01 07:00:00.0'
ds['time'] = pd.date_range(
    start=reference_date, periods=ds.sizes['time'], freq="2H")

# Posible error
# Allocate not enough: your time range is large, it cannot convert
# Numpy Array: If you just convert the time to dataframe, it cannot be offfseprint(ds)

try:
    lat_all = ds['lat']  # .values
    lon_all = ds['lon']  # .values
except:
    lat_all = ds['latitude']  # .values
    lon_all = ds['longitude']  # .values


# ------------------ Find all the 4 closest points ----------------- #
lat_target = 21.017
lon_target = 105.800

for lat_test in lat_all:
    for lon_test in lon_all:
        distance_test = np.sqrt((lat_test - lat_target)
                                ** 2 + (lon_test - lon_target)**2)
        distance.append([distance_test, lat_test, lon_test])

# -------------- Find the indices of the closest point ------------- #
closest_pts = np.argsort(np.array(distance), axis=0)[:4]
for idx in closest_pts[:, 1]:
    lat_pts = distance[idx][1]
    lon_pts = distance[idx][2]
    test_vl = ds.sel(longitude=lon_pts, latitude=lat_pts,
                     time="1961-01-01")['t2m'].values

# ---------------------- Present station data ---------------------- #
txt_file = Data_2019+"T2m_HANOI"+suffix_2019
with open(txt_file, 'r') as f:
    station = f.read()
    # Split file in to a list a row
    lines = station.split("\n")
    no_lines = len(lines)
    no_years = round((no_lines-2)/32)
    f.close()

# ---- transform line into list and cut off the 2 beginning line --- #
data_row = []
data_row_clean = []
data_row_flat=[]
for i, line in enumerate(lines[2:]):
    temp_line = int(line.split())
    data_row.append(temp_line)
    if i % 32 != 0:
        data_row_clean.append(temp_line)
    else:
        continue

data_row_flat=itertools.chain
# ------------- format the date -> only show the years ------------- #
year_start = data_row[1][0]
year_end = year_start + no_years

start = datetime.strptime(f'{"01-01-"+year_start}', "%d-%m-%Y")
end = datetime.strptime(f'{"31-12-"+year_end}', "%d-%m-%Y")
date_generated = [
    start + datetime.timedelta(days=x) for x in range(0, (end-start).days)]

# Data structure:
# Line%32=0 is the year
# columns: 12 months - rows: 31 days
# Nan value heres is -99.99

formatter = mdates.DateFormatter("%Y")  # formatter of the date
locator = mdates.YearLocator()  # where to put the labels

fig = plt.figure(figsize=(20, 5))
ax = plt.gca()
ax.xaxis.set_major_formatter(formatter)  # calling the formatter for the x-axis
ax.xaxis.set_major_locator(locator)  # calling the locator for the x-axis
plt.plot(date_generated, Y)
# fig.autofmt_xdate() # optional if you want to tilt the date labels - just try it
plt.tight_layout()
plt.show()


# ------------------------- test plot here ------------------------- #
# fig,axs=plt.subplot(subplot_kw=dict(projection=ccrs.PlateCarree()))
# map_pro = ccrs.PlateCarree()
# fig = plt.figure()
# ax = plt.subplot(111, projection=map_pro)

# print(time_period)
# plt.savefig
for i in range(13):
    plt.plot(i, data_row)
plt.savefig(Data_wd+'/image/test.jpg', format="jpg")


# REMINDER of file structure:
# 1. Name
# 2. Metadata
# 3+31*n: number of year
# Rest: Month


# for row in rows:
#    station_arr = np.array([row.split() for row in rows])

# Split the row needs only .split()
# print(rows[1].split())
"""
for pts in closest_pts:
    lat_close = lat_all[closest_pts]
    lon_close = lon_all[closest_pts]
    try:
        pts_val = ds.sel(lat=lat_close, lon=lon_close)
    except:
        pts_val = ds.sel(latitude=lat_close, longitude=lon_close)

    arr_pts_val = np.append(arr_pts_val, pts_val)

closet_val = np.mean(arr_pts_val, axis=0)
print(closet_val)

# plt.plot(ds.)
"""
