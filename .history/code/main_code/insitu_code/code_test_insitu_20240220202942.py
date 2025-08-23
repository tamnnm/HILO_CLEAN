

# ----------------------------------------- Import module ---------------------------------------- #
#region
from tkinter import ttk
from cf_units import decode_time
# from matplotlib.lines import _LineStyle
import pandas as pd
import matplotlib
import numpy as np
import os
import scipy.stats as sst
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.utils import check_random_state
import matplotlib.dates as mdates
import datetime as dt
import xarray as xr
from matplotlib.image import imread
import cfgrib
import itertools
from cdo import *
cdo = Cdo()
#This prohibits that existing files are created a second time
cdo.forceOutput = False

#endregion
# ----------------------- Import plot module ----------------------- #
#region
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
#endregion

# --------------------- Data path and constant --------------------- #
#region
Data_wd = "/data/projects/REMOSAT/tamnnm/"
Data_grid = Data_wd+"wrf_data/gribfile/synop_file/"
Data_nc  = Data_wd+"wrf_data/netcdf/para"
Data_2019 = Data_wd+"dat_daily_1961_2019/"
suffix_2019 = '_1961_2019.txt'
os.chdir(Data_nc)
#Data_2023 = Data_wd+"dat_daily_1961_2021"
param_no = [165, 166, 167, 168]
param = {'non_extreme': {165:'uwnd',166:'vwnd'},
         'extreme':{'min':'Tn','max':'Tm',288:'tp'}
         }
#endregion

# ------------------------- optional choice ------------------------ #
option = "else"
# "time" = plot timeseries
# "indices" = calculate the indices


# ------------------------ reanlysis indices ----------------------- #
def check_exist(name):
    return True if os.path.exists(name) else False


def gen_base(var,start,end,dataset,percentile:list = None):
    #create base file for wet days
    fol=f'para_{var}/'
    ifile=f'{dataset}_{var}.nc'
    ofile='base_'+ifile
    if os.path.exist(ofile):
        if var=="tp":
            cdo.selyear('start+'/'+end',input=f'-mul {ifile} -gec,1 {ifile}',output=ofile,options='L')
            cdo.setrtomiss('0','0.999',input="-selyear,"+start+'/'+end+' '+ifile,output=fol+ofile,options='L')
        else:
            cdo.selyear('start+'/'+end',input=ifile,output=ofile,options='L')
        if var in param['extreme'].keys():
            if percentile:
                ibfile=fol+ofile
                for rp in percentile:
                    pfile=fol+f'p{rp}_{ofile}'
                    if not check_exist(pfile): cdo.timpctl(str(rp),input=f'{ibfile} -timin {ibfile} -timax{ibfile}',output=pfile)
    return

class precip_ind:
    def __init__(self):
        self.fol = "para_228"
        self.ifile = None
    def __call__(self,ind,dataset):
        self.ind = ind
        self.dataset = dataset
        self.ifile = f'{dataset}_228.nc'
        self.ofile = f'{ind}_{dataset}.nc'
        if ind == "cdd":
            self.gen_cdd()
    def gen_cdd(self):
        if not check_exist(self.ofile):
            cdo.yearmax(input=f'-consects -lc,1 {self.ifile}',output=self.ofile)
    def gen_cwd(self):
        if not check_exist(self.ofile):
            cdo.yearmax(input=f'-consects -gec,1 {self.ifile}',output=self.ofile)
    def gen_r10mm(self):
        if not check_exist(self.ofile):
            cdo.yearsum(input=f'-gec,10 {self.ifile}',output=self.ofile)
    def gen_r20mm(self):
        if not check_exist(self.ofile):
            cdo.yearsum(input=f'-gec,20 {self.ifile}',output=self.ofile)
    def gen_r50mm(self):
        if not check_exist(self.ofile):
            cdo.yearsum(input=f'-gec,50 {self.ifile}',output=self.ofile)
    def gen_r1mm(self):
        if not check_exist(self.ofile):
            cdo.yearsum(input=f'-gec,1 {self.ifile}',output=self.ofile)
    def gen_rx1day(self):
        if not check_exist(self.ofile):
            cdo.monmax(input=f'{self.ifile}',output=self.ofile)
    def gen_rx5day(self):
        if not check_exist(self.ofile):
            cdo.monmax(input=f'-runsum,5 {self.ifile}',output=self.ofile)
    def gen_sdii(self):
        if not check_exist(self.ofile):
            cdo.yearmean(input=f'{self.ifile}',output=self.ofile)
    def gen_prcptot(self):
        if not check_exist(self.ofile):
            cdo.yearsum(input=f'-gec,1 {self.ifile}',output=self.ofile)
    def gen_r95p(self):
        if not check_exist(self.ofile):
            cdo.timmin(input=f'{self.ifile}',output=self.ofile)










# ---------------------- Present station data ---------------------- #
txt_file = Data_2019+"T2m_HANOI"+suffix_2019
with open(txt_file, 'r') as f:
    station = f.read()
    # Split file in to a list a row
    lines = station.split("\n")
    no_lines = len(lines)
    no_years = round((no_lines-2)/32)
    city_name = lines[0][:]
    lat_city = float(lines[1][1])
    lon_city = float(lines[1][0])
    f.close()

# Data structure:
# Line%32=0 is the year
# columns: 12 months - rows: 31 days
# Nan value heres is -99.99

# ---- transform line into list and cut off the 2 beginning line --- #
rdata_row = []
rdata_clean = []  # line without the day number, only the data
rdata_flat = []  # all data is flattened to put into dataframe

for i, line in enumerate(lines[2:]):
    temp_line = [float(x) for x in line.split()]
    rdata_row.append(temp_line)
    if i % 32 != 0:
        rdata_clean.append(temp_line[1:])
        del temp_line
    else:
        del temp_line
        continue

rdata_flat = list(itertools.chain(*rdata_clean))
print(len(rdata_flat))

# ------------- format the date -> only show the years ------------- #

year_start = int(rdata_row[0][0])
year_end = year_start+no_years-1
date_start = f'{year_start}-01-01'
date_end = f'{year_end}-12-31'

# Method 1
# start = dt.datetime.strptime(f'{year_start}-01-01', "%Y-%m-%d")
# end = dt.datetime.strptime(f'{year_end}-12-31', "%Y-%m-%d")
# date_generated = [
#    start + dt.datetime.timedelta(days=x) for x in range(0, (end-start).days)]

# Method 2
date_generated = pd.date_range(  # pd.Series(pd.date_range(
    start=date_start, end=date_end, freq='D')
print(date_generated)

# You can also use pd.date_range then put it in a dataframe


# --------------------- Call for array and list -------------------- #
# 228: Tp
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
# Numpy Array: If you just convert the time to dataframe, it cannot be offseprint(ds)

# ------------------ Find all the 4 closest points ----------------- #
with open(Data_wd+"euler_pts.txt", 'r') as f:
    station = f.readlines()
    lines = pd.DataFrame(station)
    f.close()

# --------- combine the grid and insitu data into dataframe --------- #
gdata_all = []  # all gridded data
gdata_avg = []  # average of gridded data
for i, line in enumerate(lines):
    if city_name in line:
        euler_pts = line.strip().split(',')[1:]
        for i in np.range(0, len(euler_pts), 2):
            lat_pts = euler_pts[i]
            lon_pts = euler_pts[i+1]
            gdata_val = ds.sel(longitude=lon_pts, latitude=lat_pts,
                               time=slice(date_start, date_end))
            gdata_daily = gdata_val["t2m"].resample(time="1D").mean()
            # resample will interpolate anymissing date
            gdata_all.append(gdata_all)
        gdata_avg = np.mean(gdata_all, axis=0)
        # axis 0: verticle
        # axis 1: horizontal
        if len(gdata_avg) == len(rdata_flat):
            all_data = pd.DataFrame(
                rdata_flat+gdata_avg).replace("-99.99", np.NaN)
            # transpose so that each nested list become a column
            all_data.T
            # get the name for the column
            all_data.columns = ['rdata', 'gdata']
            # set the date become the index
            all_data.index = date_generated
            print(len(all_data))
            # put the obs data into dataframe -> dataarray to group by time
        else:
            print("The dimension are not equal")

# ----------------------- Calculation or plot ---------------------- #
        if option == "time":
            formatter = mdates.DateFormatter("%Y")  # formatter of the date
            locator = mdates.YearLocator()  # where to put the labels
            fig = plt.figure(figsize=(20, 5))
            ax = plt.gca()  # calling the formatter for the x-axis
            # calling the locator for the x-axis
            ax.xaxis.set_major_formatter(formatter)
            ax.xaxis.set_major_locator(locator)
            all_data.reset_index().plot(
                x='date_generated', y=['rdata', 'gdata'])
            # reset_index so that the date column being a real column not just index
            # fig.autofmt_xdate() # optional if you want to tilt the date labels - just try it
            plt.tight_layout()
            plt.show()
            plt.savefig(Data_wd+'/image/test.jpg', format="jpg")
        elif option == "indices":
            # ------------------------- extreme indices ------------------------ #
            # ref: https://www.frontiersin.org/articles/10.3389/fenvs.2022.921659/full
            # precipitation
            prcp = sdii = rx5 = rx1 = r99p = r95p = r50p = r20mm = r10mm = r11mm = cwd = 0
            # temperature
            diso = txx = tnx = txn = tnn = sud = tr20 = tn10p = tx10p = tn90p = tx90p = wsdi = csdi = gsl = dtr = 0
        else:
            print("Im testing")

    else:
        print("Retrying.....")
        Continue

# ------------------------- test plot here ------------------------- #
# fig,axs=plt.subplot(subplot_kw=dict(projection=ccrs.PlateCarree()))
# map_pro = ccrs.PlateCarree()
# fig = plt.figure()
# ax = plt.subplot(111, projection=map_pro)

# print(time_period)
# plt.savefig
