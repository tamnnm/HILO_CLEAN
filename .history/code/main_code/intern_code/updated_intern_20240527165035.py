# ----------------------------------------- Import module ---------------------------------------- #
from ast import Continue, Pass
from re import I
from time import process_time
from selectors import EpollSelector
from tkinter import ttk
from cf_units import decode_time
from matplotlib.font_manager import ttfFontProperty
# from matplotlib.lines import _LineStyle
from pathlib import Path
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
from xtci.entropy_deficit import entropy_deficit
from xtci.potential_intensity_tcpypi import potential_intensity
from xtci.wind_shear import wind_shear
from xtci.absolute_vorticity import absolute_vorticity
from my_junk import *
import gc  # garbage collector
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
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.lines as mlines
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.cm import ScalarMappable
import geopandas as gpd
from shapely.geometry import MultiPolygon

plt.clf()


params = {
    'figure.figsize': [15, 10],  # instead of 4.5, 4.5
    'axes.titlesize': 15,
    'axes.labelsize': 15,
    'axes.linewidth': 0.5,
    'font.size': 20,
    'font.family': 'monospace',
    #    'font.monospace': 'Alma Mono',
    'legend.fontsize': 15,
    'legend.loc': 'upper right',
    'legend.labelspacing': 0.25,
    # 'xtick.labelsize': 20,
    # 'ytick.labelsize': 20,
    'lines.linewidth': 3,
    'text.usetex': False,
    # 'figure.autolayout': True,
    'ytick.right': False,
    'xtick.top': False,

    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.minor.size': 5,
    'ytick.minor.size': 5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,

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

    'grid.linestyle': '-',         # solid
    'grid.linewidth': 1.5,        # in points
    'grid.alpha':     1,        # transparency, between 0.0 and 1.0
}
# plt.style.use('fivethirtyeight')
matplotlib.rcParams.update(params)


# ------------------------ Constant variable ----------------------- #
# region
Data_path = "/work/users/tamnnm/Data/pap25_QA_1945/"
Data_path_grib = "/work/users/tamnnm/Data/wrf_data/gribfile/era5_6h_1940_2023/"
Data_path_grib_synop = "/work/users/tamnnm/Data/wrf_data/netcdf/era5_synop_1940_2023/"
direc_real = "Filter_Data/"
direc_ob = 'ob/'
direc_rean = 'cal/'
Data_final = Data_path+"temp_run/"
ob_folder_list = ["CRU", "GHCN", "UDel"]
cal_folder_list = ['era', 'cera', 'twcr']
rean_folder_list = cal_folder_list+['era5', 'wrf']
discard_var = ['lat', 'lon', 'time', 'stn', 'level']
full_folder_list = ob_folder_list+rean_folder_list
# endregion

# ----------- Creat the xarra.Dataset for each parameters ---------- #
#region
station_nc_ls = ['city_T', 'city_P']

year_grib = np.arange(1943, 1954)
# create dataset span 10 years of era5 -> era.precip
var_grib = ['total_precipitation', '2m_temperature']  # 10m_vwnd,10m_uwnd
var_grib_uniform = ['precip', 'tmp']  # u10,v10

var_real = ["pre", "tmp"]
temp_var = ["tmp", "tmx", "tmn"]
prep_var = ["precip", "no_day", "rhum"]
month = ["Jan", "Feb", "Mar", "Apr", "May", "June",
         "July", "Aug", "Sep", "Oct", "Nov", "Dec"]
season = ["JJA", "MAM", "SON", "DJF"]
option = ["f", "bss", "base"]
#endregion

# BECAREFUL FOR PRECIPITATION:
# - ERA5: 1h-23h: 23h + 0h (next day) = 24 hr accumulation

# ---------------------- Extract coord of city --------------------- #
# region
city_path = "City.csv"
city_list = pd.read_csv(f'{Data_path}{city_path}')
city_lat = city_list.iloc[:, 1]
city_lon = city_list.iloc[:, 2]
city_acr = city_list.iloc[:, 3]
city_full_name = city_list.iloc[:, 0]
#endregion

# ---------------- Extract variable from csv dataset --------------- #
#region
all_para = temp_var[1:]+prep_var[1:]
city_lat_T = []
city_lon_T = []
city_info_T = []
city_station_T = []
city_lat_P = []
city_lon_P = []
city_info_P = []
city_station_P = []
#endregion

class StationData:
    def __init__(self, station_nc_ls, Data_path,direc_real):
        self.station_nc_ls = station_nc_ls
        self.Data_path = Data_path
        self.direc_real = direc_real

    def time_create(self, dataset_csv, name):
        # Extract the np_array of the year from the 1st column
        time_column = dataset_csv.iloc[:, 0]
        no_year = time_column.dropna().to_numpy()
        time_dims = []
        len_ds = 0
        # Number of rows
        if len(no_year)*12 < len(dataset_csv.index):
            raise ValueError
        else:
            for year in no_year:
                for month in range(1, 13):
                    month = f'0{month}'if month < 10 else month
                    time_dims.append(f'19{int(year)}-0{month}-01')
                    #len_ds += 1
                    #if len_ds == len(dataset_csv.index):
                    #    break

            # Drop the dimension in which year are 43.0, 44.0
            dataset_csv = dataset_csv.drop('time', axis=1)
            # Change the name of the variable
            if name.find("_T") != -1:
                dataset_csv.columns = temp_var
            else:
                dataset_csv.columns = prep_var

            # To replace a column using a list then must call df['name']
                # It will raise a Warning but doesn't care
                # The warning askes whether we fix the org df
            dataset_csv['time'] = pd.to_datetime(time_dims)

            # Must transfer to date time so it conforms to the full time period
            dataset_csv = dataset_csv.set_index("time", drop=True)

            # Change this into dataframe
            dt_xr = xr.Dataset.from_dataframe(dataset_csv)
            dt_xr = dt_xr.reindex(time=pd.date_range(
                "1943-01-01", "1953-12-01", freq="MS"))

            # Fill the missing value with nan
            dt_xr = dt_xr.fillna(np.nan)

            return dt_xr.astype(float)

    def netcdf_create_station(self, ls_station, ls_lat, ls_lon, ls_info):
        # Create dataset for city_T and city_P with station, lat, lon variables
        ds_concat = xr.concat(ls_info, pd.Index(ls_station, name='station'))
        # Add variable with lower dimension
        ds_concat['latitude'] = (('station'), ls_lat)
        ds_concat['longitude'] = (('station'), ls_lon)
        # Create DataFrame using dict like below
        return ds_concat


    """Produce list to plot"""
    """for i in range(len(all_para)):
        if all_para[i] in temp_var:
            os.makedirs(f'T_pic/{all_para[i]}')
        else:
            os.makedirs(f'P_pic/{all_para[i]}')"""

    # Seperate data into multiple csv
    """
    for sheet_name,df in data.items():
        name=f'{direc}{sheet_name}.csv'
        ##print(name)

        #transfer sheet into csv - ONLY USE THIS TO EXTRACT SHEET INTO CSV
        df.to_csv(name,header=None,index=True)
    """
    # """

    def create_station_nc(self): #_ls, Data_path, direc_real, city_list):
        for station_nc in self.station_nc_ls:
            odir = Path(self.Data_path,self.direc_real,station_nc,".nc")
            tail = station_nc[-2:]
            # Check if the station netcdf file has been created
            if odir.exists():
                station_ds = xr.open_dataset(odir)
                # print(f'opened {station_nc}.nc')
            else:
                # Create list to hold info of the dataset
                city_station_gen, city_lat_gen, city_lon_gen, city_info_gen = []

                # use for listing all the name of the folder

                # name is the full directory link since we're using Path
                for name in Path(Data_path,direc_real).iterdir():
                    # Check if the file is csv
                    if tail+'.csv' not in name.name:
                        continue
                    raw_station_df = pd.read_csv(name, index_col=0, header=None)

                    # MUST DO!!!!
                    # Must drop empty rows before converting to xarray.Dataset()
                    # implement this in order to clear all those unnessacry

                    if len(raw_station_df.columns > 4):
                        # Set this for easy replacement
                        try:
                            # Lock the first four column: time + 3 col of variables
                            raw_station_df_org = raw_station_df.iloc[:, 0:4]
                            raw_station_df_org.columns = [
                                'time', 'var_1', 'var_2', 'var_3']
                            # Make sure the index is time, name is file name
                            raw_station_df_fin = self.time_create(raw_station_df_org, name)
                            city_ac = name[:-6]
                            try:
                                index_city = city_list.index[(
                                    city_list['AC'] == city_ac)].values[0]
                            except:
                                # print(f"Error with ac {name}")
                                continue
                            city_lat_ac = city_lat_gen[index_city]
                            city_lon_ac = city_lon_gen[index_city]

                            city_station_gen.append(city_ac)
                            city_lat_gen.append(city_lat_ac)
                            city_lon_gen.append(city_lon_ac)
                            city_info_gen.append(raw_station_df_fin)
                        except Exception as e:
                            # print(e)
                            # print(f"Error happened with {name}")
                            # #print(raw_station_df)
                            continue

            # Create dataset for city_T and city_P
            station_ds = self.netcdf_create_station(city_station_gen, city_lat_gen,
                                            city_lon_gen, city_info_gen)

            # Write to netcdf format
            station_ds.to_netcdf(odir, 'w', engine="h5netcdf",
                                format='NETCDF4')

            yield station_nc, station_ds, tail

    def process_station_data(self):
        for station_nc, station_ds, tail in self.create_station_nc(station_nc_ls):
        # --------------- Assign value for city_T and city_P --------------- #
            globals()[f'{station_nc}'] = station_ds  # dataset

        # --------------- From now we have city_T and city_P --------------- #
            globals()[f'station_focus{tail}'] = cut_time(
                station_ds, yr=1943, yr_end=1947, full="yes")
            globals()[f'station_full{tail}'] = station_ds

# -------------- Extract variable from gridded dataset ------------- #


def ds_open(folder_name):
    # Function to name each ds with its variables
    # var_left: in a dataset with multiple variables, choose the one that is not netcdf yet
    # var_skip: var that already has netcdf file -> only open the netcdf
    def ds_var_name(filename, ds_name, var_left=None, var_skip=None):
        yr = 1943
        yr_end = 1953
        if ds_name == "GHCN":
            name_final = "GHCN_precip"
            ds_dir_final = Data_final+name_final+'.nc'
            if os.path.exists(ds_dir_final):
                ds = xr.open_dataarray(ds_dir_final)
            else:
                ds = xr.open_dataset(filename, decode_times=False)
                units, reference_date = ds.time.attrs['units'].split('since')
            # #print(reference_date)
                if reference_date.find('19') == -1:
                    reference_date = '1900-1-1 00:00:0.0'
                    ds['time'] = pd.date_range(
                        start=reference_date, periods=ds.sizes['time'], freq='MS', offset='15D')
                    ds = cut_time(
                        ds['precip'], yr=yr, yr_end=yr_end, full="yes")
                    ds.to_netcdf(ds_dir_final, 'w', engine="h5netcdf",
                                 format='NETCDF4')
            globals()[name_final] = ds
            # print(f'Opened {name_final}')
            return
        else:
            if var_skip is not None:
                ds_var = xr.open_dataset(filename)
                ds_var = cut_time(ds_var, yr=yr,
                                  yr_end=yr_end, full="yes")
                globals()[f'{ds_name}_{var_skip}'] = ds_var
            elif var_left is not None:
                ds = xr.open_dataset(filename)
                # coords_name = list(ds.coords)
                for i, var_name_uniform in enumerate(var_left):
                    # if any(discard_name in var_name for discard_name in (discard_var+coords_name)):
                    #    continue
                    # else:
                    try:
                        [ds_var, var_name_uniform] = cut_var(
                            ds, var=var_name_uniform)
                    except:
                        # print(f"Cannot find {var_name_uniform} in {ds_name}")
                        continue
                    if var_name_uniform == "tmn":
                        if "C" not in ds_var.attrs['units']:
                            ds_var = ds_var - 273.15
                        else:
                            ds_var = ds_var
                    elif var_name_uniform == "precip" and "mm" not in ds_var.attrs['units']:
                        if ds_var.attrs['units'] == "kg/m^2":
                            factor_p = 30  # units: mean - kg/m^2
                        elif ds_var.attrs['units'] == "m":
                            factor_p = 1000  # unit: m
                        elif ds_var.attrs['units'] == "cm":  # unit: cm
                            factor_p = 10
                        ds_var = ds_var*factor_p
                    ds_var = cut_time(
                        ds_var, yr=yr, yr_end=yr_end, full="yes")
                    name_final = f'{ds_name}_{var_name_uniform}'
                    ds_var.to_netcdf(Data_final+name_final+".nc", 'w', engine="h5netcdf",
                                     format='NETCDF4')
                    globals()[name_final] = ds_var
                    # print(f'Opened {name_final}')
            return

    for ds_name in folder_name:
        if folder_name == ob_folder_list:
            filename = f'{Data_path}{direc_ob}{ds_name}.nc'
            # Extract each variable of ob_list
            ds_var_name(filename, ds_name)
        else:
            folder_name = f'{Data_path}{direc_rean}'
            var_name_uniform_list = []
            # Only use when the filename has the list of keys
            for ds_name_file in os.listdir(f'{folder_name}{ds_name}'):
                if ds_name_file.endswith(".nc"):
                    # using the name of the file -> extract keys
                    try:
                        # silly of me: i combine the level variable into one for cera,era
                        # -> Should not do that again and leave it alone
                        # In case, not combine use this to deal with it
                        # var_name_uniform = ds_name_file[:-3].split('_')[1]
                        # It will take only the name of the variable
                        var_name_uniform_list = ds_name_file[:-3].split('_')[
                            1:]
                    except:
                        print(
                            "You haven't opened this {ds_name_file}. Be careful and check again if needed")
                    var_left = []
                    for i, var_name_uniform in enumerate(var_name_uniform_list):
                        ds_final_path = f'{Data_final}{ds_name}_{var_name_uniform}.nc'
                        # the key has netcdf will be passed
                        if os.path.exists(ds_final_path):
                            ds_var_name(
                                ds_final_path, ds_name=ds_name, var_skip=var_name_uniform)
                        # group the keys that has not the netcdf version
                        else:
                            var_left.append(var_name_uniform)
                    if len(var_left) != 0:
                        filename = f'{folder_name}/{ds_name}/{ds_name_file}'
                    # Extract each variable of cal_list
                        ds_var_name(filename, ds_name, var_left=var_left)
                else:
                    continue
    return

# Function to extract gribfile for era5


def call_check_dataset(filename, type_option=None, open=None, check=None):
    # if os.path.exists(Data_path_grib+filename):
    filename = f'{Data_path_grib_synop+filename}.nc'
    print(filename)
    if os.path.exists(filename) is True:
        filename = filename
    else:
        filename = f'{Data_path_grib+filename}_even.nc'
        if os.path.exists(filename) is not True:
            raise ValueError("There is no such file")
        else:
            filename = filename

    if open is None:
        return filename
    else:
        try:
            if type_option == "array":
                Tm = xr.open_dataarray(filename)

            elif type_option == "set" or type_option is None:
                Tm = xr.open_dataset(filename)
        # Return the variable from the datasets
        # If it is a data-array, return the whole DataArray
            if check is None:
                print(Tm)
                return Tm
            else:
                return list_var(Tm)
        except Exception as e:
            print(e)
            raise KeyboardInterrupt("There is no such file")


# Output: ds or list of ds, name of var of list of var

# Open and get the variable for cals
ds_open(cal_folder_list)
# Output: globals()[f'{ds_name}_{var_skip}']

def grib_group(var_grib, var_grib_uniform=None):
    ds_final_dir = f'{Data_final}era5_{var_grib_uniform}.nc'
    ds_final_dir_wrf = f'{Data_final}era5_{var_grib_uniform}_wrf.nc'
    ds_merge_dir = f'{Data_final}era5_{var_grib_uniform}_merge.nc'

    if os.path.exists(ds_final_dir) is not True or os.path.exists(ds_final_dir_wrf) is not True:
        group = []

        # Don't use call_dataset for multiple file like this
        # Instead append all of these into a list then use open_mfdataset
        # It will reduce the size of the merge_array, or else, merge cannot run all
        # Can use dask_array but nah, only use that for way-to-big single dataarray or dataset
        # """
        if os.path.exists(ds_merge_dir) is True:
            ds_merge = xr.open_dataset(ds_merge_dir).sortby("time")
        else:
            for year in year_grib:
                ds_name = call_check_dataset(f'{var_grib}_{year}')
                ds_name_2 = call_check_dataset(f'{var_grib}_{year}_odd')
                group.extend([ds_name, ds_name_2])

            # To combine the list of Datasets or DataArrays, the fastest way is to use open_mfdataset
            # Open_mfdataset convert it into a single Dask Array
            # Combine the directory of the file in a list
            # REMEMBER: Time dimension in each file is format like this 0,1,2,3.....hours from reference dates
            # Each file has different reference dates
            # REMEMBER: Must keep decode_cf=True ->it calculates time according to the time reference dates
            # It will be warning on large chunk -> ignore them
            print("Trying to merge era5_synop")
            t_start = process_time()
            try:
                ds_merge = xr.open_mfdataset(
                    # chunks={'lat': 20, 'time': 50, 'lon': 24},
                    group, combine="nested", concat_dim="time",
                    decode_cf=True,
                    compat="no_conflicts", autoclose=True)
            except:
                ds_merge = xr.open_mfdataset(
                    # chunks={'lat': 20, 'time': 50, 'lon': 24},
                    group, combine="by_coords", compat="no_conflicts",
                    decode_cf=True,
                    autoclose=True)
            print("Merge succeeded")

            # Another way but not tested yet
            # units, reference_date = ds_merge.time.attrs['units'].split(
            #    'since')
            # ds_merge['time'] = pd.date_range(
            #    start=reference_date, periods=ds_merge.sizes['time'], freq='H')

            # Some variables takes in both time and step as dimension
            # MUST USE LOAD since the data is large so its format is Dask.Array
            ds_merge.load().to_netcdf(ds_merge_dir, 'w',
                                      engine="h5netcdf", format='NETCDF4')
            t_stop = process_time()
            print(t_start, t_stop)
        try:
            ds_drop = ds_merge
            ds_rename = ds_drop.squeeze(drop=True)
        # If that is not the case, then justdrop abundant coordinates
        except:
            try:
                ds_drop = ds_merge.drop(
                    ['valid_time', 'step', 'number', 'surface'])
                ds_rename = ds_drop.squeeze(drop=True)
            except:
                # Stack the `time` and `step` dimensions into a single `valid_time` dimension
                ds = ds_merge.stack(time_step=("time", "step"))
                # MUST DELETE ALL AD`DED COORDS FOR IT NOT TO BE MESSED WHEN RUN AVG
                # Rename the `valid_time` dimension to `time`
                ds_swap = ds.swap_dims({"time_step": "valid_time"}).drop(
                    ['time', 'time_step', 'step', 'number', 'surface'])
                ds_rename = ds_swap.squeeze(
                    drop=True).rename({'valid_time': 'time'})

        ds_shift = shift_time(ds_rename, hour_period="7H")
        [ds_var, var_uniform] = list_var(ds_shift)

        method_var = "mean"
        # and ("mm" not in ds_var.attrs['units']):
        if (var_uniform == "precip"):
            ds_var = ds_var*1000  # units:m
            method_var = "sum"
        else:
            if var_uniform == "tmp":
                # if "C" not in ds_var.attrs['units']:
                ds_var = ds_var - 273.15
                # else:
                #    ds_var = ds_var  # units:K
        # """
        # Extract ERA5 data to compare with wrf
        if os.path.exists(ds_final_dir_wrf):
            globals()[f'era5_{var_grib_uniform}_wrf'] = xr.open_dataset(
                ds_final_dir_wrf)
        else:
            # ds_var_wrf = ds_shift.sel(time=slice(#"1944-09-25", "1945-03-02"), drop=True)
            ds_var_wrf = cut_time(
                ds_shift, yr=1944, mon=9, day=25, yr_end=1945, mon_end=2, day_end=1, full='yes')
            ds_var_wrf.to_netcdf(ds_final_dir_wrf, 'w',
                                 engine="h5netcdf", format='NETCDF4')
            globals()[f'era5_{var_grib_uniform}_wrf'] = ds_var_wrf
        # print(ds_var_daily)
        # """

        if os.path.exists(ds_final_dir):
            globals()[f'era5_{var_grib_uniform}'] = xr.open_dataset(
                ds_final_dir)
        else:
            if method_var == "sum":
                try:
                    ds_var = resample(ds=ds_shift,
                                      freq="MS", closed='left', method='sum')
                except:
                    raise ValueError("ERA5 has problem with resample: sum")
            else:
                try:
                    ds_var = resample(ds=ds_shift,
                                      freq="MS", closed='left', method="mean")
                except:
                    raise ValueError("ERA5 has problem with resample: mean")
            ds_var.to_netcdf(ds_final_dir, 'w',
                             engine="h5netcdf", format='NETCDF4')
        # print(f'Open era5_{var_grib_uniform}')
            globals()[f'era5_{var_grib_uniform}'] = ds_var
    else:
        if os.path.exists(ds_final_dir):
            globals()[f'era5_{var_grib_uniform}'] = xr.open_dataset(
                ds_final_dir)
        if os.path.exists(ds_final_dir_wrf):
            globals()[f'era5_{var_grib_uniform}_wrf'] = xr.open_dataset(
                ds_final_dir_wrf)

        # print(f'Open era5_{var_grib_uniform} as netcdf')
    return
# Output: Call the dataset that time has been treated

# --------------- Open dataset (ob, cal, downscaling) -------------- #

# Open and get the variable for ob
# ds_open(ob_folder_list)


# Open variable of era5
for i, var_grib_ind in enumerate(var_grib):
    grib_group(var_grib_ind, var_grib_uniform[i])
# land_mask_sea = call_dataset('land_sea_mask_1944')
# is_ocean_era5 = land_mask_sea.isel(time=0).drop(
#    'time').pipe(lambda x: x*0 == 0)


# Open downscaling data
Data_wrf = "/work/users/tamnnm/wrf/WRF/test/RUN_ERA5_194412_9/"
tmn_wrf_list = []
precip_wrf_list = []
var_left = []
wrf_list = []
var_wrf_list = []
for i, var_uniform in enumerate(temp_var+prep_var):
    ds_final_dir = Data_final+"wrf_"+var_uniform+".nc"
    if os.path.exists(ds_final_dir):
        ds_var = xr.open_dataset(ds_final_dir)
        globals()[f'wrf_{var_uniform}'] = ds_var
    else:
        var_left.append([var_uniform, ds_final_dir])

# Must keep sortedfor the time to be exactly ordered
for filename in sorted(os.listdir(Data_wrf)):
    if "wrfout_d02" in filename:
        ds_wrf_name = Dataset(Data_wrf+filename)
        wrf_list.append(ds_wrf_name)
for i, var_name_dir in enumerate(var_left):
    var_uniform = var_name_dir[0]
    var_dir = var_name_dir[1]
    if var_uniform == "precip":
        wrf_name = "RAINC"
        wrf_name_2 = "RAINNC"
    elif var_uniform == "tmp":
        wrf_name = "T2"
    # elif var_uniform == "rhum":
    #    wrf_name = "rh2"
    # elif var_name == "tmn":
    #    wrf_name = "T2_MIN"
    # elif var_name == "tmx":
    #    wrf_name = "T2_MAX"
    else:
        continue

    ds_raw = getvar(wrf_list, wrf_name, timeidx=ALL_TIMES,
                    method='cat')
    ds_var = shift_time(ds_raw, hour_period="7H")
    try:
        del ds_var.attrs['projection']
    except:
        print("No projection")
    ds_var.to_netcdf(f'{Data_final}wrf_RAINC.nc', 'w', engine="h5netcdf",
                     format='NETCDF4')

    try:
        ds_raw_2 = getvar(wrf_list, wrf_name_2, timeidx=ALL_TIMES,
                          method='cat')
        ds_var_2 = shift_time(ds_raw_2, hour_period="7H")
        # print(ds_var_2.isel(south_north=0, west_east=0))
        try:
            del ds_var_2.attrs['projection']
        except:
            print("No projection")
        ds_var_2.to_netcdf(f'{Data_final}wrf_RAINNC.nc', 'w', engine="h5netcdf",
                           format='NETCDF4')
        ds_value_1 = ds_var.values
        ds_value_1[ds_value_1 < 0] = 0
        ds_value_2 = ds_var_2.values
        ds_value_2[ds_value_2 < 0] = 0
        ds_var_value_org = ds_value_1 + ds_value_2
        ds_var.data = ds_var_value_org
        # print(ds_var_value_org)
        # Assign an initial arrays for original to subtracct -> get the accumulate
        try:
            ds_var_initial = ds_var.shift(Time=1, fill_value=float(0))
        except:
            ds_var_initial = ds_var.shift(time=1, fill_value=float(0))
        ds_var = ds_var - ds_var_initial
        # ds_var.data = ds_var_value_final * 1000
        print(ds_var.isel(south_north=0, west_east=0))
    except Exception as e:
        if var_uniform == "precip":
            print(e)
        ds_var = ds_var

    # precipitation
    if (var_uniform == "precip"):  # and ("mm" not in ds_var.attrs['units']):
        ds_var = ds_var  # unit:m
    else:
        # if "C" not in ds_var.attrs['units']:
        ds_var = ds_var - 273.15  # unit: K
    # raise KeyError
    ds_var.to_netcdf(f'{Data_final}wrf_{var_uniform}.nc', 'w', engine="h5netcdf",
                     format='NETCDF4')

    globals()[f'wrf_{var_uniform}'] = ds_var
# print(wrf_precip)

# """
# Now choose Euler distance and choose the closest 4 points
city_good = ["HN", "DH", "DN", "H"]
# station_lon=105
# station_lat=15
for city_name in city_good:
    city_ds = city_T.sel(station=city_name)
    # print(city_ds)
    station_lon = city_ds['longitude'].values
    station_lat = city_ds['latitude'].values
    for ds_name in rean_folder_list:
        ds_cut_co = globals()[f'{ds_name}_tmp']
        number_pts = 0
        if ds_name == "GHCN":
            range = np.sqrt(7 ^ 2)
        elif ds_name == "twcr":
            range = np.sqrt(2)
        else:
            range = 0.1

        while number_pts < 4:
            ulat = station_lat+range
            dlat = station_lat-range
            ulon = station_lon+range
            dlon = station_lon-range
            try:
                test_co = cut_co(
                    globals()[f'{ds_name}_tmp'], ulat, dlat, ulon, dlon)
                # print(len(test_co[0].values))
                number_pts = len(test_co[0].values)*len(test_co[1].values)
                if number_pts < 4:
                    range += 0.1
                if ds_name == "wrf":
                    globals()[f'{ds_name}_{city_name}_co'] = test_co[2]
                else:
                    globals()[f'{ds_name}_{city_name}_co'] = test_co[0:2]
                # if ds_name == "era5" or ds_name == "wrf":
                # print(test_co)
            except Exception as e:
                print(e)
                print(ds_name)
                raise ValueError


# Example of extract hour from time dimension:
# ds = xr.Dataset({"foo": ("time", np.arange(365 * 4)), "time": time})
# ds.time.dt.hour -> output: array
"""
def plot_para(z, para, year_to_plot):
    year_plot = []
    for k in range(len(year_to_plot)):
        yr_plt = []
        # #print(year_to_plot[k])
        for j in range(len(year_to_plot[k])):
            yr_plt.append(year_to_plot[k][j]+1900)
        year_plot.append(yr_plt)
    for i in range(len(para[z].columns)):
        # test_yr=para[z].columns[i]
        # #print(test_yr)
        test = para[z].iloc[:, i].to_numpy()
        if year_plot[z][i] in (1943, 1944, 1945):
            a = 1
            if year_plot[z][i] == 1943:
                c = "orangered"
            elif year_plot[z][i] == 1944:
                c = "seagreen"
            else:
                c = "indigo"
         # the first 3 years (43,44,45) #too-tired to change since HP only have 1947 go on
            plt.plot(month, test, alpha=a, color=c, marker='o')
        # elif i<8:
        else:
            a = 0.25
            plt.plot(month, test, alpha=a, marker='o')
    plt.legend(year_plot[z])
"""


def appear(ax, special=None):
    # Solve the problem that the month and year can all appear in graph
    years = mdates.YearLocator()
    months = mdates.MonthLocator()
    days = mdates.DayLocator()
    months_years = mdates.DayLocator(bymonthday=1, interval=1)
    monthsFmt = mdates.DateFormatter('%b')
    daysFmt = mdates.DateFormatter('%d')
    months_yearsFmt = mdates.DateFormatter('%Y %b')
    # add some space for the year label
    yearsFmt = mdates.DateFormatter('\n\n%Y')
    if special is None:
        ax.xaxis.set_minor_locator(months)
        ax.xaxis.set_minor_formatter(monthsFmt)
        plt.setp(ax.xaxis.get_minorticklabels(), rotation=90)
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(yearsFmt)
    else:
        ax.xaxis.set_minor_locator(days)
        ax.xaxis.set_minor_formatter(daysFmt)
        plt.setp(ax.xaxis.get_minorticklabels(), rotation=90)
        ax.xaxis.set_major_locator(months_years)
        ax.xaxis.set_major_formatter(months_yearsFmt)

    return ax


# """
# print(era5_HN_co)
# raise ValueError
# Plot all 3 in the same graph
# """
opt = "full"
for var_name in (["tmp", "precip"]):
    # fig, axs = plt.subplots(6, 3, figsize=(12,8), sharex=True, sharey=True,
    # constrained_layout=False)

    folder_name = cal_folder_list
    if folder_name == cal_folder_list:
        col = ["#f8766d", "#618cff", "#2dc2bd", "#22577a"]
        name = "rean"
    else:
        col = ["m", "#e6d800", "#00bfa0"]
        name = "ob"
    for i, city_name in enumerate(city_good):
        # Timeseries monthly dataset
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # fig, axs = plt.subplots(2, 2, figsize=(12,8), sharex=True, sharey=True,
        #                    constrained_layout=False)
        # labels =folder_plot+['era5]
        # ax=axs.flat[i]
        ds_list_plot = []
        ds_plot = pd.DataFrame()
        ds_wrf_plot = pd.DataFrame()
        folder_plot = folder_name + ['era5']
        # Plot the cal datasets
        for i, folder in enumerate(folder_plot):
            print([f'{folder}_{var_name}'])
            ds_mean_city = f'{Data_final}{folder}_{var_name}_{city_name}.nc'
            if os.path.exists(ds_mean_city) is not True:
                ds_mean = cut_mlp(globals()[f'{folder}_{var_name}'], geo_data=globals()[
                    f'{folder}_{city_name}_co'], dim_mean=["geo"])
                # ds_mean.to_netcdf(ds_mean_city, 'w', engine="h5netcdf",
                #                  format='NETCDF4')
            else:
                ds_mean = xr.open_dataset(ds_mean_city)
            ds_list_plot.append(ds_mean.to_dataframe())
        ds_plot = pd.concat(ds_list_plot, axis=1)
        # print(ds_plot)
        ds_time = ds_plot.index.to_pydatetime()
        # print(ds_plot)
        for i, folder in enumerate(folder_plot):
            plot_full = ax.plot(
                ds_time, ds_plot.iloc[:, i], alpha=1, linewidth= 0.5,  color=col[i], label=folder, marker='o')
        # try:
        #    globals()[f'wrf_{var_name}'].plot(ax=ax, alpha=0.9,
        #                                      color='r', label="Downscaled ERA5", marker='o')
        # except:
        #    #print("No downscaling here")
    # TO PLOT REAL AND REAN IN THE SAME GRAPH
    # The problem: even though having the same time series but when plot it appears a little bit difference from each other
        # to merge/concat use pd.concat (can mix all series, dataframe)
        # print(ds_plot)

        if var_name in temp_var:
            ds_real = city_T
        else:
            ds_real = city_P
        try:
            real_plot = city_T.sel(station=city_name)[f'{var_name}'] + 273.15
        # ax2 = ax.twinx()
        # g2=ax2.plot(ds,dt_plt.iloc[:,-1],alpha=1,color="#0000ff",marker='o',label="obsevered")
            plot_full_real = ax.plot(
                ds_time, real_plot,linewidth=0.5, color="k", label="obsevered", marker='o')
            # print("breath")
        except Exception as e:
            print(e)
            print(city_name)

    # Incase you have twinx plot but you want to present all legend
    # #print(lines,lg_plt)
    # labs = [l.get_label() for l in lines]
        if var_name == "precip":
            title = f'Precipiation_{city_name}'
            ax.set_ylabel("mm")
        elif var_name == "rhum":
            title = f'Relative humidity_{city_name}'
            ax.set_ylabel('%')
        elif var_name == "tmp":
            title = f'Temperature_{city_name}'
            ax.set_ylabel(r'$T^o$')
        ax.set_title(title, pad=20)

        # turn the label to 90 degrees
        ax.set_xlabel('Year')
        # ax.set_label(city_name)
        # ax.label_outer()
        appear(ax)

        # handles, labels = ax.get_legend_handles_labels()
        # fig.legend(handles, labels, loc='upper center')
        # (handles=lines,labels=labs, loc=0)
        fig.legend(loc='upper left', bbox_to_anchor=(0.13, 0.85))
        # bbox_to_anchor=(x,y,width,height)

        plt.tick_params(axis='y')
        plt.tick_params(axis='x', direction='in')
        # plt.savefig(f'{Data_path}/Data/test/{ct}.jpg',format="jpg")
        plt.savefig(
            f'{Data_path}pic/final/{var_name}_{city_name}.jpg', format="jpg", dpi=500)
        # plt.legend()
        plt.tight_layout()
        plt.close()

    # Plot wrf and insitu data
        mon_list_plot = []
        day_list_plot = []

        station_lon = city_ds['longitude'].values
        station_lat = city_ds['latitude'].values
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        real_plot_2 = real_plot.sel(time=slice(
            '1944-09-30', '1945-03-01'))
        real_plot_2 = shift_time(real_plot_2, time_period="14D")
        real_plot_2.plot(ax=ax1, marker='o', color='k', label="observed")

        # Treat downscaling data
        ds_wrf_city = f'{Data_final}wrf_{var_name}_{city_name}.nc'
        if os.path.exists(ds_wrf_city) is not True:
            ds_wrf_plot = cut_mlp(globals()[f'wrf_{var_name}'], geo_data=globals()[
                f'wrf_{city_name}_co'], dim_mean=["geo"], special='yes')
            ds_era5_plot = cut_mlp(globals()[f'era5_{var_name}_wrf'], geo_data=globals()[
                f'era5_{city_name}_co'], dim_mean=["geo"], special='yes')
            # ds_wrf_plot.to_netcdf(ds_wrf_city, 'w', engine="h5netcdf",
            #                      format='NETCDF4')
        else:
            ds_wrf_plot = xr.open_dataset(ds_wrf_city)
        # print(ds_wrf_plot.where(ds_wrf_plot.coords['Time'] == '1940-10-25'))
        # Have to change to dataframe so more easily modified time index

        # Have to do this since ds_wrf_plot is still Dataset with "Time" and XTIME as variable
        # -> Not to worry about calling the variable seperately
        time_slice = slice('1944-10-01', '1945-02-28')

        ds_wrf_plot = ds_wrf_plot.sel(Time=time_slice)
        ds_era5_plot = ds_era5_plot.sel(time=time_slice)
        if var_name == "precip":
            ds_era5_plot = ds_era5_plot * 1000
            try:
                ds_wrf_mon = resample(
                    ds=ds_wrf_plot, freq="MS", closed='left', method="sum")
                ds_wrf_day = resample(
                    ds=ds_wrf_plot, freq="1D", closed="left", method="sum")
                ds_era5_mon = resample(
                    ds=ds_era5_plot, freq="MS", closed='left', method="sum")
                ds_era5_day = resample(
                    ds=ds_era5_plot, freq="1D", closed="left", method="sum")
            except:
                raise ValueError("Dataset has problem with resample: sum")
        else:
            try:
                ds_wrf_mon = resample(
                    ds=ds_wrf_plot, freq="MS", closed='left', method="mean")
                ds_wrf_day = resample(
                    ds=ds_wrf_plot, freq="1D", closed="left", method="mean")
                ds_era5_mon = resample(
                    ds=ds_era5_plot, freq="MS", closed='left', method="mean")
                ds_era5_day = resample(
                    ds=ds_era5_plot, freq="1D", closed="left", method="mean")
            except:
                raise ValueError("Dataset has problem with resample: mean")

        # Can add .iloc[:, -1:] in case multiple value of the latter dimension are added
        df_mon = pd.concat(
            [ds_wrf_mon.to_dataframe(), ds_era5_mon.to_dataframe()], axis=1)
        df_day = pd.concat(
            [ds_wrf_day.to_dataframe(), ds_era5_day.to_dataframe()], axis=1)

        df_mon_plot, df_mon_time = shift_time(
            df_mon, time_period="14D", output="pd")
        df_day_plot, df_day_time = shift_time(df_day, output="pd")
        print(df_mon_plot)
        wrf_mon_plot = df_mon_plot.plot(ax=ax1, linewidth = 0.5, color=["#5B8C5A", "#22577a"], label=[
                                        "wrf_monthly", "era5_monthly"], marker='o')
        wrf_day_plot = df_day_plot.plot(ax=ax1, linewidth = 0.5, color=["#5B8C5A", "#22577a"], label=[
                                        "wrf_daily", "era5_daily"], marker='o')

        """
        Plot bổ sung era5_wrf + era_monthly
        """

        # ds_wrf.plot.lines(ax=ax)  # , alpha=0.9, marker='o', color='r')
        appear(ax1, special='yes')
        plt.tick_params(axis='y')
        plt.tick_params(axis='x', direction='in')
        fig1.legend(loc='upper left', bbox_to_anchor=(0.13, 0.85))
        if var_name == "precip":
            title = f'WRF_Precipiation_{city_name}'
            ax1.set_ylabel("mm")
        elif var_name == "rhum":
            title = f'WRF_Relative humidity_{city_name}'
            ax1.set_ylabel('%')
        elif var_name == "tmp":
            title = f'WRF_Temperature_{city_name}'
            ax1.set_ylabel(r'$T^o$')
        ax1.set_title(title, pad=20)
        plt.tight_layout()
        plt.savefig(
            f'{Data_path}pic/final/wrf_{var_name}_{city_name}.jpg', format="jpg", dpi=500)
        #                                      color='r', label="Downscaled ERA5", marker='o')
"""
else:
    for i in range(len(folder)):
        # TO PLOT REAL AND REAN IN THE SAME GRAPH
        # The problem: even though having the same time series but when plot it appears a little bit difference from each other
        # to merge/concat use pd.concat (can mix all series, dataframe)
        try:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            subfol = folder[i]
            plot_full = ax.plot(
                ds, dt_plt.iloc[:, i], alpha=0.8, color=col[i], label=subfol)
            real = globals()[f'{ct}_{pra_ob}_focus']
            dt_plt = pd.concat([dt_plt, real], axis=1)
            ax2 = ax.twinx()
            g2 = ax2.plot(
                ds, dt_plt.iloc[:, -1], alpha=1, color="#0000ff", label="obsevered")

            # Incase you have twinx plot but you want to present all legend
            # #print(lines,lg_plt)
            # labs = [l.get_label() for l in lines]
            if pra == "pre":
                ax.set_title(f'Pre_{ct}_{subfol}', pad=20)
                ax.set_ylabel(f'mm_{subfol}')
                ax2.set_ylabel(f'mm_observed')
            else:
                ax.set_title(f'Temperature_{ct}_{subfol}', pad=20)
                ax.set_ylabel(fr'$T^o$_{subfol}')
                ax2.set_ylabel(f'C_observed')
            # turn the label to 90 degrees
            ax.set_xlabel('year')
            # (handles=lines,labels=labs, loc=0)
            fig.legend(loc='upper left', bbox_to_anchor=(0.13, 0.85))
            appear(fig, ax)
            # bbox_to_anchor=(x,y,width,height)
            plt.tick_params(axis='y')
            plt.tick_params(axis='x', direction='in')
            # plt.savefig(f'{Data_path}/Data/test/{ct}.jpg',format="jpg")
            plt.savefig(os.path.join(Data_path, "Data", fold,
                        subfol, pra_ob, f'{ct}.jpg'), format="jpg")
            # plt.legend()
            plt.tight_layout()
            plt.show()
            plt.clf()
        except:
            continue

for b in range(len(all_para)):
    for a in range(len(pre)):
        ##print(all_para[b])
        ##print(a)
        ##print(name)
        if all_para[b] in temp_var:
            folder=f'{Data_path}Data/REAL/T_pic/{all_para[b]}'
            name=temp_name[a][:-2]
            c=r'$T^o$'
            year_plot=city_T_year
        else:
            folder=f'{Data_path}Data/REAL/P_pic/{all_para[b]}'
            name=prep_name[a][:-2]
            year_plot=city_P_year
            if all_para[b]=="pre":
                c="mm"
            elif all_para[b]=="no_day":
                c="number of day"
            else:\
                c="%"
        ##print(f'{name}_{all_para[b]}')
        plot_para(a,globals()[all_para[b]],year_plot)
        plt.title(f'{name}_{all_para[b]}')
        plt.xlabel('Month')
        plt.ylabel(c)
        plt.tick_params(axis='x',direction='in')
        plt.savefig(f'{folder}/{name}_{all_para[b]}.jpg',format="jpg")
        plt.clf()
"""


# EXAMPLE: PLOT SUBPLOT
"""
import matplotlib.pyplot as plt

fig, axs = plt.subplots(6, 3, figsize=(12,8), sharex=True, sharey=True,
                        constrained_layout=False)

labels = [f"Label {i}" for i in range(1,5)]

for ax in axs.flat:
    for i, lb in enumerate(labels):
        ax.plot([1,2], [0,i+1], label=lb)
    ax.set_xlabel("x label")
    ax.label_outer()

fig.tight_layout()
fig.subplots_adjust(bottom=0.1)   ##  Need to play with this number.

fig.legend(labels=labels, loc="lower center", ncol=4)

plt.show()
"""
