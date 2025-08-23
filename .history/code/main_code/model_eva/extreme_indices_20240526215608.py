

# ------------------------------------------------------------------ #
#                          PRELIMINARY STUFF                         #
# ------------------------------------------------------------------ #

# ----------------------- Import module ---------------------------- #
# region
# Plot modules
import warnings
from shapely.geometry import MultiPolygon
import geopandas as gpd
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
import matplotlib
from matplotlib.colors import Normalize
from matplotlib.tri import Triangulation
import matplotlib.cm as cm
import matplotlib.lines as mlines
from matplotlib.animation import FuncAnimation
import shapefile as shp
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from tkinter import ttk
from cf_units import decode_time

# data modules
import pandas as pd
import numpy as np
import time
import scipy.stats as sst
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.utils import check_random_state
import matplotlib.dates as mdates
import datetime as dt
import xarray as xr
from matplotlib.image import imread
from cdo import *
from my_junk import *

# system modules
import os
import subprocess
import shlex
import concurrent.futures as con_fu
from typing import Union, Optional, List, Tuple, Dict, Literal, Callable


cdo = Cdo()
# This prohibits that existing files are created a second time
cdo.forceOutput = False

# endregion
# ----------------------- Import plot module ----------------------- #
# region
# import metpy.calc as mpcalc
# from metpy.units import units

# Ignore FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning)

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
# endregion

# --------------------- Data path and constant --------------------- #
# region
Data_wd = "/data/projects/REMOSAT/tamnnm/"
Code_wd = "/work/users/tamnnm/code/main_code/city_list_obs/"
Data_ind = Data_wd+"wrf_data/netcdf/para/indices/"
Data_csv = Data_wd+"wrf_data/netcdf/para/csv_file/"
Data_nc = Data_wd+"wrf_data/netcdf/para/"
Data_raw_2019 = Data_wd+"dat_daily_1960_2019/"
Data_nc_2019 = Data_wd+"obs/UPDATE_METEO/"
Data_obs_list = Code_wd+"city/"
Data_obs_pts = Code_wd+"city_pts/"
suffix_2019 = '_1960_2019.txt'
img_wd = "/work/users/tamnnm/code/sup/image/"
os.chdir(Data_nc)
# Data_2023 = Data_wd+"dat_daily_1961_2021"
indice_dict = {167: {'min': ('Tnn', 'Tnx', 'Tn10p', 'Tn90p', 'CSDI', 'TN20',), 'max': ('Txx', 'Txn', 'Tx90p', 'Tx10p', 'WSDI', 'SU25',), 'other': ('DTR',)},
               228: ('CDD', 'CWD', 'R10mm', 'R20mm', 'R50mm', 'R1mm', 'Rx1day', 'Rx5day', 'SDII', 'PRCPTOT', 'R99p', 'R95p')}
temp_tuple = tuple(val for key in indice_dict[167].values() for val in key)
rain_tuple = indice_dict[228]

# Keep this since each indices has a seperate file
rean_year_dict = {'cera': {'start_year': 1901, 'end_year': 2009},
                  'era': {'start_year': 1901, 'end_year': 2010},
                  'era5': {'start_year': 1940, 'end_year': 2022},
                  'noaa': {'start_year': 1887, 'end_year': 2014}}

pctl_tup = ('90p', '95p', '10p', '99p')
param_dict = {'uwnd': {'no': 165, 'type': 'non_extreme', 'acro': 'Um', },
              'vwnd': {'no': 165, 'type': 'non_extreme', 'acro': 'Vx', },
              'min': {'no': 167, 'type': 'extreme', 'acro': 'Tm', },
              'max': {'no': 167, 'type': 'extreme', 'acro': 'Tx', },
              'T2m': {'no': 167, 'type': 'non_extreme', 'acro': 'T2m', 'list_name': ('var167', 'air', 't2m', 'T2m',)},
              'R': {'no': 228, 'type': 'extreme', 'acro': 'R', 'list_name': ('var228', 'R', 'apcp', 'tp',)},
              }
metrics_compare = ['mape', 'R', 'nrmse', 'nmae', 'DISO', 'Taylor_score']
metrics_singular = METRICS_SINGULAR

def_start_year = 1961
def_end_year = 2019
# endregion

# ------------------------- optional choice ------------------------ #
option = "else"
# "time" = plot timeseries
# "indices" = calculate the indices

# ------------------------------------------------------------------ #
#                        CALCULATING THE INDEX                       #
# ------------------------------------------------------------------ #

# ------------------------ check indices ----------------------- #


def check_non_exist(name, force=False):
    return False if os.path.exists(name) else True

# ------------------------- Run cdo command ------------------------ #

# Run the cdo command directectly


def run_cdo_command(command: Callable, input: str, output: str, args: Optional[List] = None, force=False):
    if args is None:
        args = {}
    if force:
        command(*args, input=input, output=output, options='-L')
    else:
        if check_non_exist(output):
            command(*args, input=input, output=output, options='-L')
            print(output)
    return

# Run the cdo command for base file (must be sequential)


def run_cdo_base(command: str, input: str, output: str, args: Optional[List] = None):
    # Write the command
    if args is None:
        cdo_command = f'cdo -L {command} {input} {output}'

    else:
        cdo_command = f'cdo -L {command},{",".join(map(str, args))} {input} {output}'

    # Split command to the right syntax
    command = shlex.split(cdo_command)
    # We can use the subprocess.Popen (can interact with it more)
    if check_non_exist(output):
        result = subprocess.run(command, check=True)
    return result

# ---------------------- Generating base file ---------------------- #

# Basefiles are created in "/data/projects/REMOSAT/tamnnm/wrf_data/netcdf/para"
# Original file are symbolink => Do not delete them => go to the original folder


def gen_base(var: Optional[Literal["tp", "max", "min", "vwnd", "uwnd"]], dataset: Optional[Literal["cera", "era", "era5", "noaa", "obs"]],
             start: Optional[int] = 1961, end: Optional[int] = 1990, percentile: list = None, window='5', *args):
    # create base file
    no = param_dict[var]['no']
    fol = f'para_{no}/'

    def full_name(name, fol=fol):
        return fol+name

    # input file, special case for temp(prefix of min/max)
    ifile = f'{dataset}_{no}.nc' if no != 167 else f'{var}_{dataset}_{no}.nc'

    # output file, base period
    ofile = 'b_'+ifile
    full_ifile, full_ofile = full_name(ifile), full_name(ofile)

    if no == 228:
        # Number of wet days
        run_cdo_command(
            cdo.yearsum, input=f' -gec,1 {full_ifile}', output=f'indices/no_wet_{dataset}.nc')
        # Total precipitation of w
        run_cdo_command(
            cdo.yearsum, input=f'-mul {full_ifile} -gec,1 {full_ifile}', output=f'indices/PRCPTOT_{dataset}.nc')

    if check_non_exist(full_ofile):
        time_range = f'{start-1}-12-30,{end}-01-02' if dataset != 'obs' else f'{start}-01-01,{end}-12-31'
        # Extract the start year of obs to decide
        if no == 228:
            run_cdo_base(
                'seldate', input=f'-setctomiss,0 -mul {full_ifile} -gec,1 {full_ifile}', output=full_ofile, args=[time_range])
        else:
            run_cdo_base(
                'seldate', input=full_ifile, output=full_ofile, args=[time_range])
        print('Base file:', full_ofile)

    if param_dict[var]['type'] == 'extreme':
        if check_non_exist(full_ofile):
            raise ValueError("Mission failed on"+full_ofile)

        if percentile:
            # Input base file is the output file for the percentile
            ibfile = full_ofile
            pminfile, pmaxfile = full_name(
                f'bp_min_{ifile}'), full_name(f'bp_max_{ifile}')
            # Minimum values
            run_cdo_base('ydrunmin', ibfile, pminfile, [window])
            print('Min: ', pminfile)
            # Maximum values
            run_cdo_base('ydrunmax', ibfile, pmaxfile, [window])
            print('Max:', pmaxfile)
            # n-window running percentile of base value
            for rp in percentile:
                opfile = full_name(f'b{rp}p_{ifile}')
                # Quantile file
                run_cdo_base(
                    'ydrunpctl', input=f'{ibfile} {pminfile} {pmaxfile}', output=opfile, args=[rp, window, 'pm=r8'])
                print('Percentile: ', opfile)
    return


def run_gen_base(dataset):
    gen_base(var='R', start=1961, end=1990, dataset=dataset,
             percentile=[95, 99], window='5')
    gen_base(var='min', start=1961, end=1990,
             dataset=dataset, percentile=[10, 90], window='5')
    gen_base(var='max', start=1961, end=1990,
             dataset=dataset, percentile=[10, 90], window='5')
    return

# --------------------- Getting the input name --------------------- #


def get_ifiles(ind, dataset):
    # List through the dictionary of indices and check the variables corresponding with it
    # e.g. CDD -> 228
    no = next((no for no in list(indice_dict.keys()) if ind in indice_dict[no]
               or
               any(ind in indice_dict[no][sub_no] for sub_no in list(indice_dict[no].keys()))),
              None)

    # folder for each variable
    fol = f'para_{no}/'
    fmin = f'min_{dataset}_{no}.nc'
    fmax = f'max_{dataset}_{no}.nc'
    if no == 167:
        if ind in indice_dict[no]['other']:
            return fol, fmin, fmax
        if ind in indice_dict[no]['min']:
            return fol, fmin, None
        if ind in indice_dict[no]['max']:
            return fol, fmax, None
    else:
        return fol, f'{dataset}_{no}.nc', None

# ------------------------ Generating index ------------------------ #


class gen_ind:
    def __init__(self):
        self.ifile = None

    def __call__(self, ind, dataset):
        self.ind = ind
        self.dataset = dataset
        self.fol, iminfile, imaxfile = get_ifiles(ind, dataset)
        print(self.ind)
        self.iminfile = self.fol+iminfile
        self.imaxfile = self.fol+imaxfile
        # print(get_ifiles(ind,dataset))
        ifile = iminfile or imaxfile
        self.ifile = self.fol+ifile
        self.ofile = f'indices/{ind}_{dataset}.nc'
        print(ind, dataset)
        # base percentile file
        if ind == "WSDI":
            pctl_ind = '90p'
        elif ind == "CSDI":
            pctl_ind = '10p'
        else:
            # T10p, T90p,R99p, R95p
            pctl_ind = next(
                (value for value in pctl_tup if value in ind), None)
        if pctl_ind is not None:
            self.ibfile = self.fol+f'b{pctl_ind}_{ifile}'

        # calculation for index
        if self.ind in ['CDD', 'CWD']:
            self.gen_con_day()
        # elif self.ind == "PRCPTOT":
        #     self.gen_prcptot()
        elif 'Rx' in self.ind:
            self.gen_rxnday(self.ind[2])
        elif self.ind == "SDII":
            self.ifile_tot = f'indices/PRCPTOT_{dataset}.nc'
            self.ifile_day = f'indices/no_wet_{dataset}.nc'
            self.gen_sdii()
        elif self.ind in ['R95p', 'R99p']:
            self.gen_compare_rptot()
        elif self.ind in ['R10mm', 'R20mm', 'R50mm', 'R1mm', 'SU25', 'TN20']:
            self.gen_compare_thres(self.ind)
        elif self.ind in ['Tx90p', 'Tx10p', 'Tn90p', 'Tn10p']:
            self.gen_compare_pctlday()
        elif self.ind in ['Txn', 'Tnn']:
            self.gen_min()
        elif self.ind in ['Tnx', 'Txx']:
            self.gen_max()
        elif self.ind == 'DTR':
            self.gen_dtr()
        elif self.ind in ['WSDI', 'CSDI']:
            self.gen_temp_si()
        else:
            return
        run_cdo_command(self.command, self.ifile, self.ofile)
        return

    def gen_con_day(self):
        if self.ind == "CDD":
            method = 'ltc'
        if self.ind == "CWD":
            method = 'gec'
        self.command = cdo.yearmax
        self.ifile = f'-consects -{method},1 {self.ifile}'

    def gen_compare_thres(self, method, thres=1):
        # Rnmm ,SU25, TN20
        if 'mm' in self.ind:
            method = 'gec'
            thres = self.ind[1:-2]
        elif 'T' in self.ind:
            thres = self.ind[2:4]
        else:
            thres = 1
        if 'SU' in self.ind:
            method = 'gtc'
            # Convert to Kelvin
            thres = 273.15+25
        if 'TN' in self.ind:
            method = 'ltc'
            # Convert to Kelvin
            thres = 273.15+20
        self.command = cdo.yearsum
        self.ifile = f'-{method},{thres} {self.ifile}'

    def gen_rxnday(self, n='5'):
        self.command = cdo.monmax
        if n != '1':
            self.ifile = f'-runsum,{n} {self.ifile}'

    def gen_sdii(self):
        self.command = cdo.div
        self.ifile = f'{self.ifile_tot} {self.ifile_day}'

    def gen_prcptot(self, thres=1):
        self.command = cdo.yearsum
        self.ifile = f'-mul {self.ifile} -gec,{thres} {self.ifile}'

    def gen_compare_rptot(self):
        percentile = self.ind[1:3]
        method = "ltc" if int(percentile) < 50 else "gtc"
        self.command = cdo.yearsum
        self.ifile = f'-mul {self.ifile} -{method},0 -sub {self.ifile} {self.ibfile}'

    def gen_compare_pctlday(self):
        if 'R' in self.ind:
            percentile = self.ind[1:3]
        elif 'T' in self.ind:
            percentile = self.ind[2:4]
        else:
            pass
        method = "ltc" if int(percentile) < 50 else "gtc"
        # print(self.ifile, self.ibfile)
        self.command = cdo.yearsum
        self.ifile = f'-{method},0 -sub {self.ifile} {self.ibfile}'

    def gen_min(self):
        self.command = cdo.monmin

    def gen_max(self):
        self.command = cdo.monmax

    def gen_dtr(self):
        self.command = cdo.monmean
        self.ifile = f'-sub {self.imaxfile} {self.iminfile}'

    def gen_temp_si(self):
        method = "ltc" if self.ind == "CSDI" else "gtc"
        self.command = cdo.yearsum
        self.ifile = f'-gec,6 -consects -{method},0 -sub {self.ifile} {self.ibfile}'


# Initialize the class
gen_class = gen_ind()

# ----------------- MUST RUN FILE BASE FIRST OF ALL ---------------- #
# region


def base_fu():
    with con_fu.ProcessPoolExecutor() as executor:
        futures = []
        # Reanalysis
        for rean_name in list(rean_year_dict.keys()):
            futures += [executor.submit(run_gen_base, dataset=rean_name)]
        # Observation
        futures += [executor.submit(run_gen_base, dataset='obs')]
    con_fu.wait(futures)
    print("Finish creating base file")
    return
# endregion

# -------------------------- RUN THE TEST -------------------------- #



def ind_fu():
    with con_fu.ProcessPoolExecutor() as executor:
        futures = []
        # Reanalysis
        for ind in temp_tuple + rain_tuple:
            # for rean_name in list(rean_year_dict.keys()):
            # print(ind,rean_name)
            # futures += [executor.submit(gen_class,
            #                            ind=ind, dataset=rean_name)]
            # Observation
            # print(ind)
            futures += [executor.submit(gen_class, ind=ind, dataset='era5')]
    # con_fu.wait(futures)
    print("Finish creating indices file")

for ind in temp_tuple + rain_tuple:
    gen_class(ind, 'obs')
# Run indices file generator
# DONE
print("Check base file")
base_fu()
print("Check indices file")
ind_fu()

raise KeyboardInterrupt

# ------------------------------------------------------------------ #
#              OUTPUT THE RESULT FOR ALL DATASET AS CSV              #
# ------------------------------------------------------------------ #

# --------------------- Call for array and list -------------------- #
# 228: Tp
# 165: u wind
# 166: v wind
# 167: 2m temperature
# 168: 2m dewpoint temperature

# -------------------- Open grib and netcdf file ------------------- #

# --------------- Extract 4 closest points -> pair ----------------- #
# Create a dictionary of 4 closest points for each city
# form: {'city': ((lat1,lon1),(lat2,lon2),(lat3,lon3),(lat4,lon4))}


def city_ds_pts(dataset):
    city_pts = {}

    # GO BACK TO NC_SUP TO MAKE CHANGE TO CUT_POINTS SO IT WILL CONVERT THE POINT DIRECTLY TO FLOAT
    def create_pairs(line):
        return [[float(line[i]), float(line[i+1])] for i in range(1, 9, 2)]

    with open(Data_obs_pts+dataset+"_euler_pts.txt", 'r') as f:
        lines = f.readlines()
        f.close()
    for line in lines:
        line = line.split(',')
        city_pts[line[0]] = list(create_pairs(line))
    return city_pts

# --------------------- Calculate the DISO index --------------------- #

# Create a pivot dataframe compose of all the diso from all the reannalysis
# Split the process of getting diso in 2 since it could be memory high


def merge_df(var):
    # Read the netcdf observational file
    full_ds = xr.open_dataset(Data_nc_2019+"OG_"+var+"_daily_1960_2019.nc")
    # Extract city with data gap smaller than 5% + Available time >=20 years (before 1990)
    sel_ds = full_ds.where((full_ds['data_gap'] <= 0.05)  # data gap < 0.05
                           # more than 20 years
                           & (full_ds['end_year']-full_ds['start_year'] >= 20)
                           & (full_ds['start_year'] <= 1970), drop=True)  # start year must start before 1970

    # list name of cities above
    list_no_city = sel_ds['no_station'].values
    list_name_city = [item.decode('utf-8')
                      for item in sel_ds['name_station'].values]

    # ------ Generate the diso index for each reaannalysis dataset ----- #
    def gen_df(dataset):
        # define name of file of indices
        ind_csv = f'{Data_csv}single/{dataset}_{var}.csv'
        obs_csv = f'{Data_csv}single/obs_{dataset}_{var}.csv'
        if not check_non_exist(ind_csv):
            return pd.read_csv(ind_csv)

        # extract the points for a specific dataset from the text file
        pts_dict = city_ds_pts(dataset)  # dict
        list_ind = rain_tuple if var == 'R' else temp_tuple
        # subset the data to 1961-2019

        # Define the start & end year for each dataset since not all of them end by 2019
        # This also use to crop out the observational datasets

        start_year = max(rean_year_dict[dataset]['start_year'], 1960)
        end_year = min(rean_year_dict[dataset]['end_year'], 2019)

        # Search the variable in the netcdf file
        # ----- Return the subsetted data and the name of the variable ----- #
        def subset_year(file_name, var_tup, mode="obs"):
            # Open file
            with xr.open_dataset(file_name) as full_ds:
                # observation
                if mode == "obs":
                    # Using directly her name
                    main_var_name = next((var for var in list(
                        full_ds.data_vars.keys()) if var in ['R', 'T2m', 'Tm', 'Tx']), None)
                else:
                    # reanalysis
                    # Using the para_dict to hold
                    main_var_name = next((var for var in list(
                        full_ds.data_vars.keys()) if var in var_tup['list_name']), None)
                ds = full_ds[main_var_name]
            return ds.where((ds.time.dt.year >= start_year) & (ds.time.dt.year <= end_year), drop=True), main_var_name

        # USE FOR SINGLE INDEX
        # region
        # diso_df_list = []
        # for ind in list_ind:
        #     # Produce dataframe like this
        #     # City | Ind | Data_name
        #     # -----------------
        #     # ALUOI| T10p| 0.5
        #     # -----------------
        #     # HANOI| T10p| 0.6
        #     # ...............

        #     # name of the reanalysis and observational file
        #     # print(f'Processing {ind} for {dataset}')
        #     rean_ind = Data_ind+ind+f"_{dataset}.nc"
        #     obs_ind = Data_ind+ind+f"_obs.nc"

        #     # subset the rean and data in the same period
        #     # some rean ends before 2019
        #     (sub_rean, var_rean), (sub_obs, var_obs) = subset_year(
        #         rean_ind, param_dict[var], mode='rean'), subset_year(obs_ind, var)
        #     try:
        #         diso_df = pd.DataFrame({'name_station': list_name_city,
        #                                 'ind': [ind]*len(list_name_city),
        #                                 #    1. Subset wrt the position of the cities along lat, lon dimension (4 points => mean geo dimension)
        #                                 #    2. Extract the data from the reanalysis and observation (/w var_rean and var_obs)
        #                                 #    3. calculate the diso index for each city
        #                                 dataset: [DISO(sub_obs.where(sub_obs['no_station'] == no_city, drop=True),
        #                                                cut_mlp(dts=cut_points(sub_rean, pts_dict[list_name_city[i]], ['lat', 'lon'], full=True)['full'], dim_mean=['geo']))
        #                                           for i, no_city in enumerate(list_no_city)]})
        #     except Exception as e:
        #         raise ValueError("Stop here")
        #     # City that has data gap smaller then 5%
        #     diso_df_list.append(diso_df)
        # endregion

        # USE FOR MULTIPLE INDEX
        df_list = []
        df_obs_list = []
        for ind in list_ind:

            # Produce dataframe like this
            # City | Ind | Data_name
            # -----------------
            # ALUOI| T10p| 0.5
            # -----------------
            # HANOI| T10p| 0.6
            # ...............

            # name of the reanalysis and observational file
            # print(f'Processing {ind} for {dataset}')
            rean_nc = Data_ind+ind+f"_{dataset}.nc"
            obs_nc = Data_ind+ind+f"_obs.nc"

            # subset the rean and data in the same period
            # some rean ends before 2019

            print('test:', rean_nc, obs_nc)
            (sub_rean, var_rean), (sub_obs, var_obs) = subset_year(
                rean_nc, param_dict[var], mode='rean'), subset_year(obs_nc, var)
            for i, no_city in enumerate(list_no_city):
                #    1. Subset wrt the position of the cities along lat, lon dimension (4 points => mean geo dimension)
                #    2. Extract the data from the reanalysis and observation (/w var_rean and var_obs)
                #    3. calculate the diso index for each city
                obs_data = sub_obs.where(
                    sub_obs['no_station'] == no_city, drop=True)

                rean_data = cut_mlp(dts=cut_points(sub_rean, pts_dict[list_name_city[i]], [
                                    'lat', 'lon'], full=True)['full'], dim_mean=['geo'])

                sub_dict = {
                    **evaluate_compare(obs_data, rean_data, metrics_compare), **evaluate_single_all(rean_data)
                }
                obs_sub_dict = evaluate_single_all(rean_data)

                if i == 0:
                    main_dict = [sub_dict]
                    main_obs_dict = [obs_sub_dict]
                else:
                    main_dict.append(sub_dict)
                    main_obs_dict.append(obs_sub_dict)
            try:
                # Example at this step: result_df = [{'R': 1,'DISO':2},{'R': 3,'DISO':4},{'R': 5,'DISO':6]
                result_dict = {key: [d.get(key, None) for d in main_dict]  # Get the value of the key and put in list
                               for key in set().union(*main_dict)}  # Get the set of all the keys
                # Example at this step: result_df = {'R': [1,3,5],'DISO':[...]}
                result_obs_dict = {key: [d.get(key, None) for d in main_obs_dict]  # Get the value of the key and put in list
                                   for key in set().union(*main_obs_dict)}

                df = pd.DataFrame({'name_station': list_name_city,
                                   'ind': [ind]*len(list_name_city),
                                   **result_dict})
                obs_df = pd.DataFrame({'name_station': list_name_city,
                                       'ind': [ind]*len(list_name_city),
                                       **result_obs_dict})
            except Exception as e:
                print(ind, dataset)
                raise ValueError("Stop here")

            # City that has data gap smaller then 5%
            df_list.append(df)
            df_obs_list.append(obs_df)
        # Produce dataframe like this
            # City | Ind | Data_name
            # -----------------
            # ALUOI| T10p| 0.5
            # -----------------
            # HANOI| T10p| 0.6
            # ...............
            # ALUOI| T90p| 0.5
            # -----------------
            # HANOI| T90p| 0.6
            # ...............

        # ignore_index reset it from 0,1,2....
        df_all = pd.concat(df_list, ignore_index=True)
        df_obs_all = pd.concat(df_obs_list, ignore_index=True)
        # raise KeyboardInterrupt
        df_all.to_csv(ind_csv)
        df_obs_all.to_csv(obs_csv)
        return df_all
# ------------------------- Start to merge ------------------------- #
    dfs_merge = {}
    # Check whether the gen_df has been done
    for rean_name in list(rean_year_dict.keys()):
        rean_csv = f"{Data_csv}single/{rean_name}_{var}.csv"
        # Reach a dataframe of indices of individual reanalysis
        if check_non_exist(rean_csv):
            dfs_merge[rean_name] = gen_df(rean_name)
        else:
            dfs_merge[rean_name] = pd.read_csv(rean_csv)

    # Merge the indice dataframe
    for metric in metrics_compare+metrics_singular:
        # define of the merge file name
        merge_file_name = f'{Data_csv}ensemble/{var}_{metric}_merge.csv'
        pivot_file_name = f'{Data_csv}ensemble/{var}_{metric}_pivot.csv'

        if check_non_exist(merge_file_name):
            merge_df = pd.DataFrame()
        # Initiate the merge dataframe
            for i, rean_name in enumerate(list(rean_year_dict.keys())):
                # Take the subset of the dataframe and rename into name_station, ind , name_dataset
                merge_sub_df = dfs_merge[rean_name].loc[[
                    'name_station', 'ind', metric]].rename({metric: rean_name}, axis=1)

                # This can may flood the RAM but keep name_station and ind ENSURES the match of data
                if i == 0:
                    merge_df = merge_sub_df
                else:
                    # Join along these two collumns to align the reanalysis with each other
                    # Keep in mind the order of the reanlysis datasets
                    merge_df = merge_df.merge(
                        merge_sub_df, on=['name_station', 'ind'], how='outer')

            # Produce dataframe like this
            # City | Ind | Data1 | Data2 | Data3 | Data4
            # ------------------------------------------
            # ALUOI| T10p| 0.5   | ....  | ....  |.....
            # ------------------------------------------
            # HANOI| T10p| 0.5   | ....  | ....  |.....
            #  ------------------------------------------
            # ALUOI| T90p| 0.5   | ....  | ....  |.....
            #  ------------------------------------------
            # HANOI| T90p| 0.5   | ....  | ....  |.....
            # ...............

            merge_df.to_csv(merge_file_name)
        else:
            merge_df = pd.read_csv(merge_file_name)
            print(merge_df)

        # Pivot so indices became the horizontal index
        # city is the verticle index

        # e.g.
        #         Data1                     Data2                  Data3
        # Ind   T10p | T90p ..        | T10p | T90p ..     | T10p | T90p ..
        # City
        # ALUOI 0.5 | 0.5|...         | 0.5  | 0.5 | ...   | 0.5  | 0.5 | ...

        df_piv = merge_df.pivot_table(index='name_station', columns='ind')
        df_piv.to_csv(pivot_file_name)
        return df_piv

# ------------------------------------------------------------------ #
#                             PLOT FIGURE                            #
# ------------------------------------------------------------------ #

# This figure is a triangle heatmap


def plot_tri(var):
    def triangulation_for_triheatmap(M, N):
        # vertices of the little squares
        xv, yv = np.meshgrid(np.arange(-0.5, M), np.arange(-0.5, N))
        # centers of the little squares
        xc, yc = np.meshgrid(np.arange(0, M), np.arange(0, N))
        x = np.concatenate([xv.ravel(), xc.ravel()])
        y = np.concatenate([yv.ravel(), yc.ravel()])
        cstart = (M + 1) * (N + 1)  # indices of the centers

        trianglesN = [(i + j * (M + 1), i + 1 + j * (M + 1), cstart + i + j * M)
                      for j in range(N) for i in range(M)]
        trianglesE = [(i + 1 + j * (M + 1), i + 1 + (j + 1) * (M + 1), cstart + i + j * M)
                      for j in range(N) for i in range(M)]
        trianglesS = [(i + 1 + (j + 1) * (M + 1), i + (j + 1) * (M + 1), cstart + i + j * M)
                      for j in range(N) for i in range(M)]
        trianglesW = [(i + (j + 1) * (M + 1), i + j * (M + 1), cstart + i + j * M)
                      for j in range(N) for i in range(M)]
        return [Triangulation(x, y, triangles) for triangles in [trianglesN, trianglesE, trianglesS, trianglesW]]

    if check_non_exist(Data_csv+f"{var}_diso_pivot.csv"):
        df_piv = merge_df(var=var)
    else:
        df_piv = pd.DataFrame(Data_csv+f"{var}_diso_pivot.csv")
        M = len(df_piv.columns) // 4
        N = len(df_piv)
        values = [df_piv[dir] for dir in
                  list(rean_year_dict.keys())]  # these are the 4 column names in df

        triangul = triangulation_for_triheatmap(M, N)
        my_cmap = plt.get_cmap('RdYlBu_r')
        my_cmap.set_bad('#FFFFFF00')
        my_cmap.set_under('#FFFFFF00')
        norms = [plt.Normalize(0, 1) for _ in range(4)]
        fig, ax = plt.subplots(figsize=(25, 20))
        imgs = [ax.tripcolor(t, np.ravel(val), cmap=my_cmap, norm=norm, ec='white')
                for t, val, norm in zip(triangul, values, norms)]

        ax.tick_params(length=0)
        ax.set_xticks(range(M))
        ax.set_xticklabels(df_piv['cera'].columns)
        ax.set_yticks(range(N))
        ax.set_yticklabels(df_piv.index)
        ax.invert_yaxis()
        ax.margins(x=0, y=0)
        ax.set_aspect('equal', 'box')  # square cells
        plt.colorbar(imgs[0], ax=ax)
        plt.tight_layout()
        plt.show()
        fig.savefig(img_wd+var+"_triheatmap.png", format="png")


# Create merge & pivot pandaframe for indices
merge_df('R')
# merge_diso_df('T2m')

# Create the figure
# plot_tri('R')
# plot_tri('T2m')

"""
# ------------------------------------------------------------------ #
#    Only do this when you use directly the txt file for each city   #
# ------------------------------------------------------------------ #

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
            gdata_daily = gdata_val["T2m"].resample(time="1D").mean()
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
        continue

# ------------------------- test plot here ------------------------- #
# fig,axs=plt.subplot(subplot_kw=dict(projection=ccrs.PlateCarree()))
# map_pro = ccrs.PlateCarree()
# fig = plt.figure()
# ax = plt.subplot(111, projection=map_pro)

# print(time_period)
# plt.savefig
"""
