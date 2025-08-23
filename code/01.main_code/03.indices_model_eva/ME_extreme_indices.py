<<<<<<< HEAD
 
=======
# region : READme
"""
Climate Extreme Indices Calculator and Analyzer

This module provides tools for calculating, analyzing, and comparing climate extreme indices
from various reanalysis datasets and observations. It implements the ETCCDI climate indices
(https://www.wcrp-climate.org/etccdi) and provides statistical analysis capabilities.

Key Features:
-------------
1. Generation of base files for climate indices calculation
2. Calculation of temperature and precipitation extreme indices
3. Comparison of indices between reanalysis datasets and observations
4. Statistical analysis and visualization of climate extremes
5. Trend analysis and significance testing

Main Functions:
--------------
check_non_exist(name, force=False):
    Check if a file doesn't exist.

Run_cdo_command(command, input, output, args=None, force=False):
    Run CDO commands with error handling.

Run_cdo_base(command, input, output, args=None):
    Run CDO commands for base file generation.

gen_base(var, dataset, start=1961, end=1990, percentile=None, window='5', *args):
    Generate base files for indices calculation.

    Parameters:
    - var: Variable type ('tp', 'max', 'min', 'vwnd', 'uwnd')
    - dataset: Dataset name ('cera', 'era', 'era5', 'noaa', 'obs')
    - start/end: Base period years
    - percentile: List of percentiles to calculate
    - window: Running window size for percentile calculation

Run_Gen_base(dataset='cera'):
    Run base file generation for a specific dataset.

class gen_ind:
    Class for generating climate indices.

    Methods:
    - __call__(ind, dataset): Calculate a specific index for a dataset
    - gen_con_day(): Generate consecutive dry/wet days (CDD/CWD)
    - gen_compare_thres(): Generate threshold-based indices (R10mm, SU25, etc.)
    - gen_rxnday(): Generate n-day maximum precipitation (Rx1day, Rx5day)
    - gen_sdii(): Generate Simple Daily Intensity Index
    - gen_prcptot(): Generate total precipitation on wet days
    - gen_compare_rptot(): Generate percentile-based precipitation (R95p, R99p)
    - gen_compare_pctlday(): Generate percentile-based temperature days
    - gen_min(): Generate minimum temperature indices
    - gen_max(): Generate maximum temperature indices
    - gen_dtr(): Generate Diurnal Temperature Range
    - gen_temp_si(): Generate temperature spell indices (WSDI, CSDI)

base_fu():
    Generate base files for all datasets using parallel processing.

ind_fu():
    Calculate indices for all datasets using parallel processing.

merge_metric_extreme(var, metric_stop=None, rerun=None):
    Merge and process climate indices data from multiple datasets.

    Parameters:
    - var: Variable to process ('R' for precipitation, 'T2m' for temperature)
    - metric_stop: Optional stopping point for metrics calculation
    - rerun: Controls which parts to regenerate:
        - 'compare': Regenerate comparison metrics
        - 'sing': Regenerate single-dataset metrics
        - 'both': Regenerate both comparison and single-dataset metrics
        - 'merge': Regenerate merged dataframe from existing metrics
        - 'obs': Regenerate observation-only metrics
        - None: Use existing files if available

plot_tri(var, metric):
    Create triangular plots for visualizing metrics across datasets.

Global Variables:
----------------
Data_wd: Base data directory
Data_ind: Directory for indices files
Data_csv: Directory for CSV files
Data_nc: Directory for NetCDF files
temp_tuple: Tuple of temperature indices
rain_tuple: Tuple of precipitation indices
param_dict: Dictionary of parameter metadata
metrics_compare: List of comparison metrics
metrics_singular: List of single-dataset metrics

Usage Examples:
--------------
# Generate base files for all datasets
base_fu()

# Calculate indices for all datasets
ind_fu()

# Merge and analyze precipitation indices
merge_metric_extreme('R', rerun='both')

# Merge and analyze temperature indices
merge_metric_extreme('T2m', rerun='both')

# Create visualization for specific metrics
plot_tri('R', 'DISO')
plot_tri('T2m', 'Taylor_score')

Notes:
------
- This module requires CDO (Climate Data Operators) to be installed
- The module uses parallel processing for efficiency
- Output files are stored in directories defined by environment variables
- The module follows the ETCCDI climate indices definitions

References:
----------
1. Klein Tank, A.M.G., et al. (2009): Guidelines on Analysis of extremes in a changing
   climate in support of informed decisions for adaptation. Climate Data and Monitoring
   WCDMP-No. 72, WMO-TD No. 1500, 56pp.
2. Zhang, X., et al. (2011): Indices for monitoring changes in extremes based on daily
   temperature and precipitation data. WIREs Climate Change, 2: 851-870.
"""
# endregion
>>>>>>> c80f4457 (First commit)

# ------------------------------------------------------------------ #
#                          PRELIMINARY STUFF                         #
# ------------------------------------------------------------------ #

# ----------------------- Import module ---------------------------- #
# Standard library imports
import os
import subprocess
import shlex
import concurrent.futures as con_fu
import time
import datetime as dt
import warnings
import json
from typing import Union, Optional, List, Tuple, Dict, Literal, Callable

# Data manipulation and analysis
import pandas as pd
import numpy as np
# import scipy.stats as sst
# from scipy.optimize import curve_fit
# from sklearn.metrics import r2_score
# from sklearn.decomposition import PCA
# from sklearn.utils import check_random_state
import xarray as xr

# Plotting and visualization
import matplotlib
import matplotlib.style
import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# import matplotlib.cm as cm
# import matplotlib.lines as mlines
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap
from matplotlib.tri import Triangulation
# from matplotlib.animation import FuncAnimation
# from matplotlib.image import imread
# from matplotlib.patches import Wedge
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
# from matplotlib.ticker import MultipleLocator
import seaborn as sns

# Geospatial libraries
# import geopandas as gpd
# import shapefile as shp
# from shapely.geometry import MultiPolygon
# from cartopy.feature import ShapelyFeature
# from cartopy.io.shapereader import Reader
# from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
# import cartopy.feature as cfeature
# import cartopy.crs as ccrs

# Other libraries
# from tkinter import ttk
# from cf_units import decode_time
<<<<<<< HEAD
from cdo import *
from my_junk import *


=======
from cdo import Cdo
from my_junk import *
from constant import *
import city_sort_draft as csort

Cache = DataCache()
>>>>>>> c80f4457 (First commit)
cdo = Cdo()
# This prohibits that existing files are created a second time
cdo.forceOutput = False

warnings.filterwarnings("ignore")

# endregion
# ----------------------- Import plot module ----------------------- #
# region
# import metpy.calc as mpcalc
# from metpy.units import units

# Ignore FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning)

params = {
<<<<<<< HEAD
    'axes.titlesize': 40,
    'axes.labelsize': 30,
    'font.size': 50,
    'font.family': 'cmss10',
    'legend.fontsize': 30,
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
    'xtick.major.size': 10,
    'ytick.major.size': 10,
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
=======
'axes.titlesize': 40,
'axes.labelsize': 30,
'font.size': 50,
'font.family': 'cmss10',
'legend.fontsize': 30,
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
'xtick.major.size': 10,
'ytick.major.size': 10,
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
>>>>>>> c80f4457 (First commit)
}
plt.clf()
matplotlib.rcParams.update(params)
# plt.style.use('ggplot')
# endregion

<<<<<<< HEAD
# --------------------- Data path and constant --------------------- #
# region

rain_csv = "/work/users/tamnnm/code/01.main_code/01.city_list_obs/city/rain_city.txt"
temp_csv = "/work/users/tamnnm/code/01.main_code/01.city_list_obs/city/temp_city.txt"
rain_df = pd.read_csv(rain_csv, names=[
                      'name_station', 'first_year', 'last_year', 'lon', 'lat', 'elev', 'appendix'])
temp_df = pd.read_csv(temp_csv, names=[
                      'name_station', 'first_year', 'last_year', 'lon', 'lat', 'elev', 'appendix'])

Data_wd = os.getenv("data")
Code_wd = os.path.join(os.getenv("mcode"), "01.city_list_obs/")
Data_ind = os.path.join(Data_wd, "wrf_data/netcdf/para/indices/")
Data_csv = os.path.join(Data_wd, "wrf_data/netcdf/para/csv_file/")
Data_nc = os.path.join(Data_wd, "wrf_data/netcdf/para/")
Data_raw_2019 = os.path.join(Data_wd, "dat_daily_1960_2019/")
Data_nc_2019 = os.path.join(Data_wd, "obs/UPDATE_METEO/")
Data_obs_list = os.path.join(Code_wd, "city/")
Data_obs_pts = os.path.join(Code_wd, "city_pts/")
suffix_2019 = '_1960_2019.txt'
img_wd = os.getenv("img")
#print(img_wd)
os.chdir(Data_nc)
# Data_2023 = Data_wd+"dat_daily_1961_2021"
indice_dict = {167: {'min': ('Tnn', 'Tnx', 'Tn10p', 'Tn90p', 'CSDI', 'TN20', 'TN15'), 'max': ('Txx', 'Txn', 'Tx90p', 'Tx10p', 'WSDI', 'SU25', 'SU35'), 'other': ('DTR',)},
               228: ('CDD', 'CWD', 'R10mm', 'R20mm', 'R50mm', 'R1mm', 'R5mm', 'Rx1day', 'Rx5day', 'SDII', 'PRCPTOT', 'R99p', 'R95p')}
temp_tuple = ('Tnn', 'Tnx', 'Txx', 'Txn', 'Tn10p', 'Tn90p', 'Tx90p', 'Tx10p',
              'CSDI', 'WSDI', 'SU25', 'SU35', 'TN15', 'TN20', 'DTR')
rain_tuple = ('R1mm', 'R5mm', 'R10mm', 'R20mm',
              'R50mm', 'Rx1day', 'Rx5day', 'R99p', 'R95p', 'CDD', 'CWD', 'SDII', 'PRCPTOT')

# ? Use to sort the city later
# Mapping of categorical values to numerical values
temp_mapping = {value: idx + 1 for idx, value in enumerate(temp_tuple)}
rain_mapping = {value: idx + 1 for idx, value in enumerate(rain_tuple)}

# Reverse the mapping dictionaries
reverse_temp_mapping = {v: k for k, v in temp_mapping.items()}
reverse_rain_mapping = {v: k for k, v in rain_mapping.items()}


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
metrics_compare = ['mpe', 'mape', 'R', 'nme',
                   'nmae', 'nrmse', 'DISO', 'Taylor_score']
metrics_singular = list(METRICS_SINGULAR.keys())


script_path = script_path = os.path.dirname(os.path.abspath(__file__))
json_path = script_path+"/constant.json"

def_start_year = 1961
def_end_year = 2019
# endregion

=======
>>>>>>> c80f4457 (First commit)
# ------------------------- optional choice ------------------------ #
option = "else"
# "time" = plot timeseries
# "indices" = calculate the indices

# ------------------------------------------------------------------ #
#                        CALCULATING THE INDEX                       #
# ------------------------------------------------------------------ #

<<<<<<< HEAD
# ------------------------ check indices ----------------------- #


def check_non_exist(name, force=False):
    return False if os.path.exists(name) else True

=======
>>>>>>> c80f4457 (First commit)
# ------------------------- Run cdo command ------------------------ #

# Run the cdo command directectly


<<<<<<< HEAD
def run_cdo_command(command: Callable, input: str, output: str, args: Optional[List] = None, force=False):
    if args is None:
        args = {}
    if force:
        command(*args, input=input, output=output, options='-L')
    else:
        if check_non_exist(output):
            command(*args, input=input, output=output, options='-L')
            #print(output)
    return

# Run the cdo command for base file (must be sequential)


def run_cdo_base(command: str, input: str, output: str, args: Optional[List] = None):
=======
def Run_cdo_command(command: Callable, input: str, output: str, args: Optional[List] = None, force=False):
    if args is None:
        args = {}
    # Add memory optimization options
    base_options = '-L -f nc4 -z zip_6'
    if force:
        command(*args, input=input, output=output, options=base_options)
    else:
        if check_non_exist(output):
            command(*args, input=input, output=output, options=base_options)
    return
xz
# Run the cdo command for base file (must be sequential)


def Run_cdo_base(command: str, input: str, output: str, args: Optional[List] = None):
>>>>>>> c80f4457 (First commit)
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


<<<<<<< HEAD
def gen_base(var: Optional[Literal["tp", "max", "min", "vwnd", "uwnd"]], dataset: Optional[Literal["cera", "era", "era5", "noaa", "obs"]],
=======
def gen_base(var: Optional[Literal["R", "Tx", "Tn", "Vm", "Vx"]], dataset: Optional[Literal["cera", "era", "era5", "noaa", "obs"]],
>>>>>>> c80f4457 (First commit)
             start: Optional[int] = 1961, end: Optional[int] = 1990, percentile: list = None, window='5', *args):
    # create base file
    no = param_dict[var]['no']
    fol = f'para_{no}/'

    def full_name(name, fol=fol):
        return fol+name

    # input file, special case for temp(prefix of min/max)
<<<<<<< HEAD
    ifile = f'{dataset}_{no}.nc' if no != 167 else f'{var}_{dataset}_{no}.nc'
=======
    ifile = f'{var}_{dataset}.nc'
>>>>>>> c80f4457 (First commit)

    # output file, base period
    ofile = 'b_'+ifile
    full_ifile, full_ofile = full_name(ifile), full_name(ofile)

    if no == 228:
<<<<<<< HEAD
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
=======
        # TODO: Run this again
        # Number of wet days
        Run_cdo_command(
            cdo.yearsum, input=f' -gec,{threshold[dataset]} {full_ifile}', output=f'indices/no_wet_{dataset}.nc')
        # Total precipitation of w
        Run_cdo_command(
            cdo.yearsum, input=f'-mul {full_ifile} -gec,{threshold[dataset]} {full_ifile}', output=f'indices/PRCPTOT_{dataset}.nc')

    if check_non_exist(full_ofile):
        time_range = f'{start-1}-12-30,{end}-01-02' if dataset != 'obs' else f'{start}-01-01,{end}-12-31'

        # Extract the start year of obs to decide
        if no == 228:
            Run_cdo_base(
                'seldate', input=f'-setctomiss,0 -mul {full_ifile} -gec,{threshold[dataset]} {full_ifile}', output=full_ofile, args=[time_range])
        else:
            Run_cdo_base(
>>>>>>> c80f4457 (First commit)
                'seldate', input=full_ifile, output=full_ofile, args=[time_range])
        #print('Base file:', full_ofile)

    if param_dict[var]['type'] == 'extreme':
        if check_non_exist(full_ofile):
            raise ValueError("Mission failed on"+full_ofile)

        if percentile:
            # Input base file is the output file for the percentile
            ibfile = full_ofile
            pminfile, pmaxfile = full_name(
                f'bp_min_{ifile}'), full_name(f'bp_max_{ifile}')
            # Minimum values
<<<<<<< HEAD
            run_cdo_base('ydrunmin', ibfile, pminfile, [window])
            #print('Min: ', pminfile)
            # Maximum values
            run_cdo_base('ydrunmax', ibfile, pmaxfile, [window])
=======
            Run_cdo_base('ydrunmin', ibfile, pminfile, [window])
            #print('Min: ', pminfile)
            # Maximum values
            Run_cdo_base('ydrunmax', ibfile, pmaxfile, [window])
>>>>>>> c80f4457 (First commit)
            #print('Max:', pmaxfile)
            # n-window running percentile of base value
            for rp in percentile:
                opfile = full_name(f'b{rp}p_{ifile}')
                # Quantile file
<<<<<<< HEAD
                run_cdo_base(
                    'ydrunpctl', input=f'{ibfile} {pminfile} {pmaxfile}', output=opfile, args=[rp, window, 'pm=r8'])
                #print('Percentile: ', opfile)
    return

def run_gen_base(dataset):
    gen_base(var='R', start=1961, end=1990, dataset=dataset,
             percentile=[95, 99], window='5')
    gen_base(var='min', start=1961, end=1990,
             dataset=dataset, percentile=[10, 90], window='5')
    gen_base(var='max', start=1961, end=1990,
             dataset=dataset, percentile=[10, 90], window='5')
=======

                Run_cdo_base(
                    'ydrunpctl', input=f'{ibfile} {pminfile} {pmaxfile}', output=opfile, args=[rp, window])
                #print('Percentile: ', opfile)
    return

def Run_Gen_base(dataset):
    gen_base(var='R', start=1961, end=1990, dataset=dataset,
            percentile=[95, 99, 99.9], window='5')
    gen_base(var='Tn', start=1961, end=1990,
            dataset=dataset, percentile=[10, 90, 99], window='5')
    gen_base(var='Tx', start=1961, end=1990,
            dataset=dataset, percentile=[10, 90, 99], window='5')
>>>>>>> c80f4457 (First commit)
    return

# --------------------- Getting the input name --------------------- #


def get_ifiles(ind, dataset):
    """
    This function generates file paths for the given index and dataset.

    Parameters:
    ind (str): The index for which file paths are to be generated.
    dataset (str): The dataset for which file paths are to be generated.

    Returns:
    tuple: A tuple containing the folder path, minimum file path, and maximum file path.
            or
            A tuple of folder path, input file, None
<<<<<<< HEAD
           The maximum file path may be None if it's not applicable.
=======
            The maximum file path may be None if it's not applicable.
>>>>>>> c80f4457 (First commit)

    The function first finds the number associated with the given index from the `indice_dict` dictionary.
    It then constructs the folder path and the paths for the minimum and maximum files.
    If the number is 167 (temperature), it checks which category the index falls into ('other', 'min', or 'max')
    and returns the appropriate file paths.
    If the number is not 167 (others), it returns the folder path and the path for the dataset file,
    with None as the maximum file path.
    """

    # List through the dictionary of indices and check the variables corresponding with it
    # e.g. CDD -> 228
<<<<<<< HEAD
    no = next((no for no in list(indice_dict.keys()) if ind in indice_dict[no]
               or
               any(ind in indice_dict[no][sub_no] for sub_no in list(indice_dict[no].keys()))),
              None)

    # folder for each variable
    fol = f'para_{no}/'
    fmin = f'min_{dataset}_{no}.nc'
    fmax = f'max_{dataset}_{no}.nc'
    if no == 167:
=======
    def find_indice_no(ind):
        for no, values in indice_dict.items():
            if isinstance(values, dict):
                if any(ind in sublist for sublist in values.values()):
                    return no
            elif ind in values:
                return no
        return None

    no = find_indice_no(ind)  # Returns 228

    # folder for each variable
    fol = f'para_{no}/'
    if no == 167:
        fmin = f'Tn_{dataset}.nc'
        fmax = f'Tx_{dataset}.nc'
>>>>>>> c80f4457 (First commit)
        if ind in indice_dict[no]['other']:
            return fol, fmin, fmax
        if ind in indice_dict[no]['min']:
            return fol, fmin, None
        if ind in indice_dict[no]['max']:
            return fol, fmax, None
    else:
<<<<<<< HEAD
        return fol, f'{dataset}_{no}.nc', None
=======
        var = next((k for k in param_dict if param_dict[k]['no'] == no), None)
        return fol, f'{var}_{dataset}.nc', None
>>>>>>> c80f4457 (First commit)

# ------------------------ Generating index ------------------------ #


class gen_ind:
    def __init__(self):
        self.ifile = None

    def __call__(self, ind, dataset):
        self.ind = ind
        self.dataset = dataset
        self.fol, iminfile, imaxfile = get_ifiles(ind, dataset)
        # Must do this or it will raise Error as string cannot go with None
        self.iminfile = self.fol+iminfile if iminfile is not None else None
        self.imaxfile = self.fol+imaxfile if imaxfile is not None else None
        ifile = iminfile or imaxfile
        self.ifile = self.fol+ifile
        self.ofile = f'indices/{ind}_{dataset}.nc'
<<<<<<< HEAD
=======
        self.temp_file = f'indices/temp_{ind}_{dataset}.nc'

>>>>>>> c80f4457 (First commit)
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
        elif self.ind in ['R10mm', 'R20mm', 'R50mm', 'R1mm', 'R5mm', 'SU35', 'TN15', 'SU25', 'TN20']:
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
<<<<<<< HEAD
        else:
            return
        run_cdo_command(self.command, self.ifile, self.ofile)
=======
        elif self.ind in ['Tm', 'Tnm', 'Txm']:
            self.gen_mean() #TODO: Update this
        else:
            return
        Run_cdo_command(self.command, self.ifile, self.ofile)
        print("Generating index:", self.ind, "for dataset:", self.dataset)
>>>>>>> c80f4457 (First commit)
        return

    def gen_con_day(self):
        if self.ind == "CDD":
            method = 'ltc'
        if self.ind == "CWD":
            method = 'gec'
<<<<<<< HEAD
        self.command = cdo.yearmax
        self.ifile = f'-consects -{method},1 {self.ifile}'
=======

        Run_cdo_command(
            cdo.consects, input=f'-{method},{threshold[self.dataset]} {self.ifile}', output=self.temp_file)
        self.command = cdo.yearmax
        self.ifile = self.temp_file
>>>>>>> c80f4457 (First commit)

    def gen_compare_thres(self, method, thres=1):
        # Rnmm ,SU25, TN20
        if 'mm' in self.ind:
            method = 'gec'
            thres = self.ind[1:-2]
        elif 'TN' in self.ind or 'SU' in self.ind:
<<<<<<< HEAD
            thres = 273.15 + int(self.ind[2:5])
=======
            thres = int(self.ind[2:5])
>>>>>>> c80f4457 (First commit)
        else:
            thres = 1
        method = 'gtc'
        # Convert to Kelvin
        #print(thres)

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
        # #print(self.ifile, self.ibfile)
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
<<<<<<< HEAD
        self.command = cdo.yearsum
        self.ifile = f'-gec,6 -consects -{method},0 -sub {self.ifile} {self.ibfile}'
=======

        Run_cdo_command(
            command = cdo.consects, input=f"-{method},0 -sub {self.ifile} {self.ibfile}", output=self.temp_file)
        self.command = cdo.yearsum
        self.ifile = f'-gec,6 {self.temp_file}'
>>>>>>> c80f4457 (First commit)


# Initialize the class
gen_class = gen_ind()

# ----------------- MUST RUN FILE BASE FIRST OF ALL ---------------- #
# region

<<<<<<< HEAD

def base_fu():
    with con_fu.ProcessPoolExecutor() as executor:
        futures = []
        # Reanalysis
        for rean_name in list(rean_year_dict.keys()):
            futures += [executor.submit(run_gen_base, dataset=rean_name)]
        # Observation
        futures += [executor.submit(run_gen_base, dataset='obs')]
=======
def base_fu():
    with con_fu.ProcessPoolExecutor() as executor:
        futures = []
        # # Reanalysis
        for rean_name in list(rean_year_dict.keys()):
            futures += [executor.submit(Run_Gen_base, dataset=rean_name)]
        # Observation
        futures += [executor.submit(Run_Gen_base, dataset='obs')]
>>>>>>> c80f4457 (First commit)
    con_fu.wait(futures)
    #print("Finish creating base file")
    return
# endregion

# -------------------------- RUN THE TEST -------------------------- #


def ind_fu():
<<<<<<< HEAD
    with con_fu.ProcessPoolExecutor() as executor:
        futures = []
        # Reanalysis
        # ? Can change this: TUPLE
        # tuple = temp_tuple + rain_tuple
        tuple = ('TN20', 'TN15',)
        for ind in tuple:
            for rean_name in list(rean_year_dict.keys()):
=======
    max_workers = min(20, os.cpu_count() - 1 or 1)
    with con_fu.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        # Reanalysis
        # ? Can change this: TUPLE
        tuple = rain_tuple + temp_tuple
        for ind in tuple:
            for rean_name in ['era5']: #list(rean_year_dict.keys()):
>>>>>>> c80f4457 (First commit)
                futures += [executor.submit(gen_class,
                                            ind=ind, dataset=rean_name)]
            # Observation
            # #print(ind)
<<<<<<< HEAD
            futures += [executor.submit(gen_class, ind=ind, dataset='obs')]
    # con_fu.wait(futures)
    #print("Finish creating indices file")

# Manual way
# for ind in ('R5mm','SU35','TN15'):
#     gen_class(ind, 'obs')


# Auto way
# Run indices file generator
# DONE
# #print("Check base file")
# base_fu()
# #print("Check indices file")
# ind_fu()
=======
            # futures += [executor.submit(gen_class, ind=ind, dataset='obs')]
    con_fu.wait(futures)

    # tuple= rain_tuple + temp_tuple


    # for ind in tuple:
    #     for rean_name in list(rean_year_dict.keys()):
    #         gen_class(ind=ind, dataset=rean_name)
    #     # Observation
    #     #print(ind)
    #     gen_class(ind=ind, dataset='obs')

    print("Finish creating indices file")
>>>>>>> c80f4457 (First commit)


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

<<<<<<< HEAD

def merge_df(var, metric_stop=None, rerun: Optional[Literal['compare', 'sing', 'both', 'merge','obs']] = None):
    list_ind = rain_tuple if var == 'R' else temp_tuple

    # ? Change this incase you run additional
    # ------ Generate the diso index for each reaannalysis dataset ----- #

    def gen_df(dataset, rerun=rerun):
        #print("Generating dataframe for", dataset, rerun)
        # define name of file of indices
        obs_csv = f'{Data_csv}single/obs_{var}.csv'
        compare_csv = f'{Data_csv}single/obs_{dataset}_{var}.csv'
        # ?: I run additional index
        sing_csv = f'{Data_csv}single/{dataset}_{var}.csv'
        all_csv = f'{Data_csv}single/all_{dataset}_{var}.csv'
        compare_skip, sing_skip, obs_skip = False, False, False
        
        if not check_non_exist(obs_csv):
            obs_skip = True if rerun not in ['obs', 'both'] else False
            print("Skip obs for",var) if obs_skip else None
            df_obs_all = pd.read_csv(obs_csv)
        if not check_non_exist(compare_csv):
            compare_skip = True if rerun not in ['compare', 'both'] else False
            #print("Skip compare") if compare_skip else None
            df_compare_all = pd.read_csv(compare_csv)
        if not check_non_exist(sing_csv):
            sing_skip = True if rerun not in ['sing', 'both'] else False
            #print("Skip single") if sing_skip else None
            df_sing_all = pd.read_csv(sing_csv)
        if sing_skip and compare_skip and obs_skip:
            if check_non_exist(all_csv):
                return pd.read_csv(all_csv)

            df_all = pd.merge(df_compare_all, df_sing_all,
                              on=['name_station', 'ind'])
            df_all.to_csv(all_csv, index=False)
            return df_all
    
        # DONT CHANGE
        # extract the points for a specific dataset from the text file
        pts_dict = city_ds_pts(dataset)  # dict
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
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                list_no_city = data[f"no_{var}_city"]
                list_name_city = data[f"name_{var}_city"]
        except:
            # Read the netcdf observational file
            full_ds = xr.open_dataset(
                Data_nc_2019+"OG_"+var+"_daily_1960_2019.nc")
            # Extract city with data gap smaller than 5% + Available time >=20 years (before 1990)
            sel_ds = full_ds.where((full_ds['data_gap'] <= 0.05)  # data gap < 0.05
                                   # more than 20 years
                                   & (full_ds['end_year']-full_ds['start_year'] >= 20)
                                   & (full_ds['start_year'] <= 1970), drop=True)  # start year must start before 1970

            # list name of cities above
            list_no_city = sel_ds['no_station'].values
            list_name_city = [item.decode('utf-8')
                              for item in sel_ds['name_station'].values]
            with open(json_path, 'r') as f:
                data = json.load(f)
                data[f"no_{var}_city"] = list_no_city
                data[f"name_{var}_city"] = list_name_city

            with open(json_path, "w") as f:
                json.dump(data, f, indent=4)
=======
class MetricExtreme():
    """
        Initialize the MetricExtreme class.

        Parameters
        ----------
        var : str
            Variable to process ('R' for precipitation or 'T2m' for temperature)
        list_ind_option : 'Default' or List[str], optional
            Controls which indices to process
        metric_stop : str, optional
            If provided, only processing this metric is calculated. If not existed, calculate the whole group : sing/compare metrics
        rerun : str, optional
            - 'compare': Regenerate comparison metrics
            - 'sing': Regenerate single metrics for reanalysis
            - 'both': Regenerate both comparison and single-dataset metrics
            - 'obs': Regenerate single metrics for observation
            - 'Defau;t': Use existing files if available, do not rerun
            Controls which parts of the analysis to regenerate
        seasonal : str or list, optional
            - Default: No process
            - 'all': Process all seasons
            - 'annual': Process annual data only
            - 'seasonal': Process seasonal data only
            - 'DJF/JJA/....': Process special season
            - List[str]: Process specific seasons (e.g., ['DJF', 'JJA'])
            Controls which seasons to process
    """
    def __init__(self):
        """Initialize with no required parameters"""
        self._configured = False
        self.var = None
        self.list_ind_option = 'Default'
        self.season_option = 'Default'
        self.metric_stop = None
        self.rerun = 'Default'

        # Initialize state flags
        self.compare_skip = False
        self.sing_skip = False
        self.obs_skip = False
        self.season_skip = False

    def configure(self, var: Optional[str],
        list_ind_option: Optional[Union[Literal['Default'], List[str], str]] = 'Default',
        season_option: Optional[Union[Literal['Default'],Literal['all'],Literal['annual'], Literal['seasonal'], Literal['DJF'], Literal['MAM'],
                                Literal['JJA'], Literal['SON'], List[str]]] = 'Default',
        metric_stop=None,
        rerun: Optional[Literal['Default','compare', 'sing', 'both', 'obs', 'all']] = 'Default'
        ):

        # Beginning set-up
        self.var = var
        self.metric_stop = metric_stop
        self.rerun = rerun

        self.compare_skip, self.sing_skip, self.obs_skip, self.season_skip = False, False, False, False

        self._setup_season(season_option)
        self._setup_ind(list_ind_option)
        self._setup_metric(rerun)

        # Search the variable in the netcdf file
        # ----- Return the subsetted data and the name of the variable ----- #
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                self.list_no_city = data[f"no_{self.var}_city"]
                self.list_name_city = data[f"name_{self.var}_city"]
        except:
            # Read the netcdf observational file
            full_ds = xr.open_dataset(
                Data_nc_2019+"OG_"+self.var+"_daily_1960_2019.nc")
            # Extract city with data gap smaller than 5% + Available time >=20 years (before 1990)
            sel_ds = full_ds.where((full_ds['data_gap'] <= 0.05)  # data gap < 0.05
                                # more than 20 years
                                & (full_ds['end_year']-full_ds['start_year'] >= 20)
                                & (full_ds['start_year'] <= 1970), drop=True)  # start year must start before 1970

            # list name of cities above
            self.list_no_city = sel_ds['no_station'].values
            self.list_name_city = [item.decode('utf-8')
                            for item in sel_ds['name_station'].values]

            # Save the list of cities
            with open(json_path, 'r') as f:
                data = json.load(f)
                data[f"no_{self.var}_city"] = self.list_no_city
                data[f"name_{self.var}_city"] = self.list_name_city

            with open(json_path, "w") as f:
                json.dump(data, f, indent=4)
        return self

    def __call__(self, **kwargs):
        """Configure and process in one step (if kwargs provided), or process with current config"""
        if kwargs:
            return self.configure(**kwargs).process()
        return self.process()

    def _setup_season(self, season_option):
        self.season = None

        # Set-up seasonal cases
        if season_option == 'Default':
            self.season_skip = True
        elif season_option == 'all':
            self.seasons_to_process= valid_season
        elif season_option == 'annual':
            self.seasons_to_process = ['annual']
        elif season_option == 'seasonal':
            self.seasons_to_process = ['DJF', 'MAM', 'JJA', 'SON']
        elif isinstance(season_option, list):
            for season in season_option:
                if season not in valid_season:
                    raise ValueError(f"Invalid season: {season}")
            self.seasons_to_process = season_option
        else:
            raise ValueError("Invalid season option")

    def _setup_ind(self, list_ind_option):
        # Set-up list indices
        if list_ind_option == "Default":
            self.list_ind = rain_tuple if self.var == 'R' else temp_tuple
            self.ind_suffix = ''
        elif isinstance(list_ind_option, list):
            self.list_ind = list_ind_option
            self.ind_suffix = '_custom'
        elif isinstance(list_ind_option, str):
            self.list_ind = [list_ind_option]
            if len(list_ind_option.split('_')) > 1:
                self.ind_suffix = "_" + list_ind_option.split('_')[-1]
            else:
                self.ind_suffix = '_sup'
        else:
            raise ValueError("Invalid list_ind_option. It must be 'Default', a list of strings, or a single string.")

    def _setup_metric(self, rerun):
        # Set-up metric to process
        if rerun == 'compare':
            self.list_metric = metrics_compare
        elif rerun == 'sing':
            self.list_metric = metrics_singular
        elif rerun == 'obs': # Only run indices of observation
            self.list_metric = metrics_singular
        else:
            self.list_metric = metrics_compare + metrics_singular

    def _setup_constant_dataset(self):
        # Set-up path of metric files
        self.compare_csv = f'{Data_csv}single/obs_{self.dataset}_{self.sing_suffix}.csv'
        self.sing_csv = f'{Data_csv}single/{self.dataset}_{self.sing_suffix}.csv'
        self.all_csv = f'{Data_csv}single/all_{self.dataset}_{self.sing_suffix}.csv'
        self.obs_csv = f'{Data_csv}single/obs_{self.var}_{self.sing_suffix}.csv'
        # Define the start & end year for each dataset since not all of them end by 2019
        self._setup_year()

    def _setup_year(self):
            # This also use to crop out the observational datasets
        self.start_year = max(rean_year_dict[self.dataset]['start_year'], 1960)
        self.end_year = min(rean_year_dict[self.dataset]['end_year'], 2019)

    def GenMetric(self):

        """
        Generate metrics for a specific dataset and season.

        Parameters
        ----------
        dataset : str
            Dataset name (e.g., 'era5', 'cera', 'obs')
        season : str, optional
            Season to process ('annual', 'DJF', 'MAM', 'JJA', 'SON')

        Returns
        -------
        pandas.DataFrame
            DataFrame containing metrics for the specified dataset and season
        """

        print(f"Generating metric dataframe for {self.dataset} with {self.var}", f' for {self.season}' if self.season else '')

        if self.sing_skip and self.compare_skip and self.obs_skip:
            if check_exist(self.all_csv):
                return pd.read_csv(self.all_csv)
            else:
                if check_exist(self.obs_csv):
                    self.obs_skip = True if self.rerun not in ['obs', 'all'] else False
                    print("Run obs") if not self.obs_skip else None
                    df_obs_all = pd.read_csv(self.obs_csv)
                if check_exist(self.compare_csv):
                    self.compare_skip = True if self.rerun not in ['compare', 'both', 'all'] else False
                    print("Run compare") if not self.compare_skip else None
                    df_compare_all = pd.read_csv(self.compare_csv)
                if check_exist(self.sing_csv):
                    self.sing_skip = True if self.rerun not in ['sing', 'both', 'all'] else False
                    print("Run single") if not self.sing_skip else None
                    df_sing_all = pd.read_csv(self.sing_csv)
                if check_exist(self.compare_csv) and check_exist(self.sing_csv) and self.compare_skip and self.sing_skip:
                    # If both csv files exist, merge them
                    df_all = pd.merge(df_compare_all, df_sing_all,
                                    on=['name_station', 'ind'])
                    df_all.to_csv(self.all_csv, index=False)
                    return df_all

        # DONT CHANGE
        # extract the points for a specific dataset from the text file
        pts_dict = city_ds_pts(self.dataset)  # dict
        # subset the data to 1961-2019

        # Set up the start and end year
        self._setup_constant_dataset()
>>>>>>> c80f4457 (First commit)

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
        #     # #print(f'Processing {ind} for {dataset}')
        #     rean_ind = Data_ind+ind+f"_{dataset}.nc"
        #     obs_ind = Data_ind+ind+f"_obs.nc"

        #     # subset the rean and data in the same period
        #     # some rean ends before 2019
        #     (sub_rean, var_rean), (sub_obs, var_obs) = subset_year(
        #         rean_ind, param_dict[var], mode='rean'), subset_year(obs_ind, var)
        #     try:
<<<<<<< HEAD
        #         diso_df = pd.DataFrame({'name_station': list_name_city,
        #                                 'ind': [ind]*len(list_name_city),
=======
        #         diso_df = pd.DataFrame({'name_station': self.list_name_city,
        #                                 'ind': [ind]*len(self.list_name_city),
>>>>>>> c80f4457 (First commit)
        #                                 #    1. Subset wrt the position of the cities along lat, lon dimension (4 points => mean geo dimension)
        #                                 #    2. Extract the data from the reanalysis and observation (/w var_rean and var_obs)
        #                                 #    3. calculate the diso index for each city
        #                                 dataset: [DISO(sub_obs.where(sub_obs['no_station'] == no_city, drop=True),
<<<<<<< HEAD
        #                                                cut_mlp(dts=cut_points(sub_rean, pts_dict[list_name_city[i]], ['lat', 'lon'], full=True)['full'], dim_mean=['geo']))
        #                                           for i, no_city in enumerate(list_no_city)]})
=======
        #                                                cut_mlp(dts=cut_points(sub_rean, pts_dict[self.list_name_city[i]], full=True)['full'], dim_mean=['geo']))
        #                                           for i, no_city in enumerate(self.list_no_city)]})
>>>>>>> c80f4457 (First commit)
        #     except Exception as e:
        #         raise ValueError("Stop here")
        #     # City that has data gap smaller then 5%
        #     diso_df_list.append(diso_df)
        # endregion

        # USE FOR MULTIPLE INDEX
        df_compare_list = []
        df_sing_list = []
        df_obs_list = []
<<<<<<< HEAD
        
        for ind in list_ind:
=======

        for ind in self.list_ind:
>>>>>>> c80f4457 (First commit)
            #print("Running", ind)

            # Produce dataframe like this
            # City | Ind | Data_name
            # -----------------
            # ALUOI| T10p| 0.5
            # -----------------
            # HANOI| T10p| 0.6
            # ...............

<<<<<<< HEAD
            # name of the reanalysis and observational file
            # #print(f'Processing {ind} for {dataset}')
            rean_nc = Data_ind+ind+f"_{dataset}.nc"
            obs_nc = Data_ind+ind+f"_obs.nc"

            # subset the rean and data in the same period
            # some rean ends before 2019
            (sub_rean, _), (sub_obs, _) = subset_year(
                rean_nc, param_dict[var], mode='rean'), subset_year(obs_nc, var)
            main_compare_dict, main_sing_dict, main_obs_dict = [], [], []
            for i, no_city in enumerate(list_no_city):
=======
            print("Processing", ind, "for", self.dataset)

            # name of the reanalysis and observational file
            # #print(f'Processing {ind} for {dataset}')
            rean_nc = Data_ind+ind+f"_{self.dataset}{self.season_suffix}.nc"
            obs_nc = Data_ind+ind+f"_obs{self.season_suffix}.nc"
            # print('Debug', rean_nc)
            # subset the rean and data in the same period
            # some rean ends before 2019

            (sub_rean, _), (sub_obs, _) = self._subset_year(
                    rean_nc, self.var, mode='rean'), self._subset_year(obs_nc, self.var)
            main_compare_dict, main_sing_dict, main_obs_dict = [], [], []

            # Produce metrics for each city
            for i, no_city in enumerate(self.list_no_city):
>>>>>>> c80f4457 (First commit)
                #    1. Subset wrt the position of the cities along lat, lon dimension (4 points => mean geo dimension)
                #    2. Extract the data from the reanalysis and observation (/w var_rean and var_obs)
                #    3. calculate the diso index for each city
                obs_data = sub_obs.where(
                    sub_obs['no_station'] == no_city, drop=True)

<<<<<<< HEAD
                rean_data = cut_mlp(dts=cut_points(sub_rean, pts_dict[list_name_city[i]], [
                                    'lat', 'lon']), dim_mean=['geo'])
                # #print(rean_data, obs_data)
                if not compare_skip:
=======
                rean_data = cut_mlp(dts=cut_points(sub_rean, pts_dict[self.list_name_city[i]]
                                                   ), dim_mean=['geo'])

                #? For DJF, there are cases where the rean has extra value for the last year => We need to omit that
                # if len(obs_data) < len (rean_data):
                #     no_omit = len(rean_data) - len(obs_data)
                #     rean_data = rean_data[:-no_omit]

                if not self.compare_skip:
>>>>>>> c80f4457 (First commit)
                    try:
                        compare_dict = {
                            **evaluate_compare(obs_data, rean_data, metrics_compare),
                        }
<<<<<<< HEAD
                    except:
                        #print(obs_data, rean_data)
                        raise ValueError("Stop here")

                if not sing_skip:
                    sing_dict = evaluate_single_all(rean_data)

                if not obs_skip:
                    obs_dict = evaluate_single_all(obs_data)

                if not compare_skip:
                    main_compare_dict.append(compare_dict)
                if not sing_skip:
                    main_sing_dict.append(sing_dict)
                if not obs_skip:
                    main_obs_dict.append(obs_dict)
            try:
                if not compare_skip:
                    # Example at this step: result_df = [{'R': 1,'DISO':2},{'R': 3,'DISO':4},{'R': 5,'DISO':6]
                    result_compare_dict = {key: [d.get(key, None) for d in main_compare_dict]  # Get the value of the key and put in list
                                           for key in set().union(*main_compare_dict)}  # Get the set of all the keys
                if not sing_skip:
                    # Example at this step: result_df = {'R': [1,3,5],'DISO':[...]}
                    result_sing_dict = {key: [d.get(key, None) for d in main_sing_dict]  # Get the value of the key and put in list
                                        for key in set().union(*main_sing_dict)}

                if not obs_skip:
                    result_obs_dict = {key: [d.get(key, None) for d in main_obs_dict]  # Get the value of the key and put in list
                                       for key in set().union(*main_obs_dict)}

                if not compare_skip:
                    compare_df = pd.DataFrame({'name_station': list_name_city,
                                               'ind': [ind]*len(list_name_city),
                                               **result_compare_dict})
                if not sing_skip:
                    sing_df = pd.DataFrame({'name_station': list_name_city,
                                            'ind': [ind]*len(list_name_city),
                                            **result_sing_dict})

                if not obs_skip:
                    obs_df = pd.DataFrame({'name_station': list_name_city,
                                           'ind': [ind]*len(list_name_city),
=======
                        if len(obs_data.time) != len(rean_data.time):
                            print(self.season, self.dataset, obs_data.time, rean_data.time)
                            raise ValueError("Stop here")

                    except:
                        #print(obs_data, rean_data)
                        raise ValueError("Stop here")
                    main_compare_dict.append(compare_dict)
                if not self.sing_skip:
                    sing_dict = evaluate_single_all(rean_data)
                    main_sing_dict.append(sing_dict)
                if not self.obs_skip:
                    obs_dict = evaluate_single_all(obs_data)
                    main_obs_dict.append(obs_dict)

            # Create dataframe
            try:
                if not self.compare_skip:
                    # Example at this step: result_df = [{'R': 1,'DISO':2},{'R': 3,'DISO':4},{'R': 5,'DISO':6]
                    result_compare_dict = {key: [d.get(key, None) for d in main_compare_dict]  # Get the value of the key and put in list
                                           for key in set().union(*main_compare_dict)}  # Get the set of all the keys
                    compare_df = pd.DataFrame({'name_station': self.list_name_city,
                                               'ind': [ind]*len(self.list_name_city),
                                               **result_compare_dict})
                if not self.sing_skip:
                    # Example at this step: result_df = {'R': [1,3,5],'DISO':[...]}
                    result_sing_dict = {key: [d.get(key, None) for d in main_sing_dict]  # Get the value of the key and put in list
                                        for key in set().union(*main_sing_dict)}
                    sing_df = pd.DataFrame({'name_station': self.list_name_city,
                                            'ind': [ind]*len(self.list_name_city),
                                            **result_sing_dict})

                if not self.obs_skip:
                    result_obs_dict = {key: [d.get(key, None) for d in main_obs_dict]  # Get the value of the key and put in list
                                       for key in set().union(*main_obs_dict)}

                    obs_df = pd.DataFrame({'name_station': self.list_name_city,
                                           'ind': [ind]*len(self.list_name_city),
>>>>>>> c80f4457 (First commit)
                                           **result_obs_dict})

            except Exception as e:
                #print(ind, dataset)
                raise ValueError("Stop here")

<<<<<<< HEAD
            # City that has data gap smaller then 5%
            if not compare_skip:
                df_compare_list.append(compare_df)
            if not sing_skip:
                df_sing_list.append(sing_df)
            if not obs_skip:
=======

            # City that has data gap smaller then 5%
            if not self.compare_skip:
                df_compare_list.append(compare_df)
            if not self.sing_skip:
                df_sing_list.append(sing_df)
            if not self.obs_skip:
>>>>>>> c80f4457 (First commit)
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
<<<<<<< HEAD
        if not compare_skip:
            df_compare_all = pd.concat(df_compare_list)
        if not sing_skip:
            df_sing_all = pd.concat(df_sing_list)
        if not obs_skip:
            df_obs_all = pd.concat(df_obs_list)

        # raise KeyboardInterrupt
        if not compare_skip:
            df_compare_all.to_csv(compare_csv, index=False)
        if not sing_skip:
            df_sing_all.to_csv(sing_csv, index=False)
        if not obs_skip:
            df_obs_all.to_csv(obs_csv, index=False)

        df_all = pd.merge(df_compare_all, df_sing_all,
                          on=['name_station', 'ind'])
        df_all.to_csv(all_csv, index=False)
        return df_all
# ------------------------- Start to merge ------------------------- #
    # Merge the indice dataframe
    metrics_composite = {"manndall_kendall": ["trend", "h", "p", "z", "Tau",
                                              "s", "var_s", "slope", "intercept"], }
    # merge_metric = metrics_compare+metrics_singular
    metrics_merge = metrics_singular
    for metric in metrics_merge:
        #print("Running", metric)
        # * for metric in metrics_singular:
        # define of the merge file name
        # ?: I run additional index

        def merge_metric(metric, composite=False):
            merge_file_name = f'{Data_csv}ensemble/{var}_{metric}_merge.csv'
            pivot_file_name = f'{Data_csv}ensemble/pivot/{var}_{metric}_pivot.csv'
            if check_non_exist(pivot_file_name):
                # #print(f'Processing pivot {metric} for {var}')
                if check_non_exist(merge_file_name) or rerun != None:
                    #print(f'Processing merge {metric} for {var}')
                    # Dict to hold the pandaframe of each reanalysis
                    dfs_merge = {}
                    # Check whether the gen_df has been done
                    for rean_name in list(rean_year_dict.keys()):
                        all_csv = f"{Data_csv}single/all_{rean_name}_{var}.csv"
                        # Reach a dataframe of indices of individual reanalysis
                        if check_non_exist(all_csv) or rerun not in [None, 'merge']:
                            dfs_merge[rean_name] = gen_df(rean_name, rerun)
                        else:
                            dfs_merge[rean_name] = pd.read_csv(all_csv)

                    merge_df = pd.DataFrame()

                    # Initiate the merge dataframe
                    for i, rean_name in enumerate(list(rean_year_dict.keys())):
                        # Take the subset of the dataframe and rename into name_station, ind , name_dataset
                        flag = 0
                        while True:
                            try:
                                merge_sub_df = dfs_merge[rean_name][['name_station', 'ind', metric]].rename(
                                    {metric: rean_name}, axis=1)
                                break
                            except KeyError:
                                try:
                                    for metric_group in metrics_composite:
                                        if metric in metrics_composite[metric_group]:
                                            #print("Rerun for", metric)
                                            if metric_group in metrics_singular:
                                                dfs_merge[rean_name] = gen_df(
                                                    rean_name, rerun='sing')
                                            elif metric_group in metrics_compare:
                                                dfs_merge[rean_name] = gen_df(
                                                    rean_name, rerun='compare')
                                            else:
                                                raise ValueError(
                                                    "Stop here")
                                            break  # Exit the for loop if the metric is found and rerun
                                except Exception as e:
                                    flag += 1
                                    if flag == 2:
                                        raise ValueError("Stop here") from e
                            except Exception as e:
                                flag += 1
                                if flag == 2:
                                    raise ValueError("Stop here") from e
                        # This can may flood the RAM but keep name_station and ind ENSURES the match of data
                        if i == 0:
                            merge_df = merge_sub_df
                        else:
                            # Join along these two collumns to align the reanalysis with each other
                            # Keep in mind the order of the reanlysis datasets
                            # join is quicker than merge
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
                    # merge_df = pd.merge(merge_df, rain_df[['name_station', 'lon', 'lat']], on='name_station',
                    #                     how='left')

                    # # Create a categorical column based on the sort_series
                    # if 'R_' in var:
                    #     merge_df['ind'] = merge_df['ind'].map(rain_mapping)
                    # else:
                    #     merge_df['ind'] = merge_df['ind'].map(temp_mapping)

                    # # Sort the DataFrame by the numerical 'ind' and then by 'lat'
                    # merge_df = merge_df.sort_values(
                    #     by=['ind', 'lat', 'lon'], ascending=[True, False, True])

                    # # Reverse the 'ind' column back to categorical values
                    # if 'R_' in var:
                    #     merge_df['ind'] = merge_df['ind'].map(reverse_rain_mapping)
                    # else:
                    #     merge_df['ind'] = merge_df['ind'].map(reverse_temp_mapping)

                    # # Reset the index
                    # merge_df.reset_index(drop=True, inplace=True)
                    # # Drop the specified columns
                    # try:
                    #     merge_df.drop(
                    #         columns=['lon_x', 'lat_x', 'lat_y', 'lon_y'], inplace=True)
                    # except:
                    #     merge_df.drop(columns=['lon', 'lat'], inplace=True)

                    # merge_df.dropna(subset=['ind'], inplace=True)
                    merge_df.to_csv(merge_file_name, index=False)
                else:
                    merge_df = pd.read_csv(merge_file_name)

                # Pivot so indices became the horizontal index
                # city is the verticle index

                # e.g.
                #         Data1                     Data2                  Data3
                # Ind   T10p | T90p ..        | T10p | T90p ..     | T10p | T90p ..
                # City
                # ALUOI 0.5 | 0.5|...         | 0.5  | 0.5 | ...   | 0.5  | 0.5 | ...

                #! I have modified this already
                # ? Check /work/users/tamnnm/code/01.main_code/01.city_list_obs/city/city_sort_draft.py for detail.

                merge_df['ind'] = pd.Categorical(
                    merge_df['ind'], categories=list_ind)
                try:
                    df_piv = merge_df.pivot_table(
                        index='ind', columns='name_station', values=['cera', 'era', 'era5', 'noaa'], sort=False)
                except:
                    df_piv = merge_df.pivot(
                        index='ind', columns='name_station', values=['cera', 'era', 'era5', 'noaa'])

                # df_piv.to_csv(pivot_file_name, index = False)
            else:
                df_piv = pd.read_csv(pivot_file_name)
            return df_piv
        # Return the dataframe of the wanted metric
        if metric in metrics_composite.keys():
            group_file_name = f'{Data_csv}ensemble/{var}_{metric}_merge.nc'
            piv_metric = input(
                f"Input the metric for {metric} according to this list or Enter to skip {metrics_composite[metric]} : ")
            df_merge_composite = []
            for metric_ind in metrics_composite[metric]:
                df_merge = merge_metric(metric_ind)
                if piv_metric == metric_ind:
                    df_piv = df_merge
                df_merge['type'] = metric_ind
                df_merge = df_merge.set_index('type', append=True)

                df_merge_composite.append(df_merge)

            df_merge_composite = pd.concat(df_merge_composite, join='inner')
            # Change the name of the first level of the MultiIndex from None to 'dataset'
            df_merge_composite.columns.set_names(
                ['dataset', 'name_station'], inplace=True)
            df_merge_composite = df_merge_composite.T.stack(level=0)
            #print(df_merge_composite)
            # Convert the DataFrame to an xarray Dataset
            dts = xr.Dataset.from_dataframe(df_merge_composite)
            dts = dts.set_coords(['name_station', 'dataset', 'ind'])
            #print(dts)
            dts.to_netcdf(group_file_name)
            if metric == "":
                continue
        else:
            df_piv = merge_metric(metric)
            
        if metric_stop is not None:
            if (metric == metric_stop):
                df_return = df_piv
                return df_return
        else:
            df_return = None


# ------------------------------------------------------------------ #
#                             PLOT FIGURE                `:w
=======
        if not self.compare_skip:
            df_compare_all = pd.concat(df_compare_list)
            df_compare_all.to_csv(self.compare_csv, index=False)
        if not self.sing_skip:
            df_sing_all = pd.concat(df_sing_list)
            df_sing_all.to_csv(self.sing_csv, index=False)
        if not self.obs_skip:
            df_obs_all = pd.concat(df_obs_list)
            df_obs_all.to_csv(self.obs_csv, index=False)

        if not self.sing_skip and not self.compare_skip:
            df_all = pd.merge(df_compare_all, df_sing_all,
                            on=['name_station', 'ind'])
            df_all.to_csv(self.all_csv, index=False)
            return df_all
        elif not self.sing_skip:
            return df_sing_all
        elif not self.compare_skip:
            return df_compare_all
        elif not self.obs_skip:
            return df_obs_all

    def _subset_year(self,file, var_tup, mode="obs"):
        """_summary_

        Args:
            file (_type_): Can be the name of dataset or dataset itself
            var_tup (_type_): the variable you want
            mode (str, optional): _description_. Defaults to "obs".

        Returns:
            _type_: _description_
        """

        if isinstance(file, str):
            full_ds = xr.open_dataset(file)
        elif isinstance(file, xr.Dataset) or isinstance(file, xr.DataArray):
            full_ds = file.squeeze(drop=True)


        # observation
        if mode == "obs":
            # Using directly her name
            main_var_name = find_name(full_ds, var_tup, fname_group = ['R', 'T2m', 'Tm', 'Tx','Tn'], opt='var')
        else:
            # reanalysis
            # Using the para_dict to hold
            main_var_name = find_name(full_ds, var_tup, fname_group = param_dict[var_tup]['list_name'], opt='var')

        if isinstance(full_ds, xr.Dataset):
            ds = full_ds[main_var_name]
        elif isinstance(full_ds, xr.DataArray):
            ds = full_ds

        ds_sub = ds.where((ds.time.dt.year >= self.start_year) & (ds.time.dt.year <= self.end_year), drop=True)
        if not self.season_skip and 'DJF' in self.season:
            ds_sub = ds.where((ds.time.dt.year > self.start_year) & (ds.time.dt.year < self.end_year), drop=True)
        return ds_sub, main_var_name

    def MergeMetric(self,metric):
        self.merge_filename = f'{Data_csv}ensemble/{self.merge_suffix}_merge.csv'
        self.pivot_filename = f'{Data_csv}ensemble/pivot/{self.merge_suffix}_pivot.csv'

        if self.rerun == 'obs':
            return self.GenMetric()

        if check_exist(self.pivot_filename):
            return pd.read_csv(self.pivot_filename)

        # print(f'Processing pivot {metric} for {var}')
        if check_exist(self.merge_filename) and self.rerun == 'Default':
            merge_df = pd.read_csv(self.merge_filename)
        else:
            # MERGING THE METRICS
            print(f'Processing merge {metric} for {self.var}')
            # Declare the merge dataframe
            merge_df = None

            # Check whether the gen_df has been done
            for rean_dataset in list(rean_year_dict.keys()):
                self.dataset = rean_dataset
                self._setup_constant_dataset()
                # Reach a dataframe of indices of individual reanalysis
                dfs_merge = self.GenMetric()
                try:
                    merge_sub_df = dfs_merge[['name_station', 'ind', metric]].rename(
                        {metric: rean_dataset}, axis=1)
                except KeyError:
                    merge_sub_df = self._handle_missing_metric(dfs_merge, metric)
                # This can may flood the RAM but keep name_station and ind ENSURES the match of data
                if merge_df is None:
                    merge_df = merge_sub_df
                else:
                    # Join along these two collumns to align the reanalysis with each other
                    # Keep in mind the order of the reanlysis datasets
                    # join is quicker than merge
                    merge_df = merge_df.merge(
                        merge_sub_df, on=['name_station', 'ind'], how='outer')
        Cache.save_data(merge_df, "/work/users/tamnnm/code/01.main_code/03.indices_model_eva/test2.csv")

        # Sort the merge_df
        if len(self.list_ind) == 1:
            # Only sort by location
            merge_df=csort.sort_city(merge_df, sort_ind=False, var=self.var)
        else:
            merge_df=csort.sort_city(merge_df, var=self.var)
        Cache.save_data(merge_df, "/work/users/tamnnm/code/01.main_code/03.indices_model_eva/test.csv")
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

        # Save the merge dataframe
        merge_df.to_csv(self.merge_filename, index=False)

        # Create pivot table for analysis
        merge_df['ind'] = pd.Categorical(merge_df['ind'], categories=self.list_ind)
        try:
            df_piv = merge_df.pivot_table(
                index='ind', columns='name_station',
                values=list(rean_year_dict.keys()), sort=False)
        except:
            df_piv = merge_df.pivot(
                index='ind', columns='name_station',
                values=list(rean_year_dict.keys()))
        # df_piv.to_csv(self.pivot_filename, index = False)
        # Pivot so indices became the horizontal index
        # city is the verticle index
        return df_piv

    def _handle_missing_metric(self, dfs_merge, metric):
        # Take the subset of the dataframe and rename into name_station, ind , name_dataset
        flag = 0
        max_try = 2
        while True:
            try:
                for metric_group in metrics_composite:
                    if metric in metrics_composite[metric_group]:
                        #print("Rerun for", metric)
                        if metric_group in metrics_singular:
                            self.rerun='sing'
                            dfs_merge= self.GenMetric(
                                self.dataset)
                        elif metric_group in metrics_compare:
                            self.rerun='compare'
                            dfs_merge= self.GenMetric(
                                self.dataset)
                        else:
                            raise ValueError(
                                "Stop here")
                        break  # Exit the for loop if the metric is found and rerun
                return dfs_merge[['name_station', 'ind', metric]].rename(
                    {metric: self.dataset}, axis=1)

            except Exception as e:
                flag += 1
                if flag == max_try:
                    print("e", e)
                    raise ValueError("Stop here") from e

    def ProcessMetric(self, metric, season=None):

        # Set-up seasonal suffix
        if self.season_skip:
                self.season_suffix = ''
        else:
            self.season_suffix = f'_{season}'

        #TODO: Modifying this to add option
        self.sing_suffix = f'{self.var}{self.ind_suffix}{self.season_suffix}'
        self.merge_suffix = f'{self.var}_{metric}{self.ind_suffix}{self.season_suffix}'

        # CREATE the dataframe of the COMPOSITE METRIC
        # Deal with composite metric
        if metric in metrics_composite.keys():
            group_file_name = f'{Data_csv}ensemble/{self.merge_suffix}_merge.nc'
            print(group_file_name)
            if check_exist(group_file_name):
                return xr.open_dataset(group_file_name)
            else:
                df_merge_composite = []
                for metric_ind in metrics_composite[metric]:
                    print("Processing metric", metric_ind)
                    df_merge = self.MergeMetric(metric_ind)
                    df_merge['type'] = metric_ind
                    df_merge = df_merge.set_index('type', append=True)

                    df_merge_composite.append(df_merge)

                df_merge_composite = pd.concat(df_merge_composite, join='inner')
                # Change the name of the first level of the MultiIndex from None to 'dataset'
                df_merge_composite.columns.set_names(
                    ['dataset', 'name_station'], inplace=True)
                df_merge_composite = df_merge_composite.T.stack(level=0)
                #print(df_merge_composite)
                # Convert the DataFrame to an xarray Dataset
                dts = xr.Dataset.from_dataframe(df_merge_composite)
                dts = dts.set_coords(['name_station', 'dataset', 'ind'])
                #print(dts)
                dts.to_netcdf(group_file_name)
                return dts
        else:
            return self.MergeMetric(metric)

    def process(self):
        """
        Process all metrics for all seasons.

        Returns
        -------
        dict
            Dictionary of results for each season
        """

        if self.metric_stop:
            if not self.season_skip:
                results = {}
                for season in self.seasons_to_process:
                    self.season = season
                    print(f"Processing metric {self.metric_stop} for {season}")
                    results[season] = self.ProcessMetric(self.metric_stop, season=season)
                # Return results
                if len(self.seasons_to_process) == 1:
                    print(f"Returning result for single season: {self.seasons_to_process[0]}")
                    return results[self.seasons_to_process[0]]
                else:
                    print(f"Returning results for all seasons: {results}")
                    return results

            print(f"Processing single metric {self.metric_stop} without seasons")
            return self.ProcessMetric(self.metric_stop)
        else:
            print('Check')
            for metric in self.list_metric:
                if self.season_skip:
                    print(f"Processing metric {metric} without seasons")
                    self.ProcessMetric(metric)
                else:
                    for season in self.seasons_to_process:
                        self.season = season
                        print(f"Processing metric {metric} for {season}")
                        self.ProcessMetric(metric, season=season)

            print("Finish processing all metrics. Return nothing")

# Example usage:
# metric_processor = MetricExtreme(var='R', seasonal=['DJF', 'JJA'])


# ------------------------------------------------------------------ #
#                             PLOT FIGURE
>>>>>>> c80f4457 (First commit)
# ------------------------------------------------------------------ #

# This figure is a triangle heatmap

def plot_tri(var, metric='DISO'):

    def triangulation_for_triheatmap(M, N):
        step_size = 1
        # vertices of the little squares
        xv, yv = np.meshgrid(np.arange(-0.5, M, step_size),
                             np.arange(-0.5, N, step_size))
        # centers of the little squares
        xc, yc = np.meshgrid(np.arange(0, M, step_size),
                             np.arange(0, N, step_size))
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

    no_rean = len(list(rean_year_dict.keys()))

    # Don't go back and save pivot files. Just modify the *_merge file
    if check_non_exist(Data_csv+f"ensemble/{var}_{metric}_pivot.csv"):
<<<<<<< HEAD
        df_piv = merge_df(var=var, metric_stop=metric)
=======
        df_piv = MetricExtreme(var=var, metric_stop=metric).process()
>>>>>>> c80f4457 (First commit)
    else:
        #print(Data_csv+f"ensemble/{var}_{metric}_pivot.csv")
        df_piv = pd.read_csv(Data_csv+f"ensemble/{var}_{metric}_pivot.csv")

    # try:
    #     df_piv.drop(['SU35', 'TN20'], axis=0, inplace=True)
    # except:
    #     pass

    M = len(df_piv.columns) // 4
    N = len(df_piv)
    values = [df_piv[rean_name] for rean_name in
              list(rean_year_dict.keys())]  # these are the 4 column names in df
    triangul = triangulation_for_triheatmap(M, N)

    # my_cmap.set_bad('#FFFFFF00')
    # my_cmap.set_under('#FFFFFF00')

    # Create a custom colormap that includes grey for NaN values
    def create_custom_colormap(base_cmap):
        base = plt.cm.get_cmap(base_cmap)
        colors = base(np.arange(N))
        # Add grey color for NaN
        colors = np.vstack((colors, [0.5, 0.5, 0.5, 1]))
        custom_cmap = ListedColormap(colors)
        return custom_cmap

    # Create a custom normalization that maps NaN values to the last color in the colormap
    class CustomNormalize(Normalize):
        def __call__(self, value, clip=None):
            result = np.ma.masked_invalid(value)
            result = super().__call__(result, clip)
            result[np.isnan(value)] = 1.0  # Map NaN values to the last color
            return result

    my_cmap_sing = sns.color_palette("RdBu", as_cmap=True)
    if metric == "DISO":
        my_cmap = [my_cmap_sing.reversed()] * no_rean
        vmax = 4
    else:
        vmax = 1
        my_cmap = [my_cmap_sing] * no_rean
    norms = [plt.Normalize(vmin=0, vmax=vmax) for _ in range(no_rean)]

    width = 30
    height = 10
    ratio_wh = width/height
    fig, ax = plt.subplots(figsize=(width, height))

    imgs = [ax.tripcolor(t, np.ravel(val), cmap=cmap, norm=norm, ec='black')
            for t, val, cmap, norm in zip(triangul, values, my_cmap, norms)]

    # imgs = [ax.tripcolor(t, np.ravel(val).astype(float), cmap=cmap, ec='black')
    #         for t, val, cmap in zip(triangul, values, my_cmap)]
<<<<<<< HEAD
=======


>>>>>>> c80f4457 (First commit)
    ax.tick_params(length=0)
    ax.set_xticks(range(M))
    ax.set_xticklabels(df_piv['cera'].columns, rotation=90, fontsize=15)
    ax.set_yticks(range(N))
    ax.set_yticklabels(df_piv.index, fontsize=20)
    ax.invert_yaxis()
    ax.margins(x=0, y=0)
    ax.set_aspect('equal', 'box')  # square cells
    # PLot the colorbar
    cax = fig.add_axes([ax.get_position().x1+0.01,
                       ax.get_position().y0, 0.02, ax.get_position().height])
    # plt.colorbar(im, cax=cax)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = fig.colorbar(imgs[0], cax=cax)
    tick_locator = ticker.MaxNLocator(nbins=9)
    cb.locator = tick_locator
    cb.update_ticks()
    # Left, bottom, width, height
    xy_inset = np.array([[0, 0], [0, 2], [2, 0], [2, 2], [1, 1]])
    triangles_inset = [[0, 1, 4], [0, 2, 4], [1, 3, 4], [2, 3, 4]]

    width_mini = 0.05
    ins = ax.inset_axes([1.01, 1.06, width_mini, width_mini*ratio_wh])
    cmap = ListedColormap(['white'])
    ins.tripcolor(Triangulation(
        xy_inset[:, 0], xy_inset[:, 1], triangles=triangles_inset), [0, 0, 0, 0], ec='k', cmap=cmap)
    ins.set_axis_off()
    # Add labels to inset axes
    ins.text(1, 2.1, 'CERA',
             ha='center', va='bottom', fontsize=14)
    ins.text(1, -0.1, 'ERA5',
             ha='center', va='top', fontsize=14)
    ins.text(-0.1, 1, 'ERA', rotation='vertical',
             ha='right', va='center', fontsize=14)
    ins.text(2.1, 1, '20CR', rotation='vertical',
             ha='left', va='center', fontsize=14)
    plt.suptitle(f"{metric} Evaluation for {var} indices")
    # fig.tight_layout()
    fig.savefig(os.path.join(
        img_wd, f"{var}_{metric}_triheatmap.png"), format="png")
    return

<<<<<<< HEAD

# def plot_


# Create merge & pivot pandaframe for indices
merge_df('R', rerun='obs')
merge_df('T2m', rerun='obs')

raise KeyboardInterrupt

# Create the figure
for metric in ['DISO', 'Taylor_score']:
    plot_tri('R', metric)
    plot_tri('T2m', metric)
=======
# TODO: Modifying so that if new metric is added, it will be addedd in the csv instead of having to rerun the whole thing

# def plot_

if __name__ == "__main__":
    # Manual way
    # for ind in ('R5mm','SU35','TN15'):
    #     gen_class(ind, 'obs')


    # Auto way
    # Run indices file generator

    # #print("Check base file")
    # base_fu()

    # #print("Check indices file")
    # ind_fu()


    metricRun = MetricExtreme()
    # Create merge & pivot pandaframe for indices
    metricRun(var='R', rerun = "single")
    metricRun(var='T2m', rerun = "single")

    # # Create the figure
    # for metric in ['DISO', 'Taylor_score']:
    #     plot_tri('R', metric)
    #     plot_tri('T2m', metric)
>>>>>>> c80f4457 (First commit)


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
            #print(len(all_data))
            # put the obs data into dataframe -> dataarray to group by time
        else:
            #print("The dimension are not equal")

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
            #print("Im testing")
    else:
        #print("Retrying.....")
        continue

# ------------------------- test plot here ------------------------- #
# fig,axs=plt.subplot(subplot_kw=dict(projection=ccrs.PlateCarree()))
# map_pro = ccrs.PlateCarree()
# fig = plt.figure()
# ax = plt.subplot(111, projection=map_pro)

# #print(time_period)
# plt.savefig
"""
