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
from xtci.entropy_deficit import entropy_deficit
from xtci.potential_intensity_tcpypi import potential_intensity
from xtci.wind_shear import wind_shear
from xtci.absolute_vorticity import absolute_vorticity
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


# ---------------------- Directory of grib file--------------------- #
Data_path_grib = "/work/users/tamnnm/Data/wrf_data/gribfile/era5_6h_1940_2023/"
Data_result = "/work/users/tamnnm/Data/wrf_data/gribfile/sub_result/"


def call_dataset(filename, type_option=None):
    # if os.path.exists(Data_path_grib+filename):
    try:
        if type_option == "array" or type_option == None:
            Tm = xr.open_dataarray(f'{Data_path_grib+filename}',
                                   engine="cfgrib")
        elif type_option == "set":
            Tm = xr.open_dataset(f'{Data_path_grib+filename}',
                                 engine="cfgrib")
        return Tm
    except Exception as e:
        print(e)
        print("There is no such file")


# ------------------- file in original grib file ------------------- #
year_org = 1964
hgt = call_dataset("geopotential_1964_b_even.grib")
sst = call_dataset("sea_surface_temperature_1964_even.grib")
t2m = call_dataset('2m_temperature_1964_even.grib')
td2m = call_dataset('2m_dewpoint_temperature_1964_even.grib')
Tm = call_dataset('temperature_1964_b_even.grib')
# print(t2m, td2m)
rh2m = (100*(np.exp(17.625*td2m/(243.04 + td2m)) /
        np.exp(17.625 * t2m/(243.04 + t2m))))/100,
rh = call_dataset('relative_humidity_1964_b_even.grib')
sh = call_dataset('specific_humidity_1964_b_even.grib')
slp = call_dataset('mean_sea_level_pressure_1964_even.grib')
land_mask_sea = call_dataset('land_sea_mask_1964_even.grib')
is_ocean = sst.isel(time=0).drop('time').pipe(lambda x: x*0 == 0)
uwnd = call_dataset('u_component_of_wind_1964_b_even.grib')
vwnd = call_dataset('v_component_of_wind_1964_b_even.grib')
vort = call_dataset('relative_vorticity_1964_11.grib')
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
"""
for i, z_p in enumerate(p_test):
    break_factor = 0
    level_co = cut_level(Tm, level=z_p)
    latlon_co = cut_co(Tm, dlat=-5, ulat=40, dlon=80, ulon=150)
    # Extract level from wrfoutput
    # data_test = interplevel(z, p, z_p)

    # Extracct level from gribfile
    for day in np.arange(1, 2):
        time_co = cut_time(Tm, yr=1964, mon=11, day=day)
        data_test = cut_mlp(Tm, time_data=time_co,
                            geo_data=latlon_co, level_data=level_co, dim_mean=['time', ], dim_plot=['geo'])  # even though there is one level butdo this so that the data array is 2 dimension
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
"""

# -------------------------- Caculate gpi -------------------------- #
level_850 = cut_level(uwnd, 850)
level_600 = cut_level(uwnd, 600)
level_200 = cut_level(uwnd, 200)
Tm_600 = cut_mlp(Tm, level_data=level_600)
print(Tm_600)
rh_600 = cut_mlp(rh, level_data=level_600)
vort_850 = cut_mlp(vort, level_data=level_850)

# entropy deficit: (s_m_star - s_m)/(s_sst_star - s_b)
print('entropy deficit')
dname = 'chi'
ofile = os.path.join(Data_result, f'.{dname}.nc')
if os.path.exists(ofile):
    chi = xr.open_dataset(ofile)[dname]
    print('[opened]:', ofile)
else:
    p_m = 600*100  # Pa
    level_co = cut_level(Tm, level=p_m/100)
    Tm = Tm_600
    RH = rh_600
    chi = entropy_deficit(
        sst=sst,
        slp=slp,
        Tb=t2m,
        RHb=rh2m,
        p_m=p_m,
        Tm=Tm_600.squeeze(),
        RHm=rh_600.squeeze()/100
    ).where(is_ocean)
    """
    chi.to_dataset(name=dname) \
        .to_netcdf(ofile,t
            encoding={dname: {'dtype': 'float32', 'zlib': True, 'complevel': 1}},
            unlimited_dims='time')
    print('[saved]:', ofile)
    """
# entropy deficit for GPI2010: (s_b - s_m)/(s_sst_star - s_b)
print('entropy deficit for GPI2010')
dname = 'chi_sb'
ofile = os.path.join(Data_result, ('.nc', f'.{dname}.nc'))
if os.path.exists(ofile):
    chi_sb = xr.open_dataset(ofile)[dname]
    print('[opened]:', ofile)
else:
    p_m = 600*100  # Pa
    chi_sb = entropy_deficit(
        sst=sst,
        slp=slp,
        Tb=t2m,
        RHb=rh2m,
        p_m=p_m,
        Tm=Tm_600.squeeze(),
        RHm=rh_600.squeeze()/100,
        forGPI2010=True
    ).where(is_ocean)

    """
    chi_sb.to_dataset(name=dname) \
        .to_netcdf(ofile,
            encoding={dname: {'dtype': 'float32', 'zlib': True, 'complevel': 1}},
            unlimited_dims='time')
    print('[saved]:', ofile)
    """
# potential intensity
print('potential intensity')
ofile = os.path.join(Data_result, ('.nc', f'.PI.nc'))
if os.path.exists(ofile):
    PI = xr.open_dataset(ofile)
    print('[opened]:', ofile)
else:
    def reverse_plevels(x): return x.isel(level=slice(-1, None, -1))
    """
    PI = potential_intensity(
        sst=sst,
        slp=slp.where(is_ocean),
        p=Ta.level.pipe(reverse_plevels),
        T=Ta.pipe(reverse_plevels).where(is_ocean),
        q=q.pipe(reverse_plevels).where(is_ocean),
        dim_x='longitude', dim_y='latitude', dim_z='isobaricInhPa'
        )
    """
    PI = potential_intensity(
        sst=sst,
        slp=slp.where(is_ocean),
        p=Tm.level.pipe(reverse_plevels)*100,
        T=Tm.pipe(reverse_plevels).where(is_ocean),
        q=sh.pipe(reverse_plevels).where(is_ocean),
        dim_z='isobaricInhPa'
    )
    """
    encoding = {dname:{'dtype': 'float32', 'zlib': True, 'complevel': 1}
        for dname in ('pmin', 'vmax')}
    encoding['iflag'] = {'dtype': 'int32'}
    PI.to_netcdf(ofile, encoding=encoding, unlimited_dims='time')
    print('[saved]:', ofile)
    """
# wind shear: ( (u200-u850)**2 + (v200-v850)**2 )**0.5
print('wind shear')
dname = 'Vshear'
ofile = os.path.join(Data_result, ('.nc', f'.{dname}.nc'))
if os.path.exists(ofile):
    Vshear = xr.open_dataset(ofile)[dname]
    print('[opened]:', ofile)
else:
    Vshear = wind_shear(
        u850=cut_mlp(uwnd, level_850),
        v850=cut_mlp(vwnd, level_200),
        u200=cut_mlp(uwnd, level_850),
        v200=cut_mlp(vwnd, level_200)
    )
    """
    Vshear.to_dataset(name=dname) \
        .to_netcdf(ofile,
            encoding={dname: {'dtype': 'float32', 'zlib': True, 'complevel': 1}},
            unlimited_dims='time')
    print('[saved]:', ofile)
    """
# ventilation index: Vshear * chi_m /V_PI
print('ventilation index')
dname = 'VI'
ofile = os.path.join(Data_result, ('.nc', f'.{dname}.nc'))
if os.path.exists(ofile):
    VI = xr.open_dataset(ofile)[dname]
    print('[opened]:', ofile)
else:
    VI = Vshear*chi/PI.vmax.pipe(lambda x: x.where(x > 0))
    """
    VI.to_dataset(name=dname) \
        .to_netcdf(ofile,
            encoding={dname: {'dtype': 'float32', 'zlib': True, 'complevel': 1}},
            unlimited_dims='time')
    print('[saved]:', ofile)
    """
# absolute vorticity at 850hPa
print('absolute vorticity')
dname = 'eta'
ofile = os.path.join(Data_result, f'.{dname}.nc')
if os.path.exists(ofile):
    eta = xr.open_dataset(ofile)[dname]
    print('[opened]:', ofile)
else:
    geo_co = cut_co(vort)
    eta = absolute_vorticity(
        vort850=vort_850.squeeze(),
        lat=geo_co[0]
    )
    """
    eta.to_dataset(name=dname) \
        .to_netcdf(ofile,
            encoding={dname: {'dtype': 'float32', 'zlib': True, 'complevel': 1}},
            unlimited_dims='time')
    print('[saved]:', ofile)
    """

# relative humidity at 600hPa in %
print('relative humidity in %')
dname = 'H'
ofile = os.path.join(Data_result, ('.nc', f'.{dname}.nc'))
if os.path.exists(ofile):
    H = xr.open_dataset(ofile)[dname]
    print('[opened]:', ofile)
else:
    H = rh_600squeeze()
    H.attrs['long_name'] = '600hPa relative humidity'
    H.attrs['units'] = '%'
    """
    H.to_dataset(name=dname) \
        .to_netcdf(ofile,
            encoding={dname: {'dtype': 'float32', 'zlib': True, 'complevel': 1}},
            unlimited_dims='time')
    print('[saved]:', ofile)
    """
# GPI (Emanuel and Nolan 2004): |10**5\eta|**(3/2) * (H/50)**3 * (Vpot/70)**3 * (1+0.1*Vshear)**(-2)
print('GPI')
dname = 'GPI'
ofile = os.path.join(Data_result, f'.{dname}.nc')
if os.path.exists(ofile):
    GPI = xr.open_dataset(ofile)[dname]
    print('[opened]:', ofile)
else:
    GPI = (1e5 * abs(eta))**(3/2) \
        * (H/50)**3 \
        * (PI.vmax/70)**3 \
        * (1+0.1*Vshear)**(-2)
    GPI.attrs['long_name'] = 'Genesis Potential Index'
    GPI.attrs['history'] = '|10**5\eta|**(3/2) * (RH/50)**3 * (Vpot/70)**3 * (1+0.1*Vshear)**(-2)'
    GPI.to_dataset(name=dname) \
        .to_netcdf(ofile,
                   encoding={dname: {'dtype': 'float32',
                                     'zlib': True, 'complevel': 1}},
                   unlimited_dims='time')
    print('[saved]:', ofile)

# GPI2010 (Emanuel 2010): |\eta|**3 * chi**(-4/3) * max((Vpot-35),0)**2 * (25+Vshear)**(-4)
print('GPI2010')
dname = 'GPI2010'
ofile = os.path.join(Data_result, f'.{dname}.nc')
if os.path.exists(ofile):
    GPI2010 = xr.open_dataset(ofile)[dname]
    print('[opened]:', ofile)
else:
    GPI2010 = abs(eta)**3 \
        * chi_sb.where(chi_sb > 0)**(-4/3) \
        * (PI.vmax - 35).clip(min=0)**2 \
        * (25 + Vshear)**(-4)
    GPI2010.attrs['long_name'] = 'Genesis Potential Index of Emanuel2010'
    GPI2010.attrs['history'] = '|\eta|**3 * chi**(-4/3) * max((Vpot-35),0)**2 * (25+Vshear)**(-4)'
    GPI2010.to_dataset(name=dname) \
        .to_netcdf(ofile,
                   encoding={dname: {'dtype': 'float32',
                                     'zlib': True, 'complevel': 1}},
                   unlimited_dims='time')
    print('[saved]:', ofile)

latlon_co = cut_co(Tm, dlat=-5, ulat=40, dlon=80, ulon=150)
for day in np.arange(1, 2):
    (fig, ax) = set_up_plot(-5, 40, 80, 150, 5)
    time_co = cut_time(Tm, yr=1964, mon=11, day=day)
    data_plot = cut_mlp(GPI2010, time_data=time_co,
                        geo_data=latlon_co, dim_mean="time", dim_plot="geo")
# print(data_hgt.values)
    pl_gpi = data_plot.plot.contourf(
        ax=ax, cmap='RdYlBu')  # , linewidths=2, levels=1)
# Set
    ax.clabel(pl_gpi, inline=True, fontsize=10)
# If the variable dimension is not important for the plot, then you can use the ... placeholder.
    ax.set_title(f'gpi-11/1964', pad=25)
    plt.savefig(os.path.abspath(
        rf'/work/users/tamnnm/code/main_code/trop_cyc/img/era5_{day}_11.jpg'), format="jpg", dpi=300)
    plt.close()

# ----------------------- Calculate humidity ----------------------- #


# ----------------------- Calculate vorticity ---------------------- #
