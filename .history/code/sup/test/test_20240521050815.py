# print test the dataset
"""
data_link="/work/users/student6/tam/Data/wrf_data/netcdf/air.2m.1967.nc"
dset=xr.open_dataset(data_link,decode_times=False)
units,reference_date=dset.time.attrs['units'].split('since')
dset['time']=pd.date_range(start=reference_date,periods=dset.sizes['time'])
print(dset['time'])
"""

# ---------------------------------------------------------------------------- #
#                     compare data with observational data                     #
# ---------------------------------------------------------------------------- #

# ----------------------------------------- import module ---------------------------------------- #
from cdo import *
from calendar import monthrange
import xarray as xr
import numpy as np
import shapefile as shp
from matplotlib.font_manager import ttfFontProperty
# from matplotlib.lines import _LineStyle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import scipy.stats as sst
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pandas import read_csv
import csv
import xarray as xr
from matplotlib.image import imread
import numpy as np
import pandas as pd
import pygrib
from HersheyFonts import HersheyFonts
from my_junk import cut_co, func_resample
import glob
import cftime
# -------------------------------------- import gridded data ------------------------------------- #
data_path = "/work/users/tamnnm/Data"
data_path_txt = "/dat_daily_1961_2019/"
data_path_nc = "/dat_daily_1961_2021/"
data_era = "/wrf_data/gribfile/era_synop_1900_2010_fix/"
data_cera = "/wrf_data/gribfile/era5_synop_1900_2010_fix/"
data_era5 = "/wrf_data/gribfile/era5_synop_1940_2013/"

# Path to your GRIB file
grib_file = "path/to/your/file.grib"

# Path to netcdf folder

"""
nc_path = "/work/users/tamnnm/Data/wrf_data/netcdf/noaa_daily/subset/"
for nc_file in glob.glob(nc_path+"/*.nc"):
    year=nc_file.split('_')[-1].split('.')[0]
    nc_name=nc_file.split('/')[-1]
#nc_file = "/work/users/tamnnm/Data/wrf_data/gribfile/synop_file/synop_file_merge/era_228.nc"
#nc_file = "/work/users/tamnnm/Data/wrf_data/netcdf/era_daily/era_167.nc"
#nc_file_2 = "/work/users/tamnnm/Data/wrf_data/netcdf/era5_daily/full_nc/era5_167_2022.nc"
    with xr.open_dataset(nc_file) as ds:

        reference_date = f'{year}-01-01 00:00:00.0'
        ds['time'] = pd.date_range(
            start=reference_date, periods=ds.sizes['time'], freq="1H")
        #ds = cut_time(ds,name='time', start_date='1900-01-01', end_date='2010-12-31',full=True)['full_dts']
        print(ds['time'])
        if not os.path.exists(f'{nc_path}/fix_era5_228_{year}.nc'):
            ds.to_netcdf(f'{nc_path}/fix_era5_228_{year}.nc', 'w', engine="h5netcdf",
                        format='NETCDF4')
        if not os.path.exists(f'{nc_path}/era5_228_{year}_daily.nc'):
            resample_ds=ds.resample(time='1D').sum(dim='time')
            resample_ds.to_netcdf(f'{nc_path}/era5_228_{year}_daily.nc', 'w', engine="h5netcdf",
                              format='NETCDF4')
        ds.close()
        ds=cut_co(ds,ulat=40,dlat=-15,dlon=80,ulon=150,name=['lat','lon'],full=True)['full_dts']
        #print(nc_name)
        #print(ds['lat'])
        ds.to_netcdf(f'{nc_path}/subset_{nc_name}', 'w', engine="h5netcdf",
                        format='NETCDF4')
"""

# """
# # test case
# nc_file = "/data/projects/REMOSAT/tamnnm/wrf_data/netcdf/cera_daily/subset/test.nc"
# with xr.open_dataset(nc_file) as ds:
#     print(ds['time'])
# """

"""
import pygrib
import xarray as xr

# Path to your GRIB1 file
grib_file = data_path + data_era + "era_1972_89.grb"
i
# Open the GRIB1 file with pygrib
grbs = pygrib.open(grib_file)

# Convert the GRIB1 data to xarray dataset
dataset = xr.Dataset()
for grb in grbs:
    data, lats, lons = grb.data(lat1=-15, lat2=40, lon1=80, lon2=150)
    variable_name = grb.name
    dataset[variable_name] = (('latitude', 'longitude'), data)
    dataset['latitude'] = lats
    dataset['longitude'] = lons

# Close the GRIB1 file
grbs.close()
print(dataset.variables)

"""

"""
#Convert julian to standard date

# Load your dataset
ds = xr.open_dataset('your_file.nc')
# Convert Julian to standard calendar (e.g., Gregorian)
ds['time'] = xr.coding.times.decode_cf_datetime(
    ds['time'], calendar='gregorian')

ds.to_netcdf("CERA_228.nc")

"""
"""
# Convert year,month,day into time

obs_path = "/data/projects/REMOSAT/tamnnm/obs/"
ds = xr.open_dataset(obs_path+"TUNG_METEO/R_daily_1961-2019_dataset1.nc")
print(ds.where(ds['no_station'] == 141, drop=True))
raise KeyboardInterrupt
"""

"""
# Original dataset
for file in glob.glob(obs_path+"TUNG_METEO/*.nc"):
    df = xr.open_dataset(file).to_dataframe().reset_index()
    var = file.split('/')[-1].split('_')[0]
    var_name = 'Rain' if var in ['R', 'T2m'] else var
    var_unit = 'mm' if var == 'R' else 'oC'
    df.day = df.day+1
    df.month = df.month+1
    df.year = df.year+1961
    print(df.year)
    # Check if each date is valid

    # Try to avoid po
    mask = (df['day'] < 1) | (df['day'] > df.apply(
        lambda x: monthrange(x['year'], x['month'])[1], axis=1))
    df = df.loc[~mask]
    # Convert year, month, day to datetime64
    df['time'] = pd.to_datetime(df[['year', 'month', 'day']])
    df = df.drop(['day', 'month', 'year'], axis=1)

    # rename variable
    df = df.rename(columns={var_name: var})

    sup_coords = {'no_station': df['no_station'].unique()}
    # [[var_name]] is to create a dataframe, keeping time and no_station as index
    main_ds = df.set_index(['time', 'no_station'])[var].to_xarray()
    # All the rest variable has only no_station as index
    # This help take unique no_station without deleting some duplicate lat,lon like .unique()
    sup_df = df.drop_duplicates(subset=['no_station'])
    # Create variable for the final dataset

    def sup_ds(name, dtype, full_name=None, coords=None):
        if coords is None:
            coords = {'no_station': df['no_station'].unique()}
        return xr.DataArray(
            data=sup_df[name],
            coords=coords,
            dims=['no_station'],
            attrs={'standard_names': name or full_name},
        ).astype(dtype)

    ds_fin = xr.Dataset({
        f"{var}": main_ds.astype('float64'),
        'lat': sup_ds('latitude', 'float64'),
        'lon': sup_ds('latitude', 'float64'),
        'name_station': sup_ds('name_station', 'S20', 'Name of the station'),
    },
        attrs={
            'description': 'Modified verson from Nguyen Duy Tung, 2023 by Tamnnm, 2024'}
    )
    ds_fin.to_netcdf(f"{obs_path}UPDATE_METEO/{var}_daily_1961_2019.nc", engine="h5netcdf",
                     format='NETCDF4')
    print(ds_fin)

    dimensions:
        no_station = 169;
        year = 59;
        month = 12;
        day = 31;
        charlen = 20;
variables:
        double latitude(no_station);
                latitude: units = "degree north";
        double longitude(no_station);
                longitude: units = "degree east";
        double Year(year);
                Year: units = "year";
        double Rain(day, month, year, no_station);
                Rain: _FillValue = -99.;
                Rain: units = "mm";
                Rain: standard_name = "Daily rain";
        char name_station(no_station, charlen);
                name_station: units = "Name";
                name_station: standard_name = "Name of the station";
    break
"""

"""
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import numpy as np

def create_demo_data(M, N):
    # create some demo data for North, East, South, West
    # note that each of the 4 arrays can be either 2D (N by M) or 1D (N*M)
    # M columns and N rows
    valuesN = np.repeat(np.abs(np.sin(np.arange(N))), M)
    valuesE = np.arange(M * N) / (N * M)
    valuesS = np.random.uniform(0, 1, (N, M))
    valuesW = np.random.uniform(0, 1, (N, M))
    return [valuesN, valuesE, valuesS, valuesW]

def triangulation_for_triheatmap(M, N):
    xv, yv = np.meshgrid(np.arange(-0.5, M), np.arange(-0.5, N))  # vertices of the little squares
    xc, yc = np.meshgrid(np.arange(0, M), np.arange(0, N))  # centers of the little squares
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

import pandas as pd

days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
df = pd.DataFrame({'cols': np.random.choice([*'abcdefghij'], 40),
                   'rows': np.random.choice(days, 40),
                   'north': np.random.rand(40),
                   'east': np.random.rand(40),
                   'south': np.random.rand(40),
                   'west': np.random.rand(40)})
#print(df)
df['rows'] = pd.Categorical(df['rows'], categories=days)  # fix an ordering
print(df)
df_piv = df.pivot_table(index='rows', columns='cols')
df_piv.to_csv('/data/projects/REMOSAT/tamnnm/wrf_data/netcdf/para/test.csv')
print(df_piv)
M = len(df_piv.columns) // 4
N = len(df_piv)
values = [df_piv[dir] for dir in
          ['north', 'east', 'south', 'west']]  # these are the 4 column names in df

triangul = triangulation_for_triheatmap(M, N)
cmaps = ['RdYlBu'] * 4
norms = [plt.Normalize(0, 1) for _ in range(4)]
fig, ax = plt.subplots(figsize=(10, 4))
imgs = [ax.tripcolor(t, np.ravel(val), cmap=cmap, norm=norm, ec='white')
        for t, val, cmap, norm in zip(triangul, values, cmaps, norms)]

"""

# cdo = Cdo()
# file_path = "/data/projects/REMOSAT/tamnnm/wrf_data/netcdf/para/para_228/"
# ibfile = file_path+"b_era_228.nc"
# pminfile = file_path+"bp_min_era_228.nc"
# pmaxfile = file_path+"bp_max_era_228.nc"
# ofile = file_path+'test_2.nc'
# cdo.ydrunpctl(
#     '99,5,pm=r8', input=f'{ibfile} {pminfile} {pmaxfile}', output=ofile, option='-L')

link = "/data/projects/REMOSAT/tamnnm/wrf_data/netcdf/para/para_228"
# link = "/data/projects/REMOSAT/tamnnm/wrf_data/netcdf/cera_daily/subset"
link_2 = "/data/projects/REMOSAT/tamnnm/wrf_data/netcdf/cera_daily/full_nc"
fig, ax = plt.subplots()

cera = xr.open_dataset(
    link_2+"/test_2.nc")['tp'].sel(lat=21.0, lon=105.0)
# era = xr.open_dataset(link+"/era_228.nc")['var228'].sel(lat=21.0, lon=105.0)
# era_5 = xr.open_dataset(
#     link+"/era5_228.nc")['var228'].sel(lat=21.25, lon=105.25)
# noaa = xr.open_dataset(
#     link+"/noaa_228.nc")['apcp'].sel(lat=21.0, lon=105.0)
obs = xr.open_dataset(link+"/obs_228.nc")['R'].sel(no_station=61)

np.set_printoptions(threshold=np.Inf)
print(cera['time'])
reference_date = '1900-01-01 00:00:00.0'
cera['time'] = pd.date_range(
    start=reference_date, periods=cera.sizes['time'], freq="1D")
cera.plot(ax=ax, label='cera', alpha=0.5)
# era.plot(ax=ax, label='era', alpha=0.5)
# era_5.plot(ax=ax, label='era_5', alpha=0.5)
# noaa.plot(ax=ax, label='noaa', alpha=0.5)
obs.plot(ax=ax, label='obs', alpha=0.5)

plt.legend()
plt.savefig(link+"/test.png")
