#print test the dataset
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
import xarray as xr
import numpy as np
import shapefile as shp
from matplotlib.font_manager import ttfFontProperty
#from matplotlib.lines import _LineStyle
import pandas as pd
import matplotlib
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

# -------------------------------------- import gridded data ------------------------------------- #
data_path="/work/users/tamnnm/Data"
data_path_txt="/dat_daily_1961_2019/"
data_path_nc="/dat_daily_1961_2021/"
data_era="/wrf_data/gribfile/era_synop_1900_2010_fix/"
data_cera="/wrf_data/gribfile/era5_synop_1900_2010_fix/"
data_era5="/wrf_data/gribfile/era5_synop_1940_2013/"

# Path to your GRIB file
grib_file = "path/to/your/file.grib"

# Path to netcdf file
nc_file = "/work/users/tamnnm/Data/wrf_data/netcdf/cera20_synop_1900_2010/T2monthly_1901_M1.nc"
dat_file = xr.open_dataset(nc_file)
time = pd.to_datetime(dat_file['forecast_time0'].values)
print(dat_file['forecast_time0'].values)
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









