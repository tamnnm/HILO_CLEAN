##### Python Script to convert RegCM netCDF output into WPS Intermediate files #####
# The main library used is pywinter (https://pywinter.readthedocs.io/en/latest/)
import xarray as xr
import numpy as np
import pywinter.winter as pyw
import pandas as pd
import os
from pathlib import Path
from my_junk.bash_sup import source_shell as src
from my_junk.nc_sup import *
from datetime import datetime as dt

print("Start")
# env_vars = src()
# path = env_vars.get('nc_noaa')
path = "/data/projects/REMOSAT/tamnnm/wrf_data/netcdf/noaa_hourly"
os.chdir(path)
savepath = "int_file/"


print("Open datasets")

# fixed parameters
land = open_nc_file('land')
soilhgt = open_nc_file('hgt.sfc')

# Paths and Files
tsoil = open_nc_file('tsoil', sel_year=1881)
soilw = open_nc_file('soilw', sel_year=1881)

# surface
t2m = open_nc_file('air.2m', sel_year=1881)
skt = open_nc_file('skt', sel_year=1881)
u10 = open_nc_file('u10', sel_year=1881)
v10 = open_nc_file('v10', sel_year=1881)
psfc = open_nc_file('pres.sfc', sel_year=1881)
mslp = open_nc_file('prmsl', sel_year=1881)
rhum2m = open_nc_file('rhum.2m', sel_year=1881)

# level data
uwnd = open_nc_file('uwnd', sel_year=1881)
vwnd = open_nc_file('vwnd', sel_year=1881)
air = open_nc_file('air', sel_year=1881)
rhum = open_nc_file('rhum', sel_year=1881)
shum = open_nc_file('shum', sel_year=1881)
hgt = open_nc_file('hgt', sel_year=1881)

R_time_name, R_lat_name, R_lon_name, R_level_name = R_names(tsoil)
tsoil = tsoil.transpose(R_level_name, R_time_name, R_lat_name, R_lon_name)
soilw = soilw.transpose(R_level_name, R_time_name, R_lat_name, R_lon_name)

print("Extracting dimension")
R_time = tsoil[R_time_name].values
R_lat = tsoil[R_lat_name].values
R_lon = tsoil[R_lon_name].values
R_level = rhum[R_level_name]
R_level_soil = soilw[R_level_name].values
R_level = R_level.values * \
    100 if R_level.attrs['units'] in ('millibar', 'hPa') else R_level.values

print(psfc.values.max(), psfc.values.min())
print(mslp.values.max(), mslp.values.min())
print(R_level)

dlat = np.abs(R_lat[1] - R_lat[0])
dlon = np.abs(R_lon[1] - R_lon[0])  # need latitude and longitude increments
stlat = R_lat[0] if R_lat[1] > R_lat[0] else R_lat[-1]
stlon = R_lon[0] if R_lon[1] > R_lon[0] else R_lon[-1]
geo = pyw.Geo0(stlat, stlon, dlat, dlon)

# Volumeric Soil Moisture (SM) and Soil Temperature (ST) data
# top layer - bottom layer in cm

soil_layer = []
if R_level_soil[-1] == 100:
    if R_level_soil[-2] == 40:
        R_level_soil = np.append(R_level_soil, 200)
    if R_level_soil[-2] == 35:
        R_level_soil = np.append(R_level_soil, 300)
    if R_level_soil[-2] == 28:
        R_level_soil = np.append(R_level_soil, 289)
else:
    raise ValueError("Soil levels is not sufficient")

for i in range(len(R_level_soil)-1):
    soil_layer.append(f'{R_level_soil[i]:03.0f}{R_level_soil[i+1]:03.0f}')

# EXAMPLE
# actual layers are 0.05, 0.225, 0.675, 2.0 cm below the surface
# for layer, height in zip(range(), [0.05, 0.175, 0.45, 1.325]):
#     soilm.mrlsl[:, layer] = soilm.mrlsl[:, layer] * height

# ????? All values must have the same time dimension (8x daily here)

# Loop through all the available timesteps in the files
for i, t in enumerate(R_time):
    # Create the intermediate file
    findate = pd.to_datetime(t).strftime('%Y-%m-%d_%H')
    month = pd.to_datetime(t).month
    if month not in [9, 10]:
        continue

    def get_time(var):
        try:
            var = var.sel({R_time_name: t})
            if var.ndim == 3 and len(var[R_level_name]) > len(soil_layer):
                var = cut_level(var, 100, opt="gt")
        except:
            var = var.squeeze()

        if np.any(np.isnan(var.values)):
            print(f"NaN values found in {var.name}")

        return var.values

    # Select data

    stsel = get_time(tsoil)
    smsel = get_time(soilw)

    # Create fields
    ST = pyw.Vsl('ST', stsel, soil_layer)
    SM = pyw.Vsl('SM', smsel, soil_layer)

    # Create the intermediate file
    soil_fields = [ST, SM]

    # ????? Use this when the soil has different time dimension
    # pyw.cinter('SOIL', findate, geo, fields, savepath)

    # Read 2D data and create 2D fields with pyw.V2d() function
    # Surface Pressure
    # ? take at 1st timestep and convert to hPa
    if psfc.attrs['units'] in ('hPa', 'milibar'):
        psfc = psfc.values * 100
    if mslp.attrs['units'] in ('hPa', 'milibar'):
        mslp = mslp.values * 100

    # Surface pressure
    PSFC = pyw.V2d('PSFC', get_time(psfc))
    # Skin Temperature
    SKINTEMP = pyw.V2d('SKINTEMP', get_time(skt))
    # Mean Sea-level Pressure
    PMSL = pyw.V2d('PMSL', get_time(mslp))
    # 2m Air Temperature
    TT2m = pyw.V2d('TT', get_time(t2m))
    # 2m Specific Humidity
    RHUM2m = pyw.V2d('RH', get_time(rhum2m))
    # 10m wind u-component
    UU10m = pyw.V2d('UU', get_time(u10))
    # 10m wind v-component
    VV10m = pyw.V2d('VV', get_time(v10))
    # landsea
    LANDSEA = pyw.V2d('LANDSEA', get_time(land))
    # soilheight
    SOILHGT = pyw.V2d('SOILHGT', get_time(soilhgt))

    sfc_fields = [PSFC, SKINTEMP, PMSL, TT2m,
                  RHUM2m, UU10m, VV10m, LANDSEA, SOILHGT]

    # 3D Air Temperature
    TT = pyw.V3dp('TT', get_time(air), R_level)
    # 3D Relative Humidity
    RH = pyw.V3dp('RH', get_time(rhum), R_level)
    # 3D Specific Humidity
    SPECHUMD = pyw.V3dp('SPECHUMD', get_time(shum), R_level)
    # 3D Wind u-component
    UU = pyw.V3dp('UU', get_time(uwnd), R_level)
    # 3D wind v-component
    VV = pyw.V3dp('VV', get_time(vwnd), R_level)
    # 3D geopotential height
    GHT = pyw.V3dp('GHT', get_time(hgt), R_level)

    level_fields = [TT, RH, SPECHUMD, UU, VV, GHT]

    all_fields = level_fields + sfc_fields + soil_fields
    pyw.cinter('FILE', findate, geo, all_fields, savepath)
    # pyw.cinter('SOIL', findate, geo, soil_fields, savepath)
    # pyw.cinter('FILE_SFC', findate, geo, sfc_fields, savepath)
    # pyw.cinter('FILE_LEVEL', findate, geo, level_fields, savepath)
