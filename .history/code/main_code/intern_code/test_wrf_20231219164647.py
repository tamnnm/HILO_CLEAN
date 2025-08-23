"""
from netCDF4 import Dataset
from salem import open_mf_wrf_dataset
from wrf import ALL_TIMES, latlon_coords, xy_to_ll, ll_to_xy, getvar, interpline, CoordPair
import xarray as xr
import os
from my_junk import *
temp_var = ["tmp", "tmx", "tmn"]
prep_var = ["precip", "no_day", "rhum"]
Data_path = "/work/users/tamnnm/Data/pap25_QA_1945/"
Data_final = Data_path+"temp_run/"
# raise ValueError
# Open downscaling data
Data_wrf = "/work/users/tamnnm/wrf/WRF/test/RUN_ERA5_194412_9/"
tmn_wrf_list = []
precip_wrf_list = []
var_left = []
for i, var_uniform in enumerate(temp_var+prep_var):
    dts_final_dir = Data_final+"wrf_"+var_uniform+"_daily.nc"
    # if os.path.exists(dts_final_dir):
    #    dts_var = xr.open_dataset(dts_final_dir)
    # dts_var = dts_var.drop_dims(["south_north", "west_east"])
    # dts_var = dts_var.expand_dims(["south_north", "west_east"])
    #    globals()[f'wrf_{var_uniform}'] = dts_var
    # else:
    var_left.append([var_uniform, dts_final_dir])

wrf_list = []
var_wrf_list = []

for file_name in sorted(os.listdir(Data_wrf)):
    if "wrfout_d02" in file_name:
        # print(file_name)
        dts_wrf_name = Dataset(Data_wrf+file_name)
        wrf_list.append(dts_wrf_name)
        # var_name_final = f'wrf_{var_name}'
        # var_wrf_list.append(dts_var)
    else:
        continue
for i, var_name_dir in enumerate(var_left):
    method_wrf = "mean"
    var_name = var_name_dir[0]
    var_dir = var_name_dir[1]
    if os.path.exists(var_dir):
        dts_var = xr.open_dataset(var_dir)
    if var_name == "precip":
        wrf_name = "RAINC"
        method_wrf = "sum"
    elif var_name == "tmp":
        wrf_name = "T2"
    # elif var_name == "rhum":
    #    wrf_name = "rh2"
    # elif var_name == "tmn":
    #    wrf_name = "T2_MIN"
    # elif var_name == "tmx":
    #    wrf_name = "T2_MAX"
    else:
        continue

    # #print(dts_var)
    # dts_var = getvar(wrf_list, wrf_name, timeidx=ALL_TIMES,
    #                 method='cat')  # unit: mm
    # del dts_var.attrs['projection']
    # dts_var.to_netcdf(var_dir, 'w', engine="h5netcdf",
    #                  format='NETCDF4')
    # print(dts_var.coords['XLONG'].shape)
    dts_test = (dts_var.coords['XLONG'] < 105) & (dts_var.coords['XLAT'] < 25) & (
        dts_var.coords['XLONG'] > 103) & (dts_var.coords['XLAT'] > 22)
    print(dts_test.shape)
    dts_long = dts_var.where(dts_test, drop=True)

    # print(dts_long)
    dts_test = cut_co(dts_var, ulat=25, dlat=22,
                      ulon=105, dlon=103)
    # print(dts_test.shape)
    dts_new = dts_var.where(dts_test, drop=True)
    print("Complete", dts_new)

    if method_wrf == "sum":
        if dts_var.attrs['units'] == "mm":  # precipitation
            dts_var = dts_var
        else:
            dts_var = dts_var * 1000  # unit:m
        dts_var.to_netcdf(f'{Data_final}wrf_{var_name}_daily.nc', 'w', engine="h5netcdf",
                          format='NETCDF4')
        # try:
        #    dts_var = dts_var.resample(Time="1M", closed="left").sum()
        # except:
        #    dts_var = dts_var.resample(time="1M", closed="left").sum()
    else:
        if "T2" in wrf_name:
            dts_var = dts_var  # unit: K
        else:
            dts_var = dts_var
        dts_var.to_netcdf(f'{Data_final}wrf_{var_name}_daily.nc', 'w', engine="h5netcdf",
                          format='NETCDF4')
        # try:
        #    dts_var = dts_var.resample(Time="1M", closed="left").mean()
        # except:
        #    dts_var = dts_var.resample(time="1M", closed="left").mean()
    # dts_var.to_netcdf(f'{Data_final}wrf_{var_name}.nc', 'w', engine="h5netcdf",
    #                  format='NETCDF4')
    dts_var = dts_var.expand_dims(["south_north", "west_east"])
    globals()[f'wrf_{var_name}'] = dts_var
    """
import matplotlib.style as style
print(style.available)

xr.Dataset.res
