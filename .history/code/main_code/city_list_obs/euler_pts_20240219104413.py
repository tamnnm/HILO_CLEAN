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
from pandas import read_csv
import datetime as dt
import csv
import xarray as xr
from matplotlib.image import imread
import numpy as np
import pandas as pd
import shapefile as shp
import cfgrib
import itertools


# --------------------- Data path and constant --------------------- #
Data_wd = "/work/users/tamnnm/Data/"
Code_wd = "/work/users/tamnnm/code/city_list_obs/"
Data_path = Data_wd+"wrf_data/netcdf/"
Data_name = {'era':'era_daily/subset/',
             'era5':'era5_daily/full_nc/',
             'cera':'cera_daily/subset/',
             'noaa':'noaa_daily/full_nc/'
}


# -------------------- Open grib and netcdf file ------------------- #
# for file in os.listdir(Data_grid)
# for para in param_no:
#    for file in os.listdir(Data_grid):
for file_path in Data_name.keys():
    with xr.open_dataset(f'{Data_path+Data_name[file_path]+file_path}_167_2000.nc') as ds:
        try:
            lat_all = ds['lat'].values
            lon_all = ds['lon'].values
        except:
            lat_all = ds['latitude'].values
            lon_all = ds['longitude'].values
    # MUST: add values or else it appears as a dataarray subset

        # ---------------------- Present station data ---------------------- #
        with open(Code_wd+"rain_city.txt", 'r') as f:
            lines = f.readlines()
            f.close()

        # ---- transform line into list and cut off the 2 beginning line --- #
        all_city = []
        for i, line in enumerate(lines[1:]):
            coor_city = []
            name = line.strip().split(',')
            city_name = name[0]
            lon_target = float(name[3])
            lat_target = float(name[4])
            # print(i, city_name, lat_target, lon_target)
            # --------------------- Call for array and list -------------------- #
            # 4 points closest
            distance = []
            lat_t = []
            lon_t = []
            # ------------------ Find all the 4 closest points ----------------- #
            for lat_test in lat_all:
                if abs(lat_test-lat_target) > 0.5:
                    continue
                else:
                    for lon_test in lon_all:
                        if abs(lon_test-lon_target) > 0.5:
                            continue
                        else:
                            distance_test = np.sqrt((lat_test - lat_target)
                                                    ** 2 + (lon_test - lon_target)**2)
                            distance.append(distance_test)
                            lat_t.append(lat_test)
                            lon_t.append(lon_test]

            # -------------- Find the indices of the closest point ------------- #

            closest_pts= np.argpartition(np.array(distance), 4)[:4]
                        for idx in closest_pts[:, 1]:
                lat_pts = distance[idx][1]
                lon_pts = distance[idx][2]
                coor_city = coor_city+[lat_pts, lon_pts]
            # must put city_name in list to combine
            all_city.append([city_name]+[str(element) for element in coor_city])

            # Very slow - skip
            """
            closest_pts = np.argsort(np.array(distance), axis=0)[:4]
            for idx in closest_pts[:, 1]:
                lat_pts = distance[idx][1]
                lon_pts = distance[idx][2]
                coor_city = coor_city+[lat_pts, lon_pts]
            # must put city_name in list to combine
            all_city.append([city_name]+[str(element) for element in coor_city])
            """

        with open(Code_wd+"euler_pts.txt", 'w+') as f:
            for city_info in all_city:
                print(city_info)
                line = ','.join(city_info)+'\n'
                f.write(line)
            f.close()
