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
import glob

# --------------------- Data path and constant --------------------- #
Data_wd = "/work/users/tamnnm/Data/"
Code_wd = "/work/users/tamnnm/code/main_code/city_list_obs/"
Data_path = Data_wd+"wrf_data/netcdf/"
Data_name = {'era5':'era_daily/subset/',
#             'era5':'era5_daily/full_nc/',
#             'cera':'cera_daily/subset/',
#             'noaa':'noaa_daily/full_nc/'
}
print("This file only works for equal grid; if not please remap it into one")

# -------------------- Open grib and netcdf file ------------------- #
# for file in os.listdir(Data_grid)
# for para in param_no:
#    for file in os.listdir(Data_grid):
for file_path in Data_name.keys():
    if file_path == 'era5':
        file_name = "indice_full/mean_era5_167_2000.nc"
    else:
        file_name = f"subset/mean_{file_path}_167.nc"
    with xr.open_dataset(f'{Data_path}{file_path}_daily/{file_name}') as ds:
        try:
            lat_all = ds['lat'].values
            lon_all = ds['lon'].values
        except:
            lat_all = ds['latitude'].values
            lon_all = ds['longitude'].values
    # MUST: add values or else it appears as a dataarray subset
        reso_lat = lat_all[1]-lat_all[0]
        reso_lon = lon_all[1]-lon_all[0]

        city_name_list = []

        # --------------------- List of recorded cities -------------------- #
        euleur_file =Code_wd+file_path+"euler_pts.txt"
        if os.path.exists(euleur_file):
            with open(euleur_file, 'r') as f:
                lines = f.readlines()
                f.close()
            for i, line in enumerate(lines):
                city_name_list.append(line.strip().split(',')[0])

        # ---------------------- Present station data ---------------------- #
        for city_list in glob.glob(Code_wd+"*.txt"):
            with open(Code_wd+city_list, 'r') as f:
                lines = f.readlines()
                f.close()

            # ---- transform line into list and cut off the 2 beginning line --- #
            all_city = []
            for i, line in enumerate(lines[1:]):
                coor_city = []
                name = line.strip().split(',')
                city_name = name[0]
                if city_name in city_name_list:
                    continue
                else:
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
                        if abs(lat_test-lat_target) > reso_lat:
                            continue
                        else:
                            for lon_test in lon_all:
                                if abs(lon_test-lon_target) > reso_lon:
                                    continue
                                else:
                                    distance_test = np.sqrt((lat_test - lat_target)
                                                            ** 2 + (lon_test - lon_target)**2)
                                    distance.append(distance_test)
                                    lat_t.append(lat_test)
                                    lon_t.append(lon_test)

                    # -------------- Find the indices of the closest point ------------- #
                    if len(distance) > 4:
                        raise KeyError("more then 4 points is chosen")
                        closest_pts= np.argpartition(np.array(distance), 4)[:4]
                        for idx in closest_pts:
                            lat_pts = lat_t[idx]
                            lon_pts = lon_t[idx]
                            coor_city = coor_city+[lat_pts, lon_pts]
                    else:
                        for i in np.arange(len(distance)):
                            lat_pts = lat_t[i]
                            lon_pts = lon_t[i]
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

                    with open(euleur_file, 'w+') as f:
                        for city_info in all_city:
                            print(city_info)
                            line = ','.join(city_info)+'\n'
                            f.write(line)
                        f.close()
