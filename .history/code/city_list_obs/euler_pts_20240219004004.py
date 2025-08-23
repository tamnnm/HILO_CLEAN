import numpy as np
import xarray as xr
import pandas as pd
import os

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