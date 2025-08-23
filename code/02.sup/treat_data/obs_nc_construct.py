"""_summary_
    This script is used to construct the netcdf file for the observation data
    5 variables: R, T2m, Tm (min), Tx(max), Um, Vx
    Output file:
    dimensions:
        time = 21915 ;
        no_station = 169 ;
        string20 = 20 ;
variables:
        double R(time, no_station) ;
                R:_FillValue = NaN ;
        double data_gap(no_station) ;
                data_gap:_FillValue = NaN ;
                string data_gap:standard_names = "data_gap" ;
        int64 end_year(no_station) ;
                string end_year:standard_names = "end_year" ;
        double lat(no_station) ;
                lat:_FillValue = NaN ;
                string lat:standard_names = "latitude" ;
        double lon(no_station) ;
                lon:_FillValue = NaN ;
                string lon:standard_names = "longitude" ;
        char name_station(no_station, string20) ;
                string name_station:standard_names = "name_station" ;
        int64 no_station(no_station) ;
        int64 start_year(no_station) ;
                string start_year:standard_names = "start_year" ;
        int64 time(time) ;
                string time:units = "days since 1960-01-01 00:00:00" ;
                string time:calendar = "proleptic_gregorian" ;

    (!) Some years might be missing => Not continous

    _description_
    NAME
    LONGITUDE LATITUDE .... ... ....
    1973
    1    0.0    0.0    0.0    0.0    0.1   28.9    2.7    9.0    0.5   16.0    0.0    0.0
    2    0.0    0.0    0.0    0.0    0.0   23.1   15.6    0.0    1.6   12.0    1.0    0.0
    3    0.7    0.0    0.0    0.0    0.0   19.5    6.0    2.3    0.0   10.7   14.9    0.0
    4    0.1    0.0    0.0    0.0   26.1    0.0   14.3    3.2    0.0  155.4    8.5    0.0
    5    0.2    0.0    0.0    0.0    0.0    7.7    2.0    0.3    5.0   60.3    0.6    0.1
    6    0.5    0.0    0.0    0.0    0.0   65.5   23.1   16.3   24.5  104.7    6.6    2.5
    7    0.4    0.0    0.0    0.0    0.0   32.2  186.0    0.5    0.3  368.1   69.0   19.0
    8    0.0    8.0    0.0    0.0    0.2    0.5  154.7   11.0    3.4   67.7    7.0    5.7
    9    0.0   28.2    0.0    0.0    3.5    0.0    0.0    0.0   27.0    7.1    0.0    3.2
    10    0.3    0.0    0.0    0.0    8.3    0.0   48.4    0.0    3.0    4.0    3.2    0.9
    11    0.1    0.0    0.0    0.0    0.2    0.0    8.0    0.0   22.0    1.5  246.1    0.0
    12    0.3    2.4    0.0    4.0    0.2    0.0   16.0    0.0    0.0    6.3  137.6    7.0
    13    0.8    0.4    0.0    0.2   31.0    0.0    9.7    0.0    2.0   37.9   43.6    2.2
    14    0.0   18.1    0.0    0.0    0.1    0.4    2.0    0.0    0.1   40.9    2.4   16.8
    15    0.1    0.7 - 99.0    0.0    0.0    0.0   11.0    0.0   74.0  172.8    1.4    7.5
    16    0.0    0.0    0.8    0.0    0.0    2.0    0.0    0.0    0.3    0.0   32.6    0.2
    17    0.1    0.0    0.4    0.0    0.0    0.0    0.0    0.0    2.5    0.1   49.0    0.1
    18    0.0    0.0    0.3    0.0    0.0    0.0    0.0    0.0    8.0    0.0  102.7    0.0
    19    0.0    0.0    0.0    1.8    0.0    0.0    0.0    0.0   16.7    0.0   23.0    2.8
    20    4.9    0.0    0.0   12.8    0.0    0.0    0.0    0.0   26.0    0.0    1.0    2.2
    21    0.3    0.0    0.0    7.8    0.0    0.0    0.0    0.0   36.0    0.0    3.7   12.2
    22    0.0    0.0    0.0    3.2    0.0    0.0    0.0    0.0    9.3    0.0    4.7    4.5
    23    0.0    0.0    0.0   28.1    0.5    4.0    0.0   21.7    0.9   38.2    1.5    0.0
    24    0.0    0.0    0.0    0.0    1.2    0.0    0.0    2.0    1.0  124.0    0.0    3.0
    25    0.0    0.0   15.0    2.9    0.4    0.0    0.0    0.5   41.0  335.0    0.6    0.0
    26    0.2    0.0   67.1   18.5   27.0    0.0    0.0    8.0    8.0    5.4    1.9    0.0
    27    0.0    0.0    2.0    3.4   11.5   23.0   11.2    9.7    6.0  228.8    0.0    0.5
    28    2.8    0.0    0.0    6.0   72.1   19.5   20.0   14.0   31.0   15.5    0.0    1.0
    29    9.6 - 99.0    0.0   19.6   42.2    5.0   19.0   19.0    7.2    0.0    1.0    0.0
    30   17.9 - 99.0    0.0    1.5    0.0   48.2    4.0    1.0    3.0    0.1    3.4    0.5
    31    0.8 - 99.0    0.0 - 99.0   23.0 - 99.0   12.0    0.0 - 99.0    0.0 - 99.0    0.0
    - Vertical: each month(12 columns - 12 month)
    - Each month: 31 days(31 rows - 31 days) = > must fix
    """


import os
import pandas as pd
import xarray as xr
import numpy as np
import glob
from calendar import monthrange, isleap
from functools import reduce
import subprocess
import concurrent.futures as con_fu

# Create the OG file


def create_OG_file(var):
    file_path = f"{Data_obs}/OG_{var}_daily_1960_2019.nc"
    if os.path.exists(file_path):
        ds_fin = xr.open_dataset(file_path)
        print(f"OG_{var} already exists")
    else:
        # REMEMBER sorted to keep the right alphabetical order
        for no_station, txt_file in enumerate(sorted(glob.glob(f'{Data_2019}/{var}_*_{suffix_2019}'))):
            # ---------------------- Present station data ---------------------- #
            lines_split = []
            full_year_data = []

            def float_replace(x):
                return np.nan if float(x) == -99.0 else float(x)

            def nan_fill(start, end, year_data):
                for year in range(int(start), int(end)):
                    # CHECK: print(year,366 if isleap(year) else 365)
                    year_data.extend([np.nan]*366 if isleap(year)
                                     else [np.nan]*365)
                return

            with open(txt_file, 'r') as f:
                # Split file in to a list a row
                lines_split = [line.split() for line in f.readlines()]
                # Exception: file starts with no metadata
                try:
                    # Test to see if it can be convert
                    int(lines_split[0][0])
                    # Test if the first line is the name of the city or not
                    lines_data = [list(map(float_replace, sublist))
                                  for sublist in lines_split]
                    # e.g. Vx_TAMDAO_1961_2019.txt
                    city_name = txt_file.split("/")[-1].split("_")[1]
                    with open(f"{Data_2019}/R_{city_name}_1961_2019.txt", 'r') as f:
                        lon_city, lat_city = f.readlines()[1].split()[:2]
                    print(city_name, lon_city, lat_city)
                except Exception as e:
                    # Extract the city name (1st line), [lat, lon] (2nd line)
                    city_name, (lon_city,
                                lat_city) = lines_split[0][0], lines_split[1][:2]
                    # Change all value to float
                    lines_data = [list(map(float_replace, sublist))
                                  for sublist in lines_split[2:]]

                # Calculation supplementary data
                no_years = (len(lines_data))//32
                # Replace the start year if start year < 1961
                start_ind, end_ind = int(lines_data[0][0]), int(lines_data[(
                    no_years-1)*32][0])

            # Turn the data into a dataframe
            for i in range(no_years):
                # 12 month
                current_year = lines_data[i*32][0]

                # THERE CAN BE GAP OF YEAR IN THE DATA
                # Add all the year in the gaps to be nan
                # IGNORE if first year < 1961
                # e.g. first year is 1970 to 1972, 1971 must be filled with nan
                if i > 0 and (current_year - lines_data[(i-1)*32][0]) > 1:
                    nan_fill(lines_data[(i-1)*32][0]+1,
                             current_year, full_year_data)

                # FINAL MERGE WILL SOLVE THIS
                # if we reach the last year and it's ealier than 2019, fill nan
                # if i == no_years-1 and current_year < end_year:
                #     print("Last year is not 2019")
                #     nan_fill(current_year+1,end_year+1,full_year_data)

                # Extract the yearly block
                month_block = lines_data[i*32+1:(i+1)*32]
                for j in range(12):
                    # Extract days in a specfic months
                    try:
                        month_data = [day_line[j+1]
                                      for day_line in month_block]
                    except:
                        raise ValueError(
                            f"Error in {city_name} {current_year} {j+1}")

                    # Check if the monthly data length is valid
                    # monthrange return the right length of that month
                    # monthrange(year,month) -> output [int - weekday, int - number of days in month]
                    correct_monthly_length = monthrange(
                        int(lines_data[i*32][0]), j+1)[1]
                    month_data = month_data[:correct_monthly_length] if len(
                        month_data) > correct_monthly_length else month_data
                    # CHECK: no_day+=len(month_data)
                    full_year_data.extend(month_data)
                # CHECK: print(current_year, no_day)

            # Calculate the data gap
            # Subset the data in start_year to end_year
            # List can not use isnull() function => must convert to numpy array orpandas series
            full_year_data = pd.Series(full_year_data)
            # raise KeyboardInterrupt
            data_gap = full_year_data.isnull().sum()/full_year_data.size

            # Add the needed information to the supplimentary data
            add_data.append([no_station, city_name, lat_city,
                             lon_city, start_ind, end_ind, data_gap])

            # Check the total days:
            if len(full_year_data) != 21549:
                print(f"{city_name} is not 21549, {start_ind}-{end_ind}")
            else:
                print(f"{city_name} is complete with 21549 days")

            # Create dataframe for each station only main data
            # no_station must be the equivalent length with the full_year_data
            df = pd.DataFrame({str(var): full_year_data,
                               'time': pd.date_range(start=f'{start_ind}-01-01', end=f'{end_ind}-12-31', freq='D'),
                               'no_station': [no_station]*len(full_year_data),
                               })
            # Add it into a list
            main_data.append(df)

        # Merge the main data (variable(no_station, time))

        # Prepare the main pandaframe
        main_df = reduce(lambda left, right: pd.merge(
            left, right, on=['time', 'no_station', var], how='outer'), main_data)
        main_df = main_df.set_index(
            ['time', 'no_station'])[var]

        # supplimentary data(no_station) : city_name, lat, lon, start_year, end_year
        sup_df = pd.DataFrame(add_data, columns=[
            'no_station', 'name_station', 'latitude', 'longitude', 'start_year', 'end_year', 'data_gap'])

        # Convert the supplimentary data into xarray
        def creat_sup_ds(name, dtype, full_name=None):
            return xr.DataArray(
                data=sup_df[name],
                coords={'no_station': sup_df['no_station']},
                dims=['no_station'],
                attrs={'standard_names': name or full_name},
            ).astype(dtype)

        # Merge the main dataset and the supplimentary dataset
        ds_fin = xr.Dataset({
            f"{var}" if var!="Tm" else "Tn": main_df.to_xarray(),
            'lat': creat_sup_ds('latitude', 'float64'),
            'lon': creat_sup_ds('longitude', 'float64'),
            # The previous version encodes as S20, please run again as U20 when u have time
            'name_station': creat_sup_ds('name_station', 'U20', 'Name of the station'),
            'start_year': creat_sup_ds('start_year', 'int64', 'Start year'),
            'end_year': creat_sup_ds('end_year', 'int64', 'End year'),
            'data_gap': creat_sup_ds('data_gap', 'float64', 'Data gap')
        },
            attrs={
                'description': 'Modified netcdf version created by Tamnnm, 2024'}
        )
        print(ds_fin)
        ds_fin.to_netcdf(file_path if var != 'Tm' else file_path.replace('Tm', 'Tn'), engine="h5netcdf",
                         format='NETCDF4')
        print(f"Finished OG_{var}")
    return

# Create the main file


def create_main_file(var):
    file_path_main = f"{Data_obs}/{var}_daily_1961_2019.nc"
    if os.path.exists(file_path_main):
        print(f"{var} already exists")
    else:
        print(f"{var} doesn't exist")
        OG_ds = create_OG_file(var)
        ds_fin_main = OG_ds[[var, 'name_station']]
        ds_fin_main.to_netcdf(
            file_path_main, engine="h5netcdf", format='NETCDF4')
        print(f"Finished {var}")
    print("------------------")
    return

# def fixing_T2m():
#     temp_indices = ['T2m', 'Tx', 'Tm']
#     for file in os.listdir(Data_obs):
#         if "T" not in file: continue
#         for temp_ind in temp_indices:
#             if temp_ind in file:
#                 output_name=os.path.join(Data_obs,file)
#                 ds = xr.open_dataset(output_name)
#                 attrs = ds[temp_ind].attrs.copy()
#                 ds[temp_ind] = ds[temp_ind] - 273.15
#                 ds[temp_ind].attrs.update(attrs)
#                 ds[temp_ind].attrs['units'] = 'C'
#                 ds=ds.rename_vars({temp_ind:'Tn'}) if temp_ind == 'Tm' else ds
#                 ds.to_netcdf(os.path.join(Data_obs,"OG_"+file) if temp_ind != 'Tm' else os.path.join(Data_obs,"OG_"+file).replace('Tm','Tn'), format='NETCDF4')

    return

if __name__ == "__main__":
    available_vars = ['R', 'T2m', 'Tm', 'Tx', 'Um', 'Vx']
    Data_2019 = "/data/projects/REMOSAT/tamnnm/obs/dat_daily_1961_2019"
    Data_obs = "/data/projects/REMOSAT/tamnnm/obs/UPDATE_METEO/"
    suffix_2019 = "1961_2019.txt"

    for var in available_vars:
        main_data = []
        add_data = []
        start_year = 1960
        end_year = 2019

        # Create the files
        create_OG_file(var)
        # create_main_file(var)
