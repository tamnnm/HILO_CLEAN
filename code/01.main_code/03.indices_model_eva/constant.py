import os
import pandas as pd
from my_junk import METRICS_SINGULAR, METRICS_COMPOSITE

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
Data_nc_2019 = os.path.join(Data_wd, "obs/UPDATE_METEO/3.OG_obs_1960_2019/")
Data_obs_list = os.path.join(Code_wd, "city/")
Data_obs_pts = os.path.join(Code_wd, "city_pts/")
suffix_2019 = '_1960_2019.txt'
img_wd = os.path.join(os.getenv("img"))
#print(img_wd)
os.chdir(Data_nc)
# Data_2023 = Data_wd+"dat_daily_1961_2021"
indice_dict = {167: {'min': ('Tnn', 'Tnx', 'Tn10p', 'Tn90p', 'CSDI', 'TN20'),#, 'TN15'),
                     'max': ('Txx', 'Txn', 'Tx90p', 'Tx10p', 'WSDI', 'SU25'),#, 'SU35'),
                     'other': ('DTR',)},
            #    228: ('R10mm', 'R20mm', 'R50mm', 'R1mm', 'R5mm', 'Rx1day', 'Rx5day','PRCPTOT', 'R99p', 'R95p')}
               228: ('R10mm', 'R20mm', 'R50mm', 'R1mm', 'R5mm', 'Rx1day', 'Rx5day','CDD', 'CWD','SDII', 'PRCPTOT', 'R99p', 'R95p')} #'CDD', 'CWD', 'SDII' => needs to be recalculated
               #? SDII should it be also recalculated?
temp_tuple = tuple(indice_dict[167]['min'] + indice_dict[167]['max'] + indice_dict[167]['other'])
rain_tuple = indice_dict[228]
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
param_dict = {'Vm': {'no': 165, 'type': 'non_extreme', 'acro': 'Um', },
              'Vx': {'no': 165, 'type': 'non_extreme', 'acro': 'Vx', },
              'Tn': {'no': 167, 'type': 'extreme', 'acro': 'Tn', },
              'Tx': {'no': 167, 'type': 'extreme', 'acro': 'Tx', },
              'T2m': {'no': 167, 'type': 'non_extreme', 'acro': 'T2m', 'list_name': ('var167', 'air', 't2m', 'T2m',)},
              'R': {'no': 228, 'type': 'extreme', 'acro': 'R', 'list_name': ('var228', 'R', 'apcp', 'tp',)},
              }

threshold = {'cera': 10.2, 'era': 11.62, 'era5': 1.87, 'noaa': 3.5, 'obs':1}  # mm/day
metrics_compare = ['rmse', 'mae', 'mpe', 'mape', 'R', 'nme',
                   'nmae', 'nrmse', 'DISO', 'Taylor_score']
metrics_singular = list(METRICS_SINGULAR.keys())
metrics_composite = METRICS_COMPOSITE
valid_season = ['annual', 'DJF', 'MAM', 'JJA', 'SON']
script_path = script_path = os.path.dirname(os.path.abspath(__file__))
json_path = script_path+"/constant.json"

def_start_year = 1961
def_end_year = 2019