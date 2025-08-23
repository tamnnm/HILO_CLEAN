<<<<<<< HEAD
#!/usr/bin/env python
#################################################################
# Python Script to retrieve 2 online Data files of 'ds626.0',
# total 1.98G. This script uses 'requests' to download data.
#
# Highlight this script by Select All, Copy and Paste it into a file;
# make the file executable and run it on command line.
#
# You need pass in your password as a parameter to execute
# this script; or you can set an environment variable RDAPSWD
# if your Operating System supports it.
#
# Contact rdahelp@ucar.edu (RDA help desk) for further assistance.
#################################################################


import sys, os
import requests

def check_file_status(filepath, filesize):
    sys.stdout.write('\r')
    sys.stdout.flush()
    size = int(os.stat(filepath).st_size)
    percent_complete = (size/filesize)*100
    sys.stdout.write('%.3f %s' % (percent_complete, '% Completed'))
    sys.stdout.flush()

# Try to get password
if len(sys.argv) < 2 and not 'RDAPSWD' in os.environ:
    try:
        import getpass
        input = getpass.getpass
    except:
        try:
            input = raw_input
        except:
            pass
    pswd = input('Password: ')
else:
    try:
        pswd = sys.argv[1]
    except:
        pswd = os.environ['RDAPSWD']

url = 'https://rda.ucar.edu/cgi-bin/login'
values = {'email' : 'nguyenduytung@hotmail.com', 'passwd' : pswd, 'action' : 'login'}
# Authenticate
ret = requests.post(url,data=values)
if ret.status_code != 200:
    print('Bad Authentication')
    print(ret.text)
    exit(1)
dspath = 'https://rda.ucar.edu/dsrqst/NGUYEN658089/'
filelist = [
'TarFiles/e20c.oper.an.sfc.6hr.128_167_2t.regn80sc.1900010100_1900123118-1955010100_1955123118.grb.nguyen658089.nc.tar',
'TarFiles/e20c.oper.an.sfc.6hr.128_167_2t.regn80sc.1956010100_1956123118-2010010100_2010123118.grb.nguyen658089.nc.tar']
for file in filelist:
    filename=dspath+file
    file_base = os.path.basename(file)
    print('Downloading',file_base)
    req = requests.get(filename, cookies = ret.cookies, allow_redirects=True, stream=True)
    filesize = int(req.headers['Content-length'])
    with open(file_base, 'wb') as outfile:
        chunk_size=1048576
        for chunk in req.iter_content(chunk_size=chunk_size):
            outfile.write(chunk)
            if chunk_size < filesize:
                check_file_status(file_base, filesize)
    check_file_status(file_base, filesize)
    print()
=======
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import subprocess
import concurrent.futures as con_fu
import numpy as np
from era5dl import batchDownload

# ---------------------------- API file ---------------------------- #
url = 'url: https://cds.climate.copernicus.eu/api'
key = 'key: 6a02eb5f-b843-4577-a1a8-d20be0e077de'
# verify = 'verify: 1'

if not os.path.exists(os.path.expanduser('~/.cdsapirc')):
    with open(os.path.expanduser('~/.cdsapirc'), 'w') as f:
        f.write('\n'.join([url, key]))

    with open(os.path.expanduser('~/.cdsapirc')) as f:
        print(f.read())

# -------------------------- VARIABLE SET -------------------------- #

year_synop=np.arange(1940,2023)
year_wrf_full = [1944,1945,1961,1964,1968,1971,1972,1983,2022]
year_base=np.arange(1980,2011)
level_special=['200','500','700','850']

# var_synop = ['10m_u_component_of_wind','10m_v_component_of_wind','2m_temperature']
# var_press_partial = ['relative_humidity','temperature']
var_base_sfc = ['mean_sea_level_pressure','sea_surface_temperature']
var_base = ['geopotential','vorticity','relative_humidity','specific_humidity']

var_press = ['geopotential','relative_humidity','specific_humidity','temperature','u_component_of_wind','v_component_of_wind']
var_sfc   = ['10m_u_component_of_wind','10m_v_component_of_wind','2m_dewpoint_temperature','2m_temperature','geopotential',
           'mean_sea_level_pressure','land_sea_mask','sea_surface_temperature','skin_temperature','soil_temperature_level_1',
           'soil_temperature_level_2','soil_temperature_level_3','soil_temperature_level_4','surface_pressure',
           'total_precipitation','volumetric_soil_water_layer_1','volumetric_soil_water_layer_2','volumetric_soil_water_layer_3','volumetric_soil_water_layer_4']
var_sfc_not_synop = ['geopotential',
           'mean_sea_level_pressure','land_sea_mask','sea_surface_temperature','skin_temperature','soil_temperature_level_1',
           'soil_temperature_level_2','soil_temperature_level_3','soil_temperature_level_4','surface_pressure',
           'total_precipitation','volumetric_soil_water_layer_1','volumetric_soil_water_layer_2','volumetric_soil_water_layer_3','volumetric_soil_water_layer_4']
# EXAMPLE CODE
TEMPLATE_DICT_PRESSURE = {
    'data_target': 'reanalysis-era5-pressure-levels',
    'product_type': 'reanalysis',
    'format': 'grib',
    'download_format': 'unarchived',
    'variable': [
        'geopotential', 
    ],
    'pressure_level': [
        '10','20','30','50',
        '70','100','125','150',
        '175','200','225','250',
        '300','350','400','450',
        '500','550','600','650',
        '700','750','775','800',
        '825','850','875','900',
        '925','950','975','1000',
    ],
    'year': [
        '1991', '1992', '1993',
    ],
    'month': [
        '01', '02', '03',
        '04', '05', '06',
        '07', '08', '09',
        '10', '11', '12',
    ],
    'day': [
        '01', '02', '03',
        '04', '05', '06',
        '07', '08', '09',
        '10', '11', '12',
        '13', '14', '15',
        '16', '17', '18',
        '19', '20', '21',
        '22', '23', '24',
        '25', '26', '27',
        '28', '29', '30',
        '31',
    ],
    'time': [
        "00:00", "01:00", "02:00",
        "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00",
        "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00",
        "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00",
        "21:00", "22:00", "23:00",  
    ],
    'area': [40, 80, -15, 150],
}

TEMPLATE_DICT_SINGLE = {
    'data_target': 'reanalysis-era5-single-levels',
    'product_type': 'reanalysis',
    'format': 'grib',
    'download_format': 'unarchived',
    'variable': [
        'geopotential', 
    ],
    'year': [
        '1991', '1992', '1993',
    ],
    'month': [
        '01', '02', '03',
        '04', '05', '06',
        '07', '08', '09',
        '10', '11', '12',
    ],
    'day': [
        '01', '02', '03',
        '04', '05', '06',
        '07', '08', '09',
        '10', '11', '12',
        '13', '14', '15',
        '16', '17', '18',
        '19', '20', '21',
        '22', '23', '24',
        '25', '26', '27',
        '28', '29', '30',
        '31',
    ],
    'time': [
        "00:00", "01:00", "02:00",
        "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00",
        "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00",
        "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00",
        "21:00", "22:00", "23:00" 
    ],
    'area': [40, 80, -15, 150],
}


# EXAMPLE ACTUAL REQUEST
dataset = 'reanalysis-era5-single-levels'
request = {
    "product_type": ["reanalysis"],
    "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
        "mean_sea_level_pressure",
        "sea_surface_temperature",
        "surface_pressure",
        "total_precipitation",
        "skin_temperature",
        "soil_temperature_level_1",
        "soil_temperature_level_2",
        "soil_temperature_level_3",
        "soil_temperature_level_4",
        "volumetric_soil_water_layer_1",
        "volumetric_soil_water_layer_2",
        "volumetric_soil_water_layer_3",
        "volumetric_soil_water_layer_4"
    ],
    "year": ["1942", "1943"],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "time": [
        "00:00", "01:00", "02:00",
        "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00",
        "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00",
        "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00",
        "21:00", "22:00", "23:00"
    ],
    "data_format": "grib",
    "download_format": "unarchived",
    "area": [40, 80, -15, 150]
}

# client = cdsapi.Client()
# client.retrieve(dataset, request).download()

# -------------------------- DOWNLOAD NOW -------------------------- #
OUTPUTDIR = '/data/projects/REMOSAT/tamnnm/wrf_data/gribfile/era5_6h_1940_2023/1942-1943/'


JOB_DICT_SINGLE = {
    'variable': var_sfc_not_synop,
    'year': ['1942','1943'],
}
OUTPUTDIR_SINGLE = os.path.join(OUTPUTDIR, 'single_level/')

JOB_DICT_PRESSURE_A = {
    'variable': var_press,
    'year': ['1942','1943'],
    'month': ["01","02","03","04","05","06"],
}

JOB_DICT_PRESSURE_B = {
    'variable': var_press,
    'year': ['1942','1943'],
    'month': ["07","08","09","10","11","12"],
}
OUTPUTDIR_PRESSURE = os.path.join(OUTPUTDIR, 'pressure_level/')

SKIP_LIST = []

# os.makedirs(OUTPUTDIR_SINGLE, exist_ok=True)
# os.chdir(OUTPUTDIR_SINGLE)
# batchDownload(TEMPLATE_DICT_SINGLE, JOB_DICT_SINGLE, SKIP_LIST, OUTPUTDIR_SINGLE, dry=False, pause=3)


os.makedirs(OUTPUTDIR_PRESSURE, exist_ok=True)
os.chdir(OUTPUTDIR_PRESSURE)
batchDownload(TEMPLATE_DICT_PRESSURE, JOB_DICT_PRESSURE_A, SKIP_LIST, OUTPUTDIR_PRESSURE, dry=False, pause=3)
batchDownload(TEMPLATE_DICT_PRESSURE, JOB_DICT_PRESSURE_B, SKIP_LIST, OUTPUTDIR_PRESSURE, dry=False, pause=3)
>>>>>>> c80f4457 (First commit)
