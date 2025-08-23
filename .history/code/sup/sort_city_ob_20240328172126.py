import os
os.chdir('/work/users/tamnnm/code/main_code/city_list_obs/')
# print test the dataset
"""
data_link="/work/users/student6/tam/Data/wrf_data/netcdf/air.2m.1967.nc"
dset=xr.open_dataset(data_link,decode_times=False)
units,reference_date=dset.time.attrs['units'].split('since')
dset['time']=pd.date_range(start=reference_date,periods=dset.sizes['time'])
print(dset['time'])
"""

# print test the station dataset
dat_thay = "/work/users/thanhnd/dat_OBS_grid/dat_daily_1961_2019/"
rain_city = []
temp_city = []
uwnd_city = []
vwnd_city = []

# %%
for file_name in sorted(os.listdir(dat_thay)):

    # Name of the file (e.g. R_ALUOI_1961_2019.txt)
    name_element = os.path.splitext(file_name)[0].split('_')
    if name_element[0] == "R":
        ind_info = rain_city
    elif name_element[0] == "T2m":
        ind_info = temp_city
    elif name_element[0] == "Um":
        ind_info = uwnd_city
    elif name_element[0] == "Vx":
        ind_info = vwnd_city
    else:
        continue

    ## Extract the information of each city file into a list
    with open(f'{dat_thay}{file_name}', 'r') as file:
        lines = file.readlines()
        # LIST OF [LAT, LON]
        lat_lon_line = lines[1].strip().split()
        # if float(lat_lon_line[0]) > 100:
        first_year = lines[2].strip()
        last_year = lines[-32].strip()

    # Create list of [city name, first year, last year, lat, lon]
    ind_info.append([name_element[1], first_year, last_year] + lat_lon_line)

def f_write(list_city_info,file_name):
    with open(f'city/{file_name}.txt','w') as file:
        # sorted by city name - the 1st element
        for city in sorted(list_city_info, key=lambda x: x[0]):
            line =','.join(city)+'\n'
            file.write(line)
    return

print(len(ra)
f_write(rain_city,'rain_city')
f_write(temp_city,'temp_city')
f_write(uwnd_city,'uwnd_city')
f_write(vwnd_city,'vwnd_city')

# with open('city_list_obs/rain_city.txt','w') as file:
#     for city in sorted(rain_city):
#         line =','.join(city)+'\n'
#         file.write(line)



# %%
"""
import subprocess
filepath = '/city_list_obs/'
def count_city(city,filename):
    total_occ = 0
    for filename in filepath:
        results = subprocess.run(['grep','-o',city,filename],capture_output=True,text=True)
        #number of occurences
        occ = results.stdout.count(city)
        total_occ += occ
        city_occ = [city,total_occ]
"""
