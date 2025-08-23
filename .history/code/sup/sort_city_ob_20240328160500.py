import os

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
ind_info = []
rain_city = []
temp_city = []
uwnd_city = []
vwnd_city = []
list_city = [rain_city, temp_city, uwnd_city, vwnd_city]
list_city_info = []
list_city_only = []

# %%
for file_name in sorted(os.listdir(dat_thay)):
    file_full = f'{dat_thay}{file_name}'
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
    with open(file_full, 'r') as file:
        ind_info_ind = []
        lines = file.readlines()
        lat_lon_line = lines[1].strip().split()
        if float(lat_lon_line[0]) > 100:
            first_year = lines[2].strip()
            last_year = lines[-32].strip()
        else:
            first_year = lines[0].strip()
            last_year = lines[-32].strip()
            with open('city_list_obs/rain_city.txt', 'r') as file:
                lines = file.readlines()
                for line in lines:
                    name = line.strip().split(',')
                    if name[0] == name_element[1]:
                        lat_lon_line = [name[3], name[4]]
                        # print(lat_lon_line)
                    else:
                        continue

        ind_info_ind = ind_info_ind+[name_element[1], first_year, last_year]
        ind_info.append(ind_info_ind + lat_lon_line)
        list_city_only.append(name_element[1])
        list_city_info.append(ind_info_ind)

# print(list_city_info)

i = 0
x = 0
list_sample = list(set(tuple(city) for city in list_city_info))
list_city_info = []
# if you want to loop a list in a list, transfer to a tuple
for city_info_full in list_sample:
    x += 1
    # print(city_info,'\n')
    # count then you must use the original list and turn the tuple element back to list
    city_info_full = list(city_info_full)
    occ_city = list_city_info.count(city_info_full)
    city_info_full.append(occ_city)
    list_city_info.append(city_info_full)
print(len(set(list_city_only)), len(list_city_info), len(list_sample), i)
# print(rain_city)

def f_write(city_list)

with open('city_list_obs/rain_city.txt','w') as file:
    for city in sorted(rain_city):
        line =','.join(city)+'\n'
        file.write(line)

with open('city_list_obs/temp_city.txt','w') as file:
    for city in temp_city:
        line =','.join(city)+'\n'
        file.write(line)

with open('city_list_obs/uwnd_city.txt','w') as file:
    for city in uwnd_city:
        line =','.join(city)+'\n'
        file.write(line)

with open('city_list_obs/vwnd_city.txt','w') as file:
    for city in vwnd_city:
        line =','.join(city)+'\n'
        file.write(line)


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
