from matplotlib.colors import Normalize
from matplotlib.colors import from_levels_and_colors
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from typing import Optional, List
from my_junk import *
from geopy.distance import distance
from sub_TC import vort_cal
import xarray as xr
import os
import math
import numpy as np
import matplotlib
import pandas as pd
import sys
from copy import deepcopy
# ----------------------------------------- Import module - ---------------------------------------
# region

# end region

# ---------------------------- Constant ---------------------------- #
year = '1881'
path = "/data/projects/REMOSAT/tamnnm/iwtrc/ASEAN/grid_0.7/"
path_output = "/work/users/tamnnm/code/main_code/4.trop_cyc/tcdetection_code/python_ver/grid_0.7/"
os.chdir("/work/users/tamnnm/code/main_code/4.trop_cyc/tcdetection_code/python_ver/")


#! Original value from the paper
#! Slp_crit = 1005.4
#! Vort_crit = 2.825e-5
#! Z_crit = 1.115 (way too high so I change to 0.1)
#! Z_sym_crit = 14


# Critical value
# ? The first one is the name of the script
if len(sys.argv) > 1:
    test = False
    Slp_crit = float(sys.argv[1])
    Vort_crit = float(sys.argv[2])
    Z_thick_crit = float(sys.argv[3])
    Slp_crit_ano = float(sys.argv[4])
    Z_thick_re_crit = float(sys.argv[5])
else:
    test = True
    Slp_crit = 1012.5
    Vort_crit = 2.0
    Z_thick_crit = 0.8
    Slp_crit_ano = -2.0
    Z_thick_re_crit = 0.3
# if os.path.exists(f'res/res_{Slp_crit}_{Vort_crit}_{Z_thick_crit}.csv'):
#     raise KeyboardInterrupt("The result file already exists")

Z_sym_crit = 14
Distance_Cur_crit = 250  # 400  # km
Distance_Pre_crit = 250  # 400  # km
Distance_Sym_crit = 250  # 500
R_sym = 5
R_search = 4.5  # 6 degree - Centered box
max_tc = 1000  # Maximum number of TCs
max_obs = 200  # Maximum observation for each TC

ulon = 150
dlon = 100
ulat = 40
dlat = 5


# ----------------------------------------- Constant value - ---------------------------------------
Z_clim = xr.open_dataset(path+f'uwnd.{year}.nc')
R_time_name = find_name(Z_clim, fname_group='time')
Rx_name = find_name(Z_clim, fname_group='lon')
Ry_name = find_name(Z_clim, fname_group='lat')
R_level_name = find_name(Z_clim, fname_group='level')


# List of TCs
All_TC = [[{} for _ in range(max_obs)]
          for _ in range(max_tc)]  # List to hold all TC
num_obs = np.full(max_tc, -1)  # List to hold number of observation for each TC
# ? Start from 1 so that the first observation will be 0
pre_tcs = []  # List to hold previous TC
sample_TC = {"id": 0, "lat": 0, "lon": 0, "time": 0, "pmin": 0,
             "vort": 0, "abs_thick": 0, "re_thick": 0, "sym": 0}

# Initial value
time_step = 1  # Time step for each iteration
num_tc = 0  # Total number of TC
pre_num = 0  # Previous number of TC


def check_file(path):
    if os.path.exists(path):
        return True
    else:
        return False

# region- Function for this dude
# --------------------- Function for this dude --------------------- #


def nan_test(data):
    if np.any(np.isnan(data.values)):
        raise ValueError("There is Nan value in the data")
    else:
        return "No Nan value in the data"

#* Function for open file combining the two


def open_file(name, sel_year=None):  # , land_mask=land_mask):

    # ! Squeeze unnecessary dimension (here is the level)
    # Load the data
    if sel_year is None:
        data = xr.open_dataset(path + f'{name}.nc')
    else:
        data = xr.open_dataset(path + f'{name}.{sel_year}.nc')
    var_name = find_name(data, fname_group=name.split('_')[0], opt='var')
    # Check if the data is a Dataset or DataArray
    if isinstance(data, xr.Dataset):
        # If it's a Dataset, select a specific variable
        # Replace 'variable_name' with the actual variable name
        DArray1 = data[var_name]
    elif isinstance(data, xr.DataArray):
        DArray1 = data
    else:
        raise TypeError(
            "The loaded data is neither a Dataset nor a DataArray.")

    Dndim = DArray1.ndim
    if Dndim > 2:
        if Dndim == 4:
            DArray1 = DArray1.transpose(
                R_level_name, R_time_name, Rx_name, Ry_name).squeeze()
        if Dndim == 3:
            DArray1 = DArray1.transpose(
                R_time_name, Rx_name, Ry_name).squeeze()
        DArray2 = ctime_short(DArray1)
    elif Dndim == 2:
        DArray2 = DArray1.transpose(Rx_name, Ry_name).squeeze()
    else:
        DArray2 = DArray1

    # ! Take into all land pixels
    cut_DArray2 = cut_co(DArray2, ulat, dlat, ulon,
                         dlon, full=True, data=False)
    nan_test(cut_DArray2)
    return cut_DArray2
    # ! If don't, we can have All-Nan slice
    # return land_filter(DArray2, land_mask)

# * Function for min-max index

# Custom max function with skipna=True by default


def max_skipna(dataarray, **kwargs):
    return dataarray.max(skipna=True, **kwargs).values


def min_skipna(dataarray, **kwargs):
    return dataarray.min(skipna=True, **kwargs).values


def min_max_index(data, min_max, x_i=0, y_i=0):
    shape_0 = data.shape[0]
    if min_max == 'min':
        ind = np.nanargmin(data)
        value = np.nanmin(data)
    elif min_max == 'max':
        ind = np.nanargmax(data)
        value = np.nanmax(data)
    r_x = ind % shape_0
    r_y = ind // shape_0
    return value, r_x+x_i, r_y+y_i


def level_short(data, level: Optional[Union[int, List[int]]] = None):
    result = []
    if type(level) is int:
        if type(level) is not int:
            raise ValueError("Level must be integer")
        else:
            result = data.sel({R_level_name: level})
    elif type(level) is list:
        for lev in level:
            if type(lev) is not int:
                raise ValueError("Level must be integer")
            else:
                result += data.sel({R_level_name: lev})
    else:
        raise ValueError("Level must be integer or list of integer")
    return result


def geodistance(point1, point2):
    return distance(point1, point2).km

# endregion


# Data Array
print("Start loading data \n")
land_mask = open_file('land')
land_mask = cut_co(land_mask, ulat, dlat, ulon, dlon, full=True, data=False)
SLP = open_file('prmsl', year)  # !Remember to check if it's hPa or Pa
Z_sym = open_file('hgt_sym', year)  # m
Z_thick = open_file('hgt_thick', year)  # m
U = open_file('uwnd', year)  # m/s
V = open_file('vwnd', year)  # m/s


# # Numpy Array
# Zo_300, Zo_600, Zo_850, Zo_900 = level_short(Z, level=[300, 600, 850, 900])
# Zo_thick = Zo_300 - Zo_850
# Zo_sym = Zo_600 - Zo_900

# Zclim_300, Zclim_600, Zclim_850, Zclim_900 = level_short(
#     Z_clim, level=[300, 600, 850, 900])
# Zclim_thick = Zclim_300 - Zclim_850

# Z_thick = Zo_thick - Zclim_thick
# Z_sym = Zo_sym
# ? Don't need anomaly for 600-900 thinkness
print("Data loaded, Define fixed parameters\n")
try:
    U_850 = level_short(U, 850)
    V_850 = level_short(V, 850)
except:
    U_850 = U
    V_850 = V

# Coordination array
Rx = SLP[Rx_name].values
Ry = SLP[Ry_name].values
R_time = SLP[R_time_name].values
Dx = Rx[1]-Rx[0]
Dy = Ry[1]-Ry[0]
Nx = len(Rx)
Ny = len(Ry)
N_time = len(R_time)
Rx_idx_search = abs(int(R_search//Dx))
Ry_idx_search = abs(int(R_search//Dy))
Rx_idx_sym = abs(int(R_sym//Dx))
Ry_idx_sym = abs(int(R_sym//Dy))

# Vorticity
# print("Calculation of vorticity\n")
Vort = vort_cal(Uwind=U_850, Vwind=V_850, Rx=Rx, Ry=Ry,
                R_time=R_time, name=path+f'vort.{year}.nc')


# TODO: Search minimum SLP
for t in range(0, N_time, time_step):
    def ext_time(data, t=t):
        return data[t, :, :].values.squeeze()

    # List of data variables
    data_vars = [SLP, Vort, Z_thick, Z_sym]
    # Extract time slices and store in corresponding variables
    SLP_loc, Vort_loc, Z_thick_loc, Z_sym_loc = [
        ext_time(data) for data in data_vars]

    cur_num = 0  # Restart for each time-step
    cur_tcs = []  # List to hold current TC
    iter_num = 0  # Number of iteration for each time-step

    print(f"Start processing time step: {t}/{N_time-1} - {R_time[t]} \n")
    # Start from the top left (overlap each 3 degree)
    # Create overlapping box

    # ! This
    for i in range(0, Nx, 1):
        for j in range(Ny-1, 0, -1):

            # for i in range(0, Nx, Rx_idx_search // 2):
            # for j in range(Ny-1, 0, - Ry_idx_search // 2):

            # ! This failed totally
            # for i in range(Rx_idx_search, Nx, Rx_idx_search // 2):
            #     for j in range(Ry_idx_search, Ny, Ry_idx_search // 2):

            # ! This got the right point put wrong time step
            # for i in range(Nx-1, 0, - Rx_idx_search // 2):
            # for j in range(Ny - Ry_idx_search, 0, -Ry_idx_search):

            # ! This accurate for one single point time 77, 106.1, 18.5
            # for i in range(Nx-1, 0, - Rx_idx_search // 2):
            #     for j in range(Ny-1, 0, - Ry_idx_search // 2):
            iter_num += 1
            print(
                f"Start processing point: {Rx[i]}, {Ry[j]} \n The current iteration is {iter_num} \n")
            i1 = i-Rx_idx_search
            i2 = i+Rx_idx_search+1
            j1 = j-Ry_idx_search
            j2 = j+Ry_idx_search+1

            if i1 < 0:
                i1 = 0
            if i2 > Nx:
                i2 = Nx
            if j1 < 0:
                j1 = 0
            if j2 > Ny:
                j2 = Ny

            # if i1 < 0 or i2 > Nx or j1 < 0 or j2 > Ny:
            #     print("Skip point")
            #     continue
            # Check point: If False, the loop will break
            new_obs_FLAG = True
            # If True, it raises the total num_tc
            new_TC_FLAG = False

            # ? This return the index in whole range
            SLP_subloc = SLP_loc[i1:i2, j1:j2]
            SLP_min, x_c, y_c = min_max_index(SLP_subloc, 'min', i1, j1)
            # * Task A: Check whether it has been recorded in the CURRENT timestep
            for i in range(0, cur_num):
                comp_tc = cur_tcs[i]
                Distance = geodistance(
                    (Ry[y_c], Rx[x_c]), (comp_tc["lat"], comp_tc["lon"]))

                if Distance < Distance_Cur_crit:
                    print(
                        f"The TC: {Rx[x_c]}, {Ry[y_c]} is already analyzed \n")
                    new_obs_FLAG = False

            # + Check point A
            if not new_obs_FLAG:
                continue

            print(f"Looking at point:{Rx[x_c], Ry[y_c]}")
            # * Test 0: SLP
            if SLP_min < Slp_crit:
                ic1 = x_c - Rx_idx_search
                ic2 = x_c + Rx_idx_search + 1
                jc1 = y_c - Ry_idx_search
                jc2 = y_c + Ry_idx_search + 1
                if ic1 < 0:
                    ic1 = 0
                if ic2 > Nx:
                    ic2 = Nx
                if jc1 < 0:
                    jc1 = 0
                if jc2 > Ny:
                    jc2 = Ny
            else:
                print(f"Test 0: critical SLP, {SLP_min} \n")
                continue

            # * Test 1: critical SLP anomaly
            # ? Maximum of 2-d array is an array -> Convert to scalar

            Slp_ano = SLP_min - SLP_loc[ic1:ic2, jc1:jc2].mean()
            if Slp_crit_ano < 0:
                if Slp_ano > Slp_crit_ano:
                    # + Check point 1
                    print(
                        f"Test 1: critical SLP anomaly \n SLP anomaly is too small: {Slp_ano} \n")
                    continue

            # * Test 2: Vorticity test
            Vort_subloc = Vort_loc[ic1:ic2, jc1:jc2]
            # ? Maximum of 2-d array is an array -> Convert to scalar
            Vort_max = Vort_subloc.max()
            if Vort_max <= Vort_crit * 1e-5:
                # + Check point 2
                print(
                    "Test 2: Vorticity test \n Vorticity is smaller than the critical value: {Vort_max} \n")
                continue

            # * Test 3: Relative and absolute thickness
            # print("Test 3: Relative and absolute thickness \n")
            Z_thick_subloc = deepcopy(Z_thick_loc[ic1:ic2, jc1:jc2])
            land_subloc = land_mask[ic1:ic2, jc1:jc2]
            # + Sub_test 3.1: Check if the maximum thickness at the max abs thickness is positive
            Z_nb_points = Z_thick_subloc.size  # * 0.1
            i1_run = 0
            while True:
                # ? x_abs_max, y_abs_max is the index in the subloc, NOT IN THE WHOLE RANGE
                Z_abs_max, x_abs_max, y_abs_max = min_max_index(
                    Z_thick_subloc, "max")
                # ? Vorticity at max must > 0 and must be on the sea
                if Vort_subloc[x_abs_max, y_abs_max] > 0 and land_subloc[x_abs_max, y_abs_max] == 0:
                    break
                else:
                    i1_run += 1
                    if i1_run >= Z_nb_points:
                        new_obs_FLAG = False
                        print(
                            f"Sub_test 3.1: Check if the maximum thickness at the max abs thickness is positive \n Cannot find the maximum thickness \n")
                        break
                    Z_thick_subloc[x_abs_max, y_abs_max] = 0

            # + Check point 3.1
            if not new_obs_FLAG:
                continue

            # + Sub_test 3.2: Check if the maximum and relative thickness > the critical value
            # Box to test maximum thickness anomaly (Radius = R_search)
            im1 = x_abs_max + ic1 - Rx_idx_search
            im2 = x_abs_max + ic1 + Rx_idx_search + 1
            jm1 = y_abs_max + jc1 - Ry_idx_search
            jm2 = y_abs_max + jc1 + Ry_idx_search + 1

            if im1 < 0:
                im1 = 0
            if im2 > Nx:
                im2 = Nx
            if jm1 < 0:
                jm1 = 0
            if jm2 > Ny:
                jm2 = Ny

            Z_avg = Z_thick_loc[im1:im2, jm1:jm2].mean()
            Z_re_max = Z_abs_max - Z_avg
            if not ((Z_abs_max > Z_thick_crit and Z_re_max > Z_thick_re_crit)
                    or (Z_re_max > Z_thick_crit and Z_abs_max > 0)):
                # + Check point 3.2
                print(
                    f"Sub_test 3.2: Check if the maximum and relative thickness > the critical value \n Thickness is smaller than the critical value: {Z_abs_max, Z_re_max, Z_avg} \n")
                continue

            # #! Debug
            # if pre_tcs != []:
            #     print("Check 1 - The pre_tcs:", t,  iter_num, pre_tcs)

            cur_obs = {}  # Dictionary to hold current observation
            cur_obs["id"] = 0
            cur_obs["lon"] = Rx[x_c]
            cur_obs["lat"] = Ry[y_c]
            cur_obs["time"] = t
            cur_obs["pmin"] = SLP_min
            cur_obs["ano"] = Slp_ano
            cur_obs["vort"] = Vort_max
            cur_obs["re_thick"] = Z_re_max
            cur_obs["abs_thick"] = Z_abs_max
            cur_obs["land"] = False
            cur_obs["sym"] = -99.99

            # #! Debug
            # if pre_tcs != []:
            #     print("Check 2 - The pre_tcs:", t,  iter_num, pre_tcs)

            # * Task B: Check if the TC recorded in the PREVIOUS timestep
            # print("Task B: Check if the TC recorded in the PREVIOUS timestep \n")
            pre_exist_FLAG = False  # Check if the previous TC exist
            Distance_pre_ref = 0  # Distance between current and previous TC
            pre_idx_list = []  # List of previous TC that close to the current TC

            for i in range(0, pre_num):
                comp_tc = pre_tcs[i]
                Distance_pre = geodistance(
                    (cur_obs["lat"], cur_obs["lon"]), (comp_tc["lat"], comp_tc["lon"]))
                if Distance_pre <= Distance_Pre_crit and Distance_pre != 0:
                    pre_exist_FLAG = True
                    pre_idx_list.append([i, Distance_pre])
                    if Distance_pre < Distance_pre_ref:
                        Distance_pre_ref = Distance_pre
                        pre_obs_final = comp_tc
                        pre_idx_final = i

            if len(pre_idx_list) > 1:
                print(f"The current TC is close to more than 1 previous TC \n \
                                                The center now is {x_c}, {y_c}")

            # NEW!!!!!!!!
            # * Test 4: Symmetry
            # ? Only do when there is a previous TC exist
            if pre_exist_FLAG:
                # Box to test symmetry (Radius = R_sym)
                x1_sym = x_c - Rx_idx_sym
                x2_sym = x_c + Rx_idx_sym + 1
                y1_sym = y_c - Ry_idx_sym
                y2_sym = y_c + Ry_idx_sym + 1

                # if x1_sym < 0 or x2_sym > Nx or y1_sym < 0 or y2_sym > Ny:
                # print("Test 4: Symmetry \n Symmetry box is out of range \n \
                #         The center now is {x_c}, {y_c} \n \
                #         The id of the previous TC is {pre_obs['id']}")
                if x1_sym < 0:
                    x1_sym = 0
                if x2_sym > Nx:
                    x2_sym = Nx
                if y1_sym < 0:
                    y1_sym = 0
                if y2_sym > Ny:
                    y2_sym = Ny

                # ? If the closest TC is not fit the symmetry criteria, test with other close TC
                    # ? Sort the list of previous TC by distance
                    # ? Extract the first element - index in the pre_tcs
                pre_idx_sorted = sorted(pre_idx_list, key=lambda x: x[1])
                for i in range(len(pre_idx_sorted)):
                    pre_obs = pre_tcs[pre_idx_sorted[i][0]]
                    x_pre = pre_obs["lat"]
                    y_pre = pre_obs["lon"]

                    # Initial value for two semi-circle
                    left_sum = 0
                    left_count = 0
                    right_sum = 0
                    right_count = 0
                    # Vector of two center
                    vec_cen = np.array(
                        [y_pre - cur_obs["lat"], cur_obs["lon"] - x_pre])
                    for i in range(x1_sym, x2_sym):
                        for j in range(y1_sym, y2_sym):
                            Distance_sym = geodistance(
                                (cur_obs["lat"], cur_obs["lon"]), (Ry[j], Rx[i]))
                            if Distance_sym <= Distance_Sym_crit and Distance_sym != 0:
                                vec = np.array(
                                    [Rx[i]-x_pre, Ry[j] - y_pre])
                                val_point = Z_sym_loc[i, j]
                                if np.dot(vec, vec_cen) > 0:
                                    right_sum += val_point
                                    right_count += 1
                                else:
                                    left_sum += val_point
                                    left_count += 1
                    # The storm-motion-relative 900–600-hPa thickness asymmetry
                    B = (right_sum/right_count) - (left_sum/left_count)
                    if B <= Z_sym_crit and B >= 0:
                        cur_obs["id"] = pre_obs["id"]
                        cur_obs["sym"] = B
                        # ? Satistfy the symmetry criteria
                        break
                    else:
                        if i == math.ceil(len(pre_idx_sorted)/2):
                            print(
                                f"Test 4: Symmetry \nThe current TC symmetry is not satisfied \n ")
                            new_obs_FLAG = False
                            break

            else:
                #     continue
                # else:
                cur_obs["id"] = num_tc
                new_TC_FLAG = True

            if land_mask[x_c, y_c] == 1:
                print(f"The center is a new TC and on the land \n")
                cur_obs["land"] = True

            # + Check point 4
            if not new_obs_FLAG:
                continue

            #!!!!!!! BIG WARNING - deep copy
            # ? If not, the change in cur_obs will affect pre_tcs + cur_tcs later on
            # ? copy.deepcopy is create a hard/deep copy vs a shallow copy if only assign
            # ? Deep copy = copy the value of the object, not the reference
            # ? Shallow copy = copy the reference of the object, not the value
            cur_obs_copy = deepcopy(cur_obs)

            # * Test C: Check whether the TC is already recorded in the ALL TC list
            # ? If the TC is totally new, it's automatically skip this
            # ? If the TC is already recorded and the mslp of the center is smaller, it will be updated
            # ? num_tc will not change
            for i in range(0, cur_num):
                comp_tc = cur_tcs[i]
                if cur_obs_copy["id"] == comp_tc["id"]:
                    # + Check point C
                    new_obs_FLAG = False
                    new_TC_FLAG = False
                    # print(
                    #    f" Test C: Check whether the TC is already recorded in the ALL TC list \nThe TC in timestep {t} is already recorded\n")
                    if cur_obs["pmin"] < comp_tc["pmin"]:
                        cur_tcs[i] = cur_obs_copy
                        print(
                            f"Update the center at timestep {t} with id {cur_obs_copy['id']} \n")
                        break

            # #! Debug
            # print("Before: num_cur_tc", cur_num, "num_all_tc", num_tc)

            # NEW!!!!!!!!
            if new_TC_FLAG:
                print(cur_obs)
                num_tc += 1
                # print(f"New TC is found at timestep {t} \n")

            # * Update the current TC list
            # ? If the TC is totally new, it will be registered here
            if new_obs_FLAG:
                cur_tcs.append(cur_obs_copy)
                cur_num += 1
                # print(R_time[t], cur_obs["id"],
                #       cur_obs["lat"], cur_obs["lon"], '\n')
                # print(f"The new center: {Rx[x_c]}, {Ry[y_c]} \n")

            # print("num_cur_tc", cur_num, "num_all_tc", num_tc)

    # * Update global All_TC list
    # print(f"Update global All_TC list \n")
    if cur_num > 0:
        for i in range(0, cur_num):
            tc_idx = cur_tcs[i]["id"]
            num_obs[tc_idx] += 1
            All_TC[tc_idx][num_obs[tc_idx]] = cur_tcs[i]

    if pre_tcs != []:
        print("The pre_tcs:", t, pre_tcs)
    if cur_tcs != []:
        print("The cur_tcs:", t, cur_tcs)

    # * Update the previous TC list
    # ? cur_num is a number -> not affected
    pre_num = cur_num
    # !!!!!!!!!!!!!! BIG WARNING - deep copy
    # ? We need to have deep copy -> if not, change in cur_tcs appear in the pre_tcs
    # ? Dict is MUTABLE so change in cur_obs will change the value in pre_tcs
    pre_tcs = deepcopy(cur_tcs)


print(f"Total number of TC: {num_tc} \n")

if num_tc == 0:
    raise KeyboardInterrupt("No TC is found")


# Remove the not-used slot
All_TC = All_TC[:num_tc]
# ? Must flatten the All_TC before add to the dataframe
All_TC_pdf = pd.DataFrame([obs for tc in All_TC for obs in tc])
# ? Remove all Nan value (empty observation)
All_TC_pdf = All_TC_pdf.dropna(how='all')
# ? Save the result
folder_output = f"{path_output}res_{Slp_crit_ano}/"

if not os.path.exists(folder_output):
    os.mkdir(folder_output)

All_TC_pdf.to_csv(
    f'{folder_output}res_{Slp_crit}_{Vort_crit}_{Z_thick_crit}_{Z_thick_re_crit}.csv' if not test else 'source/res_test.csv')
# All_TC_pdf.to_csv(f'res_test_{Slp_crit}_{Vort_crit}_{Z_thick_crit}.csv')
