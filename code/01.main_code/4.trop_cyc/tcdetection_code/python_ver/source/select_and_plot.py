import matplotlib.colors as mcolors
from matplotlib.colors import Normalize, from_levels_and_colors
from matplotlib.patches import Circle
import matplotlib.ticker as mticker
import matplotlib.axes as maxes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Optional, List
from my_junk import *
from geopy.distance import distance as geodistance
from matplotlib.ticker import FormatStrFormatter
import geopandas as gpd
from shapely.geometry import Point
import xarray as xr
import os
import numpy as np
from matplotlib import font_manager
import pandas as pd
import sys
import cartopy.crs as ccrs
import h5py
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

obs_track = [[113.5, 15], [110.46, 18.67], [110.3, 20.067], [110.3, 20.05]]
path = "/work/users/tamnnm/code/main_code/4.trop_cyc/tcdetection_code/python_ver/grid_0.7/"
path_data = "/data/projects/REMOSAT/tamnnm/iwtrc/ASEAN/grid_0.7/"

year = 1881
R_sym = 5
Z_sym_crit = 14
Distance_Sym_crit = 1000  # km (10deg)

ulon = 135
dlon = 100
ulat = 25
dlat = 5


# # Your font path goes here
font_dirs = '/work/users/tamnnm/code/2.sup/Hershey_font_TTF/ttf'
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)


def open_file(name, sel_year=None):  # , land_mask=land_mask):

    # ! Squeeze unnecessary dimension (here is the level)
    # Load the data
    if sel_year is None:
        data = xr.open_dataset(path_data + f'{name}.nc')
    else:
        data = xr.open_dataset(path_data + f'{name}.{sel_year}.nc')

    var_name = find_name(data, fname_group=name.split('_')[0], opt='var')
    # Check if the data is a Dataset or DataArray
    if isinstance(data, xr.Dataset):
        # If it's a Dataset, select a specific variable
        # Replace 'variable_name' with the actual variable name
        DArray1 = data[var_name].squeeze()
    elif isinstance(data, xr.DataArray):
        DArray1 = data.squeeze()
    else:
        raise TypeError(
            "The loaded data is neither a Dataset nor a DataArray.")
    return DArray1
    # ! Take into all land pixels
    # ! If don't, we can have All-Nan slice


def transpose(DArray1):
    if DArray1.ndim == 3:
        DArray1 = DArray1.transpose(R_time_name, Rx_name, Ry_name).squeeze()
        DArray2 = ctime_short(DArray1)
    else:
        DArray1 = DArray1.transpose(Rx_name, Ry_name).squeeze()
        DArray2 = DArray1
    return DArray2


def sym_test(x_c, y_c, time_c, pre_tcs):
    x_c_i = np.argmin(abs(Rx-x_c))
    y_c_i = np.argmin(abs(Ry-y_c))
    x1_sym = x_c_i - Rx_idx_sym
    x2_sym = x_c_i + Rx_idx_sym + 1
    y1_sym = y_c_i - Ry_idx_sym
    y2_sym = y_c_i + Ry_idx_sym + 1

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

    id, sym = None, None
    for index, pre_obs in pre_tcs.iterrows():
        if int(pre_obs['time']) - time_c > 16:
            continue
        if int(pre_obs['time']) <= time_c:
            break
        Z_sym_loc = Z_sym[int(pre_obs["time"])-1, :, :]
        x_pre = pre_obs["lat"]
        y_pre = pre_obs["lon"]

        # Initial value for two semi-circle
        left_sum = 0
        left_count = 0
        right_sum = 0
        right_count = 0
        # Vector of two center
        vec_cen = np.array(
            [y_pre - y_c, x_c - x_pre])
        for i in range(x1_sym, x2_sym):
            for j in range(y1_sym, y2_sym):
                if geodistance((y_c, x_c), (Ry[j], Rx[i])) <= Distance_Sym_crit:
                    vec = np.array([Rx[i]-x_c, Ry[j] - y_c])
                    val_point = Z_sym_loc[i, j]
                    if np.dot(vec, vec_cen) > 0:
                        right_sum += val_point
                        right_count += 1
                    else:
                        left_sum += val_point
                        left_count += 1
        # The storm-motion-relative 900–600-hPa thickness asymmetry
        B = (right_sum/right_count) - (left_sum/left_count)
        if B <= Z_sym_crit:
            id = pre_obs["id"]
            sym = B
            # ? Satistfy the symmetry criteria
            break
    return id, sym


Z_sym = open_file('hgt_sym', 1881)
Z_thick = open_file('hgt_thick', 1881)
Rx_name = find_name(Z_sym, fname_group='lon')
Ry_name = find_name(Z_sym, fname_group='lat')
R_time_name = find_name(Z_sym, fname_group='time')

land = transpose(cut_co(open_file('land'), ulat, dlat, ulon, dlon))
Z_sym = transpose(Z_sym)
Z_thick = transpose(Z_thick)

Z_thick = ctime_short(Z_thick)  # , start_date=pd.to_datetime(
# "1881-10-04"), end_date=pd.to_datetime("1881-10-06"))

Rx = Z_thick[Rx_name].values
Ry = Z_thick[Ry_name].values
R_time = Z_thick[R_time_name].values

Dx = Rx[1]-Rx[0]
Dy = Ry[1]-Ry[0]
Nx = len(Rx)
Ny = len(Ry)
Rx_idx_sym = abs(int(R_sym//Dx))
Ry_idx_sym = abs(int(R_sym//Dy))

gif = False
if gif:
    # Initialize the plot
    fig, ax = ini_plot(dlat, ulat, dlon, ulon, pad=5, figsize=[
        12, 7], label_size=10, title=R_time[0], grid=True)
    plot = ax.pcolormesh(
        Rx, Ry, Z_thick[0, :, :].T, cmap='viridis', alpha=0.5, zorder=1)
    cbar = fig.colorbar(plot, ax=ax, orientation='vertical',
                        pad=0.02, aspect=50)

    # Add text annotations for each data point
    # texts = []
    # for i in range(len(Rx)):
    #     for j in range(len(Ry)):
    #         text = ax.text(
    #             Rx[i], Ry[j], f'{Z_thick[0, i, j]:.2f}', ha='center', va='center', color='black', transform=ax.transData)
    #         texts.append(text)
    # Update function for animation

    def update_plot(frame):
        plot.set_array(Z_thick[frame, :, :].T.values.ravel())
        ax.set_title(R_time[frame])
        return plot
        # for text in texts:
        #     text.remove()
        # texts.clear()
        # for i in range(len(Rx)):
        #     for j in range(len(Ry)):
        #         text = ax.text(
        #             Rx[i], Ry[j], f'{Z_thick[frame, i, j]:.2f}', ha='center', va='center', color='black', transform=ax.transData)
        #         texts.append(text)
        # return plot, *texts

    # Create the animation
    ani = animation.FuncAnimation(
        fig, update_plot, frames=len(R_time), blit=False)

    # Save the animation as a GIF
    ani.save("test_thick.gif", writer='pillow', fps=2)

test = True
if not test:
    no_img = 0
    for folder in os.listdir(path):
        if folder.startswith("res"):  # and folder.endswith("3.0"):
            print(folder)
            for file in os.listdir(path+folder):
                if file.endswith("_new.csv"):
                    continue
                elif file.endswith(".csv"):
                    point_csv = pd.read_csv(path + folder + '/'+file)
                    for index, obs in point_csv.iterrows():
                        id, sym = sym_test(
                            obs["lat"], obs["lon"], obs["time"], point_csv)
                        if id and sym:
                            point_csv.at[index, 'id'] = id
                            point_csv.at[index, 'sym'] = sym
                            # and len(np.unique(point_csv["id"])) < 3:
    #                point_csv.to_csv(path + folder + '/' + file[:-4] + "_new.csv")

                    point_csv = point_csv[(point_csv["time"] <= 105) & (
                        point_csv["lat"] >= 10) & (point_csv["lat"] <= 25)]

                    # len(point_csv[(point_csv["time"] <= 80)]) < 3 \

                    if len(point_csv[(point_csv["land"] == False)]) < 2 or len(point_csv) < 3:
                        continue

                    point_csv['real_time'] = pd.to_datetime([R_time[int(index)]
                                                            for index in point_csv['time']]) + pd.Timedelta(hours=7)
                    print(file)

                    lat_points = point_csv["lat"].values
                    lon_points = point_csv["lon"].values

                    fig, ax = ini_plot(dlat, ulat, dlon, ulon, pad=5,
                                       figsize=[12, 7], label_size=10, title="Test_trajectory", grid=True)

                    ax.scatter(lon_points, lat_points, color="red",
                               zorder=3,  label="Trajectory")
                    for i, time in enumerate(point_csv["real_time"]):
                        ax.annotate(
                            time.strftime('%m-%d:%H'), (lon_points[i] + 0.25, lat_points[i] + 0.25), fontsize=9, ha='left')

                    no_img += 1
                    fig.savefig(f"Trajectory_{no_img}.png", dpi=300)
                    point_csv.to_csv(f"test_new_{no_img}.csv")
                    plt.close()

if test:
    point_csv = pd.read_csv("./res_2_-2_130.csv")
    point_csv = point_csv[(point_csv["time"] <= 105) & (
        point_csv["lat"] >= 10) & (point_csv["lat"] <= 22) & (point_csv["lon"] >= 105)].sort_values(by=['time'])

    # len(point_csv[(point_csv["time"] <= 80)]) < 3 \

    # if len(point_csv[(point_csv["land"] == False)]) < 2 or len(point_csv) < 3:
    #     raise ValueError("Not enough data")

    point_csv['real_time'] = pd.to_datetime([R_time[int(index)]
                                             for index in point_csv['time']]) + pd.Timedelta(hours=7)
    point_csv = point_csv.sort_values(
        by=['time', 'lon'], ascending=[True, False])
    filtered_points = []
    first_thres = 120
    for i in range(len(point_csv)):
        if len(filtered_points) == 0:
            if point_csv.iloc[i]['lon'] > first_thres:
                filtered_points.append(point_csv.iloc[i])
        else:
            if point_csv.iloc[i]['time'] > filtered_points[-1]['time'] and point_csv.iloc[i]['lon'] < filtered_points[-1]['lon']:
                filtered_points.append(point_csv.iloc[i])

    filtered_df = pd.DataFrame(filtered_points)

    # Save the filtered DataFrame to a CSV file
    filtered_df.to_csv(f"filtered_test_new.csv")

    # # Identify duplicate rows and keep the last occurrence
    # duplicates = point_csv[point_csv.duplicated(
    #     subset=['lat', 'lon'], keep='last')]

    # # Identify non-duplicate rows
    # non_duplicates = point_csv[~point_csv.duplicated(
    #     subset=['lat', 'lon'], keep=False)]

    # # Combine duplicates and non-duplicates, ensuring no duplicates are included twice
    # point_csv = pd.concat([duplicates, non_duplicates]).drop_duplicates(
    #     subset=['lat', 'lon'], keep='last')

    # print(point_csv)

    lat_points = filtered_df["lat"].values
    lon_points = filtered_df["lon"].values
    point_dub = np.zeros(len(lat_points))

    fig, ax = ini_plot(5, 25, 100, 135, pad=5,
                       figsize=[12, 7], label_size=10, title="Test_trajectory", grid=True)
    # # Define the number of positions around each point
    # num_positions = len(point_csv)
    # angle_step = 2 * np.pi / num_positions

    ax.plot(lon_points, lat_points, color="red",
            zorder=3,  label="Trajectory")
    ax.scatter(lon_points, lat_points, color="red", marker='o', s=100,
               zorder=3,  label="Trajectory")
    # Draw circles at each point
    for lon, lat in zip(lon_points, lat_points):
        circle = Circle((lon, lat), radius=2,
                        color='red', fill=True, zorder=3, alpha=0.5)
        ax.add_patch(circle)

    for i, time in enumerate(filtered_df["real_time"]):
        # # Calculate the position for the annotation
        # angle = (i % num_positions) * angle_step
        # # Adjust the multiplier to control the distance
        # offset_x = np.cos(angle) * 2.0
        # # Adjust the multiplier to control the distance
        # offset_y = np.sin(angle) * 2.0

        # ax.annotate(
        #     time.strftime('%m-%d:%H'),
        #     xy=(lon_points[i], lat_points[i]),  # Original point
        #     xytext=(lon_points[i] * 1.1, lat_points[i]
        #             * 1.13
        #             ),  # Offset position
        #     fontsize=9,
        #     ha='right',
        #     arrowprops=dict(arrowstyle="->", color='black')
        # )

        # ax.annotate(
        #     time.strftime('%m-%d:%H'),
        #     xy=(lon_points[i], lat_points[i]),  # Original point
        #     xytext=(lon_points[i] * 1.1, lat_points[i]
        #             * 1.13
        #             ),  # Offset position
        #     fontsize=9,
        #     ha='right',
        #     arrowprops=dict(arrowstyle="->", color='black')
        # )
        if lat_points[i] > 20 or (lat_points[i] > 15 and lat_points[i] < 17):
            rotate = -30
        else:
            rotate = -30

        ax.annotate(
            time.strftime('%m-%d:%H'), (lon_points[i]+0.01, lat_points[i]-2.5), fontsize=15, ha='left', rotation=rotate)

    fig.savefig(f"Trajectory_test.png", dpi=300)
    plt.close()

# # Plot the land-sea mask
# mesh = ax.pcolormesh(land_mask.longitude,  # Plot the land-sea mask
#                      land_mask.latitude, land_mask, transform=ccrs.PlateCarree(), cmap='Greys', alpha=0.5)

# # Add a colorbar
# cbar = fig.colorbar(mesh, ax=ax, orientation='vertical', pad=0.02, aspect=50)
# cbar.set_label('Land-Sea Mask')
# fig.savefig("test_land.png", dpi=300)
