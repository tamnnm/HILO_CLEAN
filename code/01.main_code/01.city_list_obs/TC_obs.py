import time
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize, from_levels_and_colors
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.axes as maxes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List
from my_junk import *
from geopy.distance import distance as geodistance
from matplotlib.ticker import FormatStrFormatter
import geopandas as gpd
from shapely.geometry import Point
import xarray as xr
import os
import numpy as np
import matplotlib
import pandas as pd
import sys
import cartopy.crs as ccrs
import h5py
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


params = {
    'axes.titlesize': 40,
    'axes.labelsize': 25,
    'axes.labelpad': 15,
    'font.size': 15,
    'font.family': 'cmss10',
    'mathtext.fontset': 'stixsans',
    'legend.fontsize': 20,
    'legend.loc': 'upper right',
    'legend.labelspacing': 0.25,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'lines.linewidth': 3,
    # 'text.usetex': True,
    # 'figure.autolayout': True,
    'ytick.right': False,
    'xtick.top': False,

    'figure.figsize': [12, 10],  # instead of 4.5, 4.5
    'axes.linewidth': 1.5,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.minor.size': 5,
    'ytick.minor.size': 5,

    'xtick.major.width': 3,
    'ytick.major.width': 3,
    'xtick.minor.width': 3,
    'ytick.minor.width': 3,

    'xtick.major.pad': 10,
    'ytick.major.pad': 12,
    # 'xtick.minor.pad': 14,
    # 'ytick.minor.pad': 14,

    'xtick.direction': 'inout',
    'ytick.direction': 'inout',
    'axes.unicode_minus': False,
}
plt.clf()
matplotlib.rcParams.update(params)

path = "/data/projects/REMOSAT/tamnnm/obs/"

# ------ Extract the track of TC in the South East Asia region ----- #
# if os.path.exiflag("IBTrACS.SEA.nc"):
#     track_VN = xr.open_dataset("IBTrACS.SEA.nc")

# * Define compression settings
# track_path = "IBTrACS.SEA.nc"
track_WP_path = path + "IBTrACS.WP.v04r00.nc"
# track_VN = xr.open_dataset(track_path)
track_WP = xr.open_dataset(track_WP_path)

# * Ranking the category of TC
TC_rank = ("TD", "TS", "CAT 1", "CAT 2", "CAT 3", "CAT 4", "CAT 5")
# TC_rank = ("CAT 1", "CAT 2", "CAT 3", "CAT 4", "CAT 5")

# * Colormap
# Get the jet colormap
cmap = sns.color_palette("magma", as_cmap=True)
# Generate colors for each category by sampling the colormap
cat_colors = [cmap(i / (len(TC_rank) - 1)) for i in range(len(TC_rank))]

# Only include CAT 1 to CAT 5
cat_colors = cat_colors[1:]
# Create a colormap and norm based on the category colors
cmap = mcolors.ListedColormap(cat_colors)
norm = mcolors.BoundaryNorm(boundaries=range(
    len(cat_colors)+1), ncolors=len(cat_colors))

ulon = 150
dlon = 100
ulat = 25
dlat = 5

# Rx_name = find_name(track, fname_group='lon')
# Ry_name = find_name(track, fname_group='lat')
# R_time_name = find_name(track, fname_group='time')
# Rx = track[Rx_name]
# Ry = track[Ry_name]
# R_time = track[R_time_name]
# track_VN = cut_co(track, ulat=40, dlat=-5, ulon=150, dlon=100,
#                   name=[Ry_name, Rx_name], full=True, data=False)

# compression_settings = {
#     'zlib': True,
#     # Compression level (1-9), higher means more compression
#     'complevel': 5
# }
# track_VN.to_netcdf("IBTrACS.SEA.nc", encoding={
#     var: compression_settings for var in track_VN.data_vars})

# ****** Plot the track of TC in the South East Asia region *****#


def extract_info(track_data, storm):
    track_loc = track_data.sel(storm=storm)
    lat_loc = track_loc.lat.values
    lon_loc = track_loc.lon.values
    lon_lat_loc = np.stack((lon_loc, lat_loc), axis=-1)
    time_loc = track_loc.time.values
    cat_loc = track_loc.usa_sshs.values
    landfall_loc = track_loc.landfall.values

    SEA_flag, max_SEA_flag, gen_SEA_flag = True, True, True
    land_SEA_flag = False

    # *** Filter nan value of the SEA region

    # ? We assume that all the non-Nan lat-lon has accordingly non-Nan time value
    mask_WP = ~np.isnan(lon_lat_loc).any(axis=1)
    # * CHECK POINT 1
    # !!! The track doesn't exist
    if not mask_WP.any():
        return {}, {}
    gen_index_WP = np.where(mask_WP)[0][0]

    # Create a mask to filter out rows with NaN values
    if np.nanmax(lat_loc) > ulat or np.nanmin(lat_loc) < dlat \
            or np.nanmax(lon_loc) > ulon or np.nanmin(lon_loc) < dlon:
        # print(storm)
        filter_lat = np.where(
            (lat_loc <= ulat) & (lat_loc >= dlat), lat_loc, np.nan)
        filter_lon = np.where(
            (lon_loc <= ulon) & (lon_loc >= dlon), lon_loc, np.nan)
        # Stack the coordinates
        lon_lat_SEA = np.stack((filter_lon, filter_lat), axis=-1)

        mask_SEA = ~np.isnan(lon_lat_SEA).any(axis=1)
        # !!!! The track is outside the SEA region
        if not mask_SEA.any():
            return {}, {}
        # Temporary flag
        SEA_flag = False
        gen_index_SEA = np.where(mask_SEA)[0][0]

    # *** Calculate the indices
    # ! All the track is inside of the SEA region
    if SEA_flag:
        mask_SEA = mask_WP
        max_idx = np.nanargmax(cat_loc)
        gen_idx = gen_index_WP
    else:
        # * CHECK POINT 2
        # ! There exits a part of the track in the SEA region
        SEA_flag = True

        # * Find the index of maximum value of category
        max_idx_WP = np.nanargmax(cat_loc[mask_WP])
        max_idx_SEA = np.nanargmax(cat_loc[mask_SEA])

        #! Max value of category will always be in the WP region
        #! We just need to check the location of that max value
        max_SEA_flag = max_idx_WP == max_idx_SEA
        # ? The maximum index is not in the SEA region
        max_idx = max_idx_SEA if max_SEA_flag else max_idx_WP

        # * Find the index of the genesis
        gen_SEA_flag = gen_index_WP >= gen_index_SEA
        # ? The genesis index is not in the SEA region
        gen_idx = gen_index_SEA if gen_SEA_flag else gen_index_WP

    # *** Finalize all the information
    cat_max = int(cat_loc[max_idx])
    time_max = time_loc[max_idx]
    lat_max = lat_loc[max_idx]
    lon_max = lon_loc[max_idx]

    cat_gen = int(cat_loc[gen_idx])
    time_gen = time_loc[gen_idx]
    lat_gen = lat_loc[gen_idx]
    lon_gen = lon_loc[gen_idx]

    year_gen = int(pd.to_datetime(time_gen).to_pydatetime().year)

    # *** Check landfall
    prev_0 = False
    for i in range(len(landfall_loc)):
        if landfall_loc[i] == 0: # The TC is going through land
            if prev_0: # If the previous point is also land,skip this point
                continue
            if in_vietnam(lat_loc[i], lon_loc[i]):
                land_SEA_flag = True
                land_idx = i
                break
            else:
                prev_0 = True
                continue
        else:
            prev_0 = False
    
    if land_SEA_flag:
        cat_land = int(cat_loc[land_idx])
        time_land = time_loc[land_idx]
        lat_land = lat_loc[land_idx]
        lon_land = lon_loc[land_idx]
    else:
        cat_land = None
        time_land = None
        lat_land = None
        lon_land = None

    # *** Nearest grid point

    return ({"lon_lat_full": lon_lat_loc[mask_WP],
            "time_full": time_loc[mask_WP],
             "cat_full": cat_loc[mask_WP],
             "landfall_full": landfall_loc[mask_WP],

             "lon_lat": lon_lat_loc[mask_SEA],
             "time": time_loc[mask_SEA],
             "cat": cat_loc[mask_SEA]},

            {"year": year_gen,

             "cat_max": cat_max,
             "time_max": time_max,
             "lat_max": lat_max,
             "lon_max": lon_max,

             "cat_gen": cat_gen,
             "time_gen": time_gen,
             "lat_gen": lat_gen,
             "lon_gen": lon_gen,

             "cat_land": cat_land,
             "time_land": time_land,
             "lat_land": lat_land,
             "lon_land": lon_land,

             "max_SEA_flag": max_SEA_flag,
             "gen_SEA_flag": gen_SEA_flag,
             "land_SEA_flag": land_SEA_flag})


def save_dict_to_hdf5(dicts, filename):
    string_dt = h5py.string_dtype(encoding='utf-8', length=None)
    with h5py.File(filename, 'w') as f:
        num_digits = len(str(len(dicts)))
        for i, d in enumerate(dicts):
            group = f.create_group(f'dict_{i:0{num_digits}d}')
            for key, value in d.items():
                if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.datetime64):
                    # Convert datetime64 to string
                    value = np.array([np.datetime_as_string(
                        n, timezone='UTC').encode('utf-8') for n in value])
                    group.create_dataset(
                        key, data=value, maxshape=(None), chunks=True)
                else:
                    group.create_dataset(key, data=value)


def read_hdf5_to_dict(filename):
    def read_group(group):
        data_dict = {}
        for key, value in group.items():
            if isinstance(value, h5py.Group):
                # Recursively read the group
                data_dict[key] = read_group(value)
            else:
                data = value[()]
                if isinstance(data, np.ndarray) and data.dtype.kind == 'S':
                    longest_string = len(max(data, key=len))

                    # Convert byte string array to Unicode string array
                    data = data[:].astype('U'+str(longest_string))
                    # Check if it can be converted to datetime64
                    try:
                        data = data.astype('datetime64[s]')
                    except ValueError:
                        pass
                data_dict[key] = data
        return data_dict

    with h5py.File(filename, 'r') as f:
        return list(read_group(f).values())



def idx_grid_single(value, real_grid, real_value=False, **kwargs):
    if real_grid is None:
        raise ValueError("real_grid must be provided")

    dif = abs(int(value) - real_grid)
    id = np.argmin(np.abs(dif))
    if real_value:
        return real_grid[id]
    else:
        return id

def idx_grid_point(lon, lat, lon_grid, lat_grid, real_value=False, **kwargs):
    return idx_grid_single(lon, lon_grid), idx_grid_single(lat, lat_grid)


def idx_grid_full(lon_lat_array, lat_grid, lon_grid, real_value=False, **kwargs):
    lat_data = lon_lat_array[:, 1]
    lon_data = lon_lat_array[:, 0]

    lat_final_data = []
    lon_final_data = []

    for lat in lat_data:
        lat_fin = idx_grid_single(value=lat, real_grid=lat_grid)
        lat_final_data.append(lat_fin)

    for lon in lon_data:
        lon_fin = idx_grid_single(value=lon, real_grid=lon_grid)
        lon_final_data.append(lon_fin)

    return lon_final_data, lat_final_data


def remove_leading_zeros(arr):
    first_non_zero_idx = np.argmax(arr != 0)
    return arr[first_non_zero_idx:]


full_file = "TC_SEA_full.h5"
loc_file = "TC_SEA.csv"

if not os.path.exists(full_file) or not os.path.exists(loc_file):
    # * Extract the information of TC
    TC_dict = []
    TC_SEA_full = []
    num_TC = 0
    plotted_storms = 0
    # * Creation of TC list and plot the track

    for storm in track_WP.storm:
        # *  Information of the storm
        loc_info_full, loc_info = extract_info(track_WP, storm)

        # * CHECK POINT A
        if loc_info == {}:
            continue

        # * CHECK POINT B
        if loc_info["cat_max"] < -1 or not loc_info["gen_SEA_flag"]:
            continue

        storm_id = storm.values
        loc_info["storm_id"] = storm_id

        # * Update the list of TC
        TC_dict.append(loc_info)
        TC_SEA_full.append(loc_info_full)
        num_TC += 1

    # * Save the data
    TC_SEA = pd.DataFrame([tc for tc in TC_dict])
    TC_SEA = TC_SEA.dropna(how='all')
    TC_SEA.to_csv(loc_file)
    save_dict_to_hdf5(TC_SEA_full, full_file)
else:
    print("The data is already existed")
    grid_size = 1
    grid_plot = np.zeros((int((ulat - dlat)//grid_size),
                         int((ulon - dlon)//grid_size)))

    lon_grid = np.arange(dlon, ulon+grid_size, grid_size)
    lat_grid = np.arange(dlat, ulat+grid_size, grid_size)
    no_grid_lon = len(lon_grid)
    no_grid_lat = len(lat_grid)

    count_TC_density = np.zeros((no_grid_lon, no_grid_lat))
    count_TC_density_cat = [np.zeros((no_grid_lon, no_grid_lat))
                            for _ in range(len(TC_rank))]
    count_gen_density = np.zeros((no_grid_lon, no_grid_lat))

    # Read
    TC_SEA = pd.read_csv(loc_file)
    TC_SEA_full = read_hdf5_to_dict(full_file)


# if os.path.exists("Track_TC_SEA_cat.png"):
i = 0
num = 0

fig_size = [12, 10]

fig, ax = ini_plot(dlat, ulat, dlon, ulon, pad=5,
                   figsize=fig_size, label_size=10, title="Track of TC in the South East Asia region", grid=True)
if True:
    for row_idx, loc_info in enumerate(TC_SEA.itertuples()):
        # * Plot the track of TC
        lon_lat_loc = TC_SEA_full[row_idx]["lon_lat"]
        cat_max_loc = int(loc_info.cat_max)
        if cat_max_loc >= 0:
            color_code = cat_colors[cat_max_loc]
        else:
            continue
            color_code = cat_colors[cat_max_loc+1]

        ax.plot(lon_lat_loc[:, 0], lon_lat_loc[:, 1],
                linewidth=2, color=color_code, alpha=0.4, zorder=1)

    # region: EXAMPLE
    # cyclone_centers = np.array([
    #     [149.99474, 13.60078],
    #     [149.80312, 13.9000025],
    #     [149.63676, 14.199213],
    #     [149.5, 14.5],
    #     [149.38673, 14.799819],
    #     [149.30307, 15.112478],
    #     [149.2447, 15.457013],
    #     [149.2, 15.8],
    #     [149.15001, 16.108364],
    #     [149.10005, 16.403133],
    #     [149.05008, 16.700087],
    #     [149., 17.]
    # ])

    # lon = cyclone_centers[:, 0]
    # lat = cyclone_centers[:, 1]
    # ax.plot(lon, lat, label=storm,
    #         color=cat_colors[0])
    # endregion

    # * Set up the plot
    make_cbar(ax, cmap, norm = norm, ticks = range(len(TC_rank)-1), labels = TC_rank[1:598
                                                                                     ])
    # * Track density
    # Save the figure
    # Adjust layout to fit the figure tightly
    fig.tight_layout()
    fig.savefig("Track_TC_SEA_cat.png", bbox_inches='tight')
    plt.clf()

begin_year = 1945
end_year = 2024
year_list = []
year_range = range(begin_year, end_year+1)
count_TC_year = np.zeros(len(year_range))
count_gen_year = np.zeros(len(year_range))
count_TC_year_cat = [np.zeros(len(year_range)) for _ in range(len(TC_rank))]


# if not os.path.exists("TC_SEA_density.png or TC_SEA_year.png"):
for row_idx, loc_info_full in enumerate(TC_SEA_full):
    idx_grid_loc = idx_grid_full(loc_info_full["lon_lat"], lat_grid, lon_grid)
    loc_info = TC_SEA.loc[row_idx]

    # Count density array
    for loc_idx in idx_grid_loc[0]:
        for lat_idx in idx_grid_loc[1]:
            count_TC_density[loc_idx, lat_idx] += 1
            count_TC_density_cat[int(loc_info.cat_max)][loc_idx, lat_idx] += 1

    idx_grid_loc = idx_grid_point(
        loc_info.lon_gen, loc_info.lat_gen, lon_grid, lat_grid)
    count_gen_density[idx_grid_loc[0], idx_grid_loc[1]] += 1

    year_loc = int(loc_info.year)
    cat_max_loc = int(loc_info.cat_max)

    # Time-series array
    if year_loc > begin_year and year_loc < end_year:
        count_TC_year[year_loc - begin_year] += 1
        count_TC_year_cat[cat_max_loc][year_loc - begin_year] += 1
        count_gen_year[year_loc - begin_year] += 1
    else:
        continue

# year_min = min(year_list)
# year_max = max(year_list)
# print(year_min, year_max)

# if begin_year < year_min or year_max < end_year:
#     cut_begin_idx = year_min - begin_year
#     cut_end_idx = year_max - end_year
#     count_TC_year = count_TC_year[(cut_begin_idx): (cut_end_idx)]
#     count_TC_year_cat = [count_TC_year_cat[i][(cut_begin_idx): (cut_end_idx)]
#                          for i in range(len(TC_rank))]
#     count_gen_year = count_gen_year[(cut_begin_idx): (cut_end_idx)]
#     year_range = range(max(year_min, begin_year), min(year_max, end_year)+1)

annual_TC_density = count_TC_density.T / len(year_range)

tick_label = np.arange(annual_TC_density.min(), annual_TC_density.max()+20, 20)
formatted_tick_label = [round(tick, 1) for tick in tick_label]
# print(tick_label)

fig, ax = ini_plot(dlat, ulat, dlon, ulon, pad=5,
                   figsize=fig_size, label_size=10, title="Density of TC in the South East Asia region")
c = ax.pcolormesh(lon_grid, lat_grid, annual_TC_density,
                  cmap='winter', zorder=1, vmin=0, vmax=300)
# cbar = plt.colorbar(c, ax=ax)

# Set the tick labels
make_cbar(ax, cmap="winter", sm=c, custom_cbar=False)
fig.tight_layout()
fig.savefig("TC_SEA_density_all.png")
plt.clf()

for i, rank in enumerate(TC_rank):
    fig, ax = ini_plot(dlat, ulat, dlon, ulon, pad=5,
                       figsize=fig_size, label_size=10, params = params)
    annual_TC_density = count_TC_density_cat[i].T / len(year_range)
    tick_label = np.arange(annual_TC_density.min(),
                           annual_TC_density.max()+20, 20)
    formatted_tick_label = [round(tick, 1) for tick in tick_label]
    # print(tick_label)

    fig_1, ax_1 = ini_plot(dlat, ulat, dlon, ulon, pad=5,
                       figsize=[12, 7], label_size=10, title=f"Density of TC - level:{TC_rank[i]} in the South East Asia region", params = params)
    c = ax_1.pcolormesh(lon_grid, lat_grid, annual_TC_density,
                      cmap='winter', zorder=1)
    # cbar = plt.colorbar(c, ax=ax)

    # Set the tick labels
    make_cbar(ax_1, cmap="winter", sm=c, custom_cbar=False)
    fig_1.tight_layout()
    fig_1.savefig(f"TC_SEA_density_TC_rank{i}.png")
    plt.close(fig)
    

# Calculate the total count for each year
total_counts = np.sum(count_TC_year_cat[1:], axis=0)

# Normalize each category's count by the total count for that year
normalized_counts = [100 * count /
                     total_counts for count in count_TC_year_cat[1:]]

# Initialize the bottom array with zeros
bottom = np.zeros(len(year_range))

# Plotting
fig = plt.figure(figsize=[20, 7])
gs = fig.add_gridspec(2, 1, hspace = 0 )
ax = fig.add_subplot(gs[1])
ax2 = fig.add_subplot(gs[0], sharex=ax)

# ? Another way to create a secondary y-axis
# # Create a secondary y-axis
# ax2 = ax.twinx()

# Plot each category as a stacked bar with specified colors
for i, rank_label in enumerate(TC_rank[1:]):  # Skip the first category
    ax.bar(year_range, normalized_counts[i],
           label=rank_label, bottom=bottom, color=cat_colors[i], alpha = 0.8)
    # Update the bottom array
    bottom += normalized_counts[i]

# Plot the total counts as a time series on the secondary y-axis
# ax2.plot(year_range, total_counts, label='Total TC', color='black', marker='o')
ax2.bar(year_range, total_counts, label='Total TC', color='grey', alpha = 0.8)

# Remove top ticks
ax.tick_params(top=False)
ax2.tick_params(top=False, labelbottom = False)

# Get handles and labels from both axes
handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

# Combine handles and labels
handles = handles1 + handles2
labels = labels1 + labels2

# Create a single legend outside the figure
fig.legend(handles, labels, loc='lower center',
           bbox_to_anchor=(0.5, -0.05), ncol=len(handles),
           handlelength = 1, handletextpad=0.5,
        #    borderaxespad =0.05,
           frameon=False)

# Set labels and title
ax.set_xlabel("Year")
ax.set_xlim(begin_year, end_year)
ax.set_ylabel("Percentage(%)")
# ax.set_title("Categorized TC in the South East Asia region", pad=20)
ax2.set_ylabel("Total TC")

fig.tight_layout()
plt.margins(x=0) # Reduce the white space btw the 1st bar and the y-axis
fig.savefig("TC_SEA_year.png", bbox_inches='tight')

raise KeyboardInterrupt
# print(test.lat.values)
# print(test.lon.values)
