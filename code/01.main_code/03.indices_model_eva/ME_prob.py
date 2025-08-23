# Calculate the STA using unbalance Silkthorne and soft-DTW

# Pseudo code:
# 1. Load the data from the cache.
# - Observation data: [timesteps, nb of points]
# - Reanalysis data: [timesteps, nb of points]
#     . Mask the data => Extract all the possible points inside Vietnam
# 2. Calculate the STA using unbalance Silkthorne and soft-DTW.
# - Calculate K:
#     . Calculate distance matrix [nb of obs points, nb of reanalysis points]
#     . Calculate the kernel matrix K
# - Calculate the STA using sta_distance
# 3. Save the results to the cache.
# 4. Plot the results.

from sta import sta_distances
import xarray as xr
from sklearn.metrics.pairwise import haversine_distances
import json
import numpy as np
from joblib import Parallel, delayed
from my_junk import *
from ME_extreme_indices import MetricExtreme
from constant import *
import time
Cache = DataCache()

os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open(json_path, 'r') as json_file:
    data_dict = json.load(json_file)

rean_dataset = "era"

#* Flag to check processing status
#? True = Run, False = Load from cache
# Flag to check if the distance matrix and reanalysis 2D values are already calculated
matrix_flag = check_non_exist(f"{rean_dataset}_distance_matrix.pkl")
# Flag to check if the kernel matrix is already calculated
kernel_flag = check_non_exist(f"{rean_dataset}_kernel_matrix.pkl") or matrix_flag
# Flag to check if the reanalysis 2D values are already calculated
rean_2d_flag = check_non_exist(f"{rean_dataset}_reshaped.pkl") or matrix_flag
# Create mask for reanalysis data
mask_flag = check_non_exist(f"{rean_dataset}_mask.nc") or matrix_flag
# Calculate K for each timestep if True, otherwise use the same K for all timesteps
missing_flag = True

#* Load the data
obs_ds = Cache.get_data(os.path.join(Data_nc, "para_167", "Tx_obs.nc"))
rean_ds = Cache.get_data(os.path.join(Data_nc, "para_167", f"Tx_{rean_dataset}.nc"))

#* Create the mask
print("Creating mask for reannalysis data...")
# Mask the data
if mask_flag:
    mask_df = mask_create(lon_array=rean_ds.lon.values, lat_array=rean_ds.lat.values, output_df = True, save_path="{rean_dataset}_mask.nc")
else:
    mask_df = xr.open_dataset(f"{rean_dataset}_mask.nc")
mask_array = mask_df.mask.values

#* Extract the year range and data array

print("Extracting year range and data array...")
instance = object.__new__(MetricExtreme)
instance.dataset=rean_dataset
instance.season = "none"
instance._setup_year()
obs_var_ds,_ = instance._subset_year(obs_ds, "Tx") # TODO: Change if you want to use only > 95% data

if rean_2d_flag:
    if check_non_exist(f"Tx_{rean_dataset}_subset_year.nc"):
        rean_var_ds,_ = instance._subset_year(rean_ds, "T2m", mode='rean')
        Cache.save_data(rean_var_ds, f"Tx_{rean_dataset}_subset_year.nc")
    else:
        rean_var_ds = Cache.get_data(f"Tx_{rean_dataset}_subset_year.nc")
        rean_var_ds = cut_var(rean_var_ds, "T2m")

# Save memory
del rean_ds

# #* Filter stations and grid points inside Vietnam
# print("Filtering observation and reanalysis datasets...")
# obs_filter_ds = obs_ds.sel(no_station = data_dict["no_T2m_city"], drop = True)

#* Flatten value matrix
print("Treating 2d input arrays...")
if rean_2d_flag:
    # Reshape the 3D array to a 2D array where rows are time steps and columns are all spatial points
    rean_values_3d = rean_var_ds.values
    n_times = rean_values_3d.shape[0]
    n_spatial_points = rean_values_3d.shape[1] * rean_values_3d.shape[2]
    rean_values_2d = rean_values_3d.reshape(n_times, n_spatial_points)
    # Now select only the valid columns from this 2D array
    rean_data_2d = rean_values_2d[:, mask_array.flatten()]
    Cache.save_data(rean_data_2d, f"{rean_dataset}_reshaped.pkl")
else:
    # Load the reshaped reanalysis values from cache
    rean_data_2d = Cache.get_data(f"{rean_dataset}_reshaped.pkl", type_cache='pickle')

obs_data_2d = obs_var_ds.values

#* Create the lon, lat arrays
if rean_2d_flag:
    rean_lons, rean_lats = np.meshgrid(mask_df.lon.values, mask_df.lat.values) # Shape: (lat, lon)
    #? Reanalysis: Following the grid
    # Extracting the non-Nan values
    valid_lons = rean_lons[mask_array]
    valid_lats = rean_lats[mask_array]

    # Extract coordinate pairs for valid points
    rean_points = np.column_stack((valid_lons,valid_lats))
    print(rean_points)

#* Extract the observation data
if matrix_flag:
    if missing_flag:
        #* Calculate the kernel matrix
        obs_lons, obs_lats = obs_ds.lon.values, obs_ds.lat.values
        obs_points = []
        M_matrix = []

        #* Function to calculate the distance matrix
        def calculate_distance_matrix(i, obs_data_2d, obs_lons, obs_lats,obs_points, rean_points):
            """
            Calculate the distance matrix using haversine distance.
            """
            valid_stations = np.nonzero(~np.isnan(obs_data_2d[i, ...]))
            valid_lons = obs_lons[valid_stations]
            valid_lats = obs_lats[valid_stations]
            obs_points = np.column_stack((valid_lons, valid_lats))
            return haversine_distances(obs_points, rean_points)

        #* Extract location
        #? Observation: Following each station
        print("Calculating distance matrix...")

        # for i in range(obs_var_ds.time.size):
        #     print(f"Calculating distance matrix for {i} timestep...")
        #     valid_stations = np.nonzero(~np.isnan(obs_data_2d[i, ...]))
        #     valid_lons = obs_lons[valid_stations]
        #     valid_lats = obs_lats[valid_stations]

        #     obs_points = np.column_stack((valid_lons, valid_lats))
        #     #* Kernel matrix
        #     # Calculate the distance matrix using haversine distance
        #     M_matrix.append(haversine_distances(obs_points, rean_points))

        start_time = time.time()
        M_matrix = Parallel(n_jobs=-1) \
                    (delayed(calculate_distance_matrix)(i, obs_data_2d, obs_lons, obs_lats, obs_points, rean_points) \
                     for i in range(obs_var_ds.time.size))
        end_time = time.time()
        print("Distance matrix calculated in {:.2f} seconds.".format(end_time - start_time))
    else:
        obs_filter_ds = obs_var_ds.sel(no_station = data_dict["no_T2m_city"], drop = True)
        #* Calculate the kernel matrix
        obs_lons, obs_lats = obs_filter_ds.lon.values, obs_filter_ds.lat.values

        obs_points = np.column_stack((obs_lons, obs_lats))
        print("Calculating distance matrix...")
        #* Kernel matrix
        # Calculate the distance matrix using haversine distance
        M_matrix = haversine_distances(obs_points, rean_points)
        print("Distance matrix calculated.")

    Cache.save_data(M_matrix, f"{rean_dataset}_distance_matrix.pkl")

else:
    print("Loading distance matrix from cache...")
    # Load the distance matrix from cache
    M_matrix = Cache.get_data(f"{rean_dataset}_distance_matrix.pkl", type_cache='pickle')


#* Calculate the distance matrix
print("Calculating distance matrix...")
if kernel_flag:
    if isinstance(M_matrix, np.ndarray) and M_matrix.ndim == 2:
        # If M_matrix is 2D, it means we have a single distance matrix
        print("M_matrix is 2D, shape:", M_matrix.shape)
        # Convert to meters
        M_matrix = M_matrix * 6371000  # Convert to meters
        # Normalize the distance matrix by median
        #? Using median is better than mean because it is less sensitive to outliers
        #? This is a common practice in distance-based methods to ensure that the distances are on a similar scale.
        #? It helps to avoid the influence of extreme values on the distance calculations
        #? Normalizing the distance matrix can help improve the performance of the STA algorithm.
        M = M_matrix / np.median(M_matrix)

        # Calculate the epsilon value
        n_features = np.min(M.shape)
        epsilon = 10. / n_features
        # To test only
        # betas = [0, 0.001, 0.01, 0.1, 0.5, 1., 2., 3., 5., 10.]
        K = np.exp(- M / epsilon)

    else:
        # Process each timestep individually
        def compute_kernel(t):
            M_t = M_matrix[t]
            n_features = min(M_t.shape)
            epsilon = 10.0 / n_features
            M_median = np.median(M_t)
            M_normalized = M_t / M_median
            return np.exp(-0.1 * M_normalized / epsilon)

        start_time = time.time()
        print("M_matrix is a list of distance matrices, calculating kernel matrix for each timestep...")
        K_list = []
        for t in range(len(M_matrix)):
            k = compute_kernel(t)
            K_list.append(k)
        # K_list = Parallel(n_jobs=-1)(delayed(compute_kernel)(t) for t in range(len(M_matrix)))
        end_time = time.time()
        print("Kernel matrix calculated in {:.2f} seconds.".format(end_time - start_time))

        K = np.empty(len(K_list), dtype=object)
        K[:] = K_list
        Cache.save_data(K, f"{rean_dataset}_kernel_matrix.pkl")
else:
    print("Loading kernel matrix from cache...")
    # Load the kernel matrix from cache
    K = Cache.get_data(f"{rean_dataset}_kernel_matrix.pkl", type_cache='pickle')


obs_data_2d_cleaned = [row[~np.isnan(row)] for row in obs_data_2d]
obs_data_2d = np.empty(len(obs_data_2d_cleaned), dtype=object)
obs_data_2d[:] = obs_data_2d_cleaned

# Testchunk
obs_data_2d = obs_data_2d[:5]
print("obs_data_2d shape:", obs_data_2d.shape)
rean_data_2d = rean_data_2d[:5, :]

print(obs_data_2d.shape, rean_data_2d.shape)
# * Set the beta value
beta = 0.1

# sys.exit()

#* Calculate STA distances
# Calculate the STA using unbalance Silkthorne and soft-DTW
print("Calculating STA distances...")
sta_data = sta_distances(obs_data_2d, rean_data_2d, K, beta, amari = None)

testprint(sta_data)
# beta
