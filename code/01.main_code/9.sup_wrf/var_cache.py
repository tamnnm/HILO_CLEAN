from wrf import (getvar, extract_vars, ALL_TIMES, omp_enabled,
                 omp_get_num_procs, omp_set_num_threads,
                 omp_set_schedule, OMP_SCHED_STATIC,
                 OMP_SCHED_DYNAMIC,
                 OMP_SCHED_GUIDED, OMP_SCHED_AUTO)
from time import time
from netCDF4 import Dataset
import sys
from wrf import getvar, ALL_TIMES
import os
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import xarray as xr

#!!! RUN IN COMPUTE NODE ONLY

if not omp_enabled():
    raise RuntimeError("OpenMP is not available in this build")


# ? Option 1: Merge from seperate configuration (merge time)
# if len(sys.argv) > 1:
#     no_folder = int(sys.argv[0])
#     nest = int(sys.argv[1])

# if no_folder <= 9:
#     no_folder = f'0{no_folder}'

# WRF_DIRECTORY = f"/prod/projects/gemmes/tamnnm/wrf/wrf_188110/NOAA_188110_{no_folder}/"
# WRF_FILES = glob.glob(os.path.join(WRF_DIRECTORY, f"wrfout_d0{nest}*"))
# _WRF_FILES = [os.path.abspath(
#     os.path.join(WRF_DIRECTORY, f)) for f in WRF_FILES]

# # Check that the WRF files exist
# try:
#     for f in _WRF_FILES:
#         if not os.path.exists(f):
#             raise ValueError("{} does not exist. "
#                              "Check for typos or incorrect directory.".format(f))
# except ValueError as e:
#     # Try downloading then check again
#     os.system("git submodule init")
#     os.system("git submodule update")
#     os.system("GIT_DIR={}/.git git checkout -- .".format(WRF_DIRECTORY))
#     for f in _WRF_FILES:
#         if not os.path.exists(f):
#             raise e


# Create functions so that the WRF files only need
# to be specified using the WRF_FILES global above
# def single_wrf_file():
#     global _WRF_FILES
#     return _WRF_FILES[0]


# def multiple_wrf_files():
#     global _WRF_FILES
#     return _WRF_FILES
# print("All tests passed!")


# file_paths = multiple_wrf_files()

# start = time()
# wrf_files = [Dataset(f) for f in file_paths]
# end = time()
# print("The time to open for multiple files (30 files) is {}s".format(end-start))


# ? Option 2: For already merged configuration (Each file ~ 15-30gb/30 days)

out_folder = "/prod/projects/gemmes/tamnnm/wrf/wrf_188110/merge_var/"
in_name = sys.argv[1]
out_name = in_name.split('/')[-1]

# ? Option 1

# ? Option 2
wrf_file = [Dataset(in_name),]

# Open all files at once
cache = extract_vars(wrf_file, ALL_TIMES,
                     ("P", "PSFC", "PB", "PH", "PHB",
                      "T", "QVAPOR", "HGT", "U", "V",
                      "W"))

# ? Full_var

# vars = ("avo", "eth", "cape_2d", "cape_3d", "ctt", "dbz", "mdbz",
#         "geopt", "helicity", "lat", "lon",  "omg", "p", "pressure",
#         "pvo", "pw", "rh2", "rh", "slp", "ter", "td2", "td", "tc",
#         "theta", "tk", "times", "tv", "twb", "updraft_helicity", "ua", "va",
#         "wa", "uvmet10", "uvmet", "wspd_wdir", "wspd_wdir10", "z", "cfrac")

vars = ("avo", "lat", "lon", "pressure",
        "pvo", "pw", "slp", "tk",
        "times", "z")


omp_set_num_threads(omp_get_num_procs())
print("Set number of threads")
chunk_size = 0

sched_string_map = {int(OMP_SCHED_AUTO): "OMP_SCHED_AUTO"}


def save_variable(var, wrf_file=wrf_file, cache=cache, out_folder=out_folder, out_name=out_name):
    if os.path.exists(f"{out_folder}{var}_{out_name}.nc"):
        return f"File {out_folder}{var}_{out_name}.nc already exists"

    v = getvar(wrf_file, var, ALL_TIMES, cache=cache)
    var_name = v.name
    print(var_name)
    encoding = {
        var_name: {
            "zlib": True,  # Enable compression
            "complevel": 5,  # Compression level (1-9)
            "shuffle": True,  # Enable shuffle filter
        }
    }

    # Drop the 'projection' attribute if it exists
    if 'projection' in v.attrs:
        del v.attrs['projection']

    try:
        v.to_netcdf(f"{out_folder}{var}_{out_name}.nc", encoding=encoding)
    except Exception as e:
        print(v)
        raise e
    return f"Creating netCDF {var_name} for {out_name}"


for sched in (OMP_SCHED_AUTO,):
    # (OMP_SCHED_STATIC, OMP_SCHED_DYNAMIC,
    #  OMP_SCHED_GUIDED, OMP_SCHED_AUTO):
    omp_set_schedule(sched, chunk_size)
    sched_string = sched_string_map[int(sched)]
    print("Running with sheduler: {}".format(sched_string))

    start = time()
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(save_variable, var) for var in vars]
        for future in as_completed(futures):
            print(future.result())
    end = time()

    print("Time taken using scheduler {}: {} s".format(sched_string, end-start))
