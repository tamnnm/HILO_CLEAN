#!/bin/bash

# Cause error when 
# module purge

## Core environment and libs
module load intel/2019.u5
module load slurm/21.08.5
module load netcdf/4.6.1_intel_64
#module load hdf5/1.8.15p1_intel_64
module load mvapich2/2.3.6_intel

## Some Intel apps(to run WRF)
module load nco/4.6.1_intel_64
module load ImageMagick/6.9.2-3_intel_64
module load PnetCDF/1.9.0_intel_64
module load jasper/1.900.1_gnu_64
module load geos/3.5.0_gnu_64
module load ncl_ncarg/6.4.0_gnu_64
module load proj/4.9.2_intel_64
module load libpng/1.6.2_gnu_64
module load zlib/1.2.8_gnu_64
module load cdo/1.9.3_gnu_64
module load anaconda3/2021.11

# If you running WRF, do not load ncview
module load ncview/2.1.7_gnu_64
## You don't use Grads so skip this
# module load grads/2.2.0_gnu_64


## Export self base apps
export MKL_DISABLE_FAST_MM=1
export PATH="/work/apps/gnu_4.8.5:$PATH"
export GFORTRAN_PATH="/usr/bin/gfortran:$PATH"
export LD_LIBRARY_PATH="/home/tamnnm/.conda/envs/tamnnm/lib/python3.9/site-packages:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/work/users/thanhnx/nx-lib/apps_base/lib:$LD_LIBRARY_PATH"i
export PATH="/home/tamnnm/.vscode-server/cli/servers/Stable-e54c774e0add60467559eb0d1e229c6452cf8447/server/bin/remote-cli:$PATH"
export DISPLAY=localhost:10.0

# export WRADLIB_DATA="/work/users/tamnnm/geo_info/vnm/wradlib-data/data"
# export GASCRP=/work/users/thanhnx/nx-lib/apps_base/lib/grads
