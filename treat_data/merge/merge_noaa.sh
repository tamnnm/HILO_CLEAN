#!/bin/bash

#SBATCH -J wrf_tamnnm_root
#SBATCH --time=10:00:00
#SBATCH --nodes=2
#SBATCH --partition=scalable
#SBATCH --exclusive

list=(apcp air.2m tmax.2m tmin.2m uwnd.10m vwnd.10m)
for i in "${list[@]}"; do
 if [ ! -f "${i}.20cr.nc" ]; then
 	cdo mergetime "${i}.*.nc" "${i}.20cr.nc"
 else
	continue
 fi 
done
