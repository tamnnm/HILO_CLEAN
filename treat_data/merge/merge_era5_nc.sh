#!/bin/bash
#SBATCH --time=10-00:00:00
#SBATCH --nodes=2
#SBATCH --partition=scalable
#SBATCH --exclusive

grib_path="/work/users/tamnnm/Data/wrf_data/gribfile/synop_file/synop_file_full/era5_synop_1940_2023"
for ((i=1940; i<2023;i++)); do
 if [ ! -f "era5_89_${i}.nc" ]; then
  cdo mergetime "$grib_path/total_precipitation_${i}.grib" "$grib_path/total_precipitation_${i}_odd.grib"  int.grib
  wait
  cdo -f nc copy int.grib "era5_89_${i}.nc"
  wait
  rm int.grib
 else
  echo "era5_89_${i}.nc"
 fi

 if [ ! -f "era5_167_${i}.nc" ]; then
  cdo mergetime "$grib_path/2m_temperature_${i}.grib" "$grib_path/2m_temperature_${i}_odd.grib" int.grib 
  wait
  cdo -f nc copy int.grib "era5_167_${i}.nc"
  wait
  rm int.grib
 else
  echo "era5_167_${i}.nc"
 fi

 if [ ! -f "era5_166_${i}.nc" ]; then
  cdo mergetime "$grib_path/10m_v_component_of_wind_${i}.grib" "$grib_path/10m_v_component_of_wind_${i}_odd.grib" int.grib
  wait
  cdo -f nc copy int.grib "era5_166_${i}.nc"
  wait
  rm int.grib
 else
  echo "era5_166_${i}.nc" 
 fi

 if [ ! -f "era5_165_${i}.nc" ]; then 
  cdo mergetime "$grib_path/10m_u_component_of_wind_${i}.grib" "$grib_path/10m_u_component_of_wind_${i}_odd.grib" int.grib
  wait
  cdo -f nc copy int.grib "era5_165_${i}.nc"
  wait
  rm int.grib
 else
  echo "era5_165_${i}.nc"
 fi

done
