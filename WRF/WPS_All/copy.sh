#!/bin/bash

source /home/tamnnm/load_env_vars.sh
for folder in $work/wrf/WPS/WPS_*; do
  base_folder=$(basename $folder)
  echo $base_folder
  mkdir $base_folder 
  mv $folder/namelist.wps $folder/met_em* $folder/FILE* $folder/geo_em* $base_folder
done

