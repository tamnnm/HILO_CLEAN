#!/bin/bash

# Script to process seasonal means from climate data files
# Usage: ./process_seasonal_means.sh

pwd
# Process all mean_*_167.nc files
for file in *_6019_228.nc; do
  # Extract just the dataset name
  dataset=$(echo $file | sed 's/\(.*\)_167\.nc/\1/')
  
  # Print information about current processing
  echo "Processing $file for dataset: $dataset"
  
  # Run CDO command to split by seasons and calculate seasonal means
  cdo splitseas $file Rm_full_${dataset}_
  
  # Run CDO command to split by seasons and calculate seasonal means
  cdo splitseas -monmean $file Rm_mon_${dataset}_
  
  # cdo.yearmean '$file Tm_${dataset}_annual.nc'
  
  # cdo timmean Rm_${dataset}_annual.nc Rm_${dataset}_annual_clim.nc
  
  # cdo yseasmean $file Rm_${dataset}_seasonal_clim.nc
  
  # Check if the command was successful
  if [ $? -eq 0 ]; then
    echo "Successfully processed $dataset"
  else
    echo "Error processing $dataset"
  fi
done

echo "All processing complete"