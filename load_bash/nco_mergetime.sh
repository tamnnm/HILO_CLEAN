#!/bin/bash

if [ ! -d "merge_output" ]; then
 mkdir ./merge_output
 mkdir -p ./merge_output/nohup
fi

for folder in ./*; do
 if [ -d "$folder" ]; then
   cd "$folder" || continue
   no_folder=$(echo "$folder" | awk -F'_' '{print $NF}')
   
   if [ ! -f ../merge_output/wrf_d01_${no_folder} ]; then
     nohup ncrcat wrfout_d01* "../merge_output/wrf_d01_${no_folder}" > "../merge_output/nohup/nohup_d01_$no_folder.out" 2>&1 &
   fi
   
   if [ ! -f ../merge_output/wrf_d02_${no_folder} ]; then
     nohup ncrcat wrfout_d02* "../merge_output/wrf_d02_${no_folder}" > "../merge_output/nohup/nohup_d02_$no_folder.out" 2>&1 &
   fi   
   
   echo "Merge for $no_folder"
   cd ..
 fi
done
