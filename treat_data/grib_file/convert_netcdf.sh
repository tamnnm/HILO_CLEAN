#!/bin/bash

input_folder="$data/wrf_data/gribfile"
output_folder="$data/wrf_data/netcdf"
converted_list="$data/wrf_data/gribfile/list_grib.txt"

echo "$input_folder"

for folder_name in $input_folder/*_synop_*; do
 folder="$input_folder/$folder"
 cd $folder
 for gribfile in $folder/*.grb; do
	filename=$(basename "$gribfile")
	echo "$filename"
	netfile=$(basename "$gribfile" .grb).nc
	if grep -q "$filename" "$converted_list"; then
		echo "Skipping $filename"
	else
		cp "$gribfile" temp.grb
		cdo -f grb copy temp.grb "$output_folder/$folder/$netfile"
	rm temp.nc
	echo "Converted $gribfile to $netfile"
	echo "$filename" >> "$converted_list"
	fi
 done
done

echo "Conversion complete"
