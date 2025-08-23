nput_folder="$data/wrf_data/netcdf/noaa_hourly_1877_1972"
output_folder="$data/wrf_data/gribfile/noaa_hourly_1977_1972"
converted_list="$data/wrf_data/gribfile/list_grib.txt"

echo "$input_folder"

# Use GNU Parallel to convert files simultaneously
find $input_folder -name "*.nc" | parallel -j 4 --no-notice
	filename=$(basename {})

# Check if the file has already been converted
	if grep -q "$filename" $converted_list; then
		echo "Skipping $filename"
	else
# Convert the file to GRIB format
		gribfile=$(basename {} .nc).grb
		cdo -f grb copy {} $output_folder/$gribfile
# Add the converted file to the list
		echo $filename >> $converted_list
		echo "Converted $filename to $gribfile"
	fi
echo "Conversion complete"
