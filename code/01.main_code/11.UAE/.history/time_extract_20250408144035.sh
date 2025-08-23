#!/bin/bash

folder=$1  # Remove space around = in assignment

cd "$folder" || exit 1  # Add error handling for cd

last_out=$(ls . | grep "^wrfout_d01_.*00:00:00$" | sort | tail -n 1)
first_out=$(ls . | grep "^wrfout_d01_.*00:00:00$" | sort | head -n 1)

last_datetime=$(echo "$last_out" | awk -F'_' '{print $3}')
first_datetime=$(echo "$first_out" | awk -F'_' '{print $3}')

num_files=$(ls . | grep "^wrfout_d01_.*00:00:00$" | wc -l)  # Remove space around =


# Calculate time difference in seconds
time_diff=$((last_seconds - first_seconds))

if ((num_files < 2)); then
    echo "Error: Less than two output files found."
    exit 1
fi

# Calculate average time in seconds
avg_time=$((time_diff / (num_files - 1)))

# Convert average time to minutes
avg_time_minutes=$(echo "scale=2; $avg_time / 60" | bc)

cd ..

# Remove "easier parsing" text and just output the result
echo "$avg_time_minutes"


