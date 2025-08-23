/bin/bash

folder = $1

cd $folder

last_out=$(ls . | grep "^wrfout_d01_.*00:00:00$" | sort | tail -n 1)
first_out=$(ls . | grep "^wrfout_d01_.*00:00:00$" | sort | head -n 1)

last_datetime=$(echo "$last_ind_out" | awk -F'_' '{print $3}')
first_datetime=$(echo "$first_ind_rst" | awk -F'_' '{print $3}')

num_files = $(ls . | grep "^wrfout_d01_.*00:00:00$" | wc -l)

# Convert dates to seconds since epoch
last_seconds=$(date -d "${last_datetime:0:4}-${last_datetime:4:2}-${last_datetime:6:2} ${last_datetime:8:2}:${last_datetime:10:2}:${last_datetime:12:2}" +%s)
first_seconds=$(date -d "${first_datetime:0:4}-${first_datetime:4:2}-${first_datetime:6:2} ${first_datetime:8:2}:${first_datetime:10:2}:${first_datetime:12:2}" +%s)

# Calculate time difference in seconds
time_diff=$((last_seconds - first_seconds))

# Calculate average time in seconds
avg_time=$((time_diff / (num_out - 1)))

# Convert average time to minutes
avg_time_minutes=$(echo "scale=2; $avg_time / 60" | bc)

easier parsing
echo "