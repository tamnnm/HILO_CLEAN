/bin/bash

folder = $1

cd $folder

last_out=$(ls . | grep "^wrfout_d01_.*00:00:00$" | sort | tail -n 1)
first_out=$(ls . | grep "^wrfout_d01_.*00:00:00$" | sort | head -n 1)

last_datetime=$(echo "$last_ind_out" | awk -F'_' '{print $3}')
first_datetime=$(echo "$first_ind_rst" | awk -F'_' '{print $3}')

num_out =

