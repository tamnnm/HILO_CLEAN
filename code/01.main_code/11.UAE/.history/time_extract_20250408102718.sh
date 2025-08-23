/bin/bash

folder = $1


last_ind_out=$(ls $folder | grep "^wrfout_d01_.*00:00:00$" | sort | tail -n 1)
first_ind_out=$(ls $folder | grep "^wrfout_d01_.*00:00:00$" | sort | head -n 1)
rst_datetime=$(echo "$last_ind_rst" | awk -F'_' '{print $3}')
out_datetime=$(echo "$last_ind_out" | awk -F'_' '{print $3}')
sub_rst_datetime=$(echo "$sub_ind_rst" | awk -F'_' '{print $3}')
    