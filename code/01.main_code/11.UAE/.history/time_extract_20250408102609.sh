/bin/bash

last_ind_out=$(ls $prod_ind_folder | grep "^wrfout_d01_.*00:00:00$" | sort | tail -n 1)
  