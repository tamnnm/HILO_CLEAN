/bin/bash

folder = $1
last_ind_out=$(ls  | grep "^wrfout_d01_.*00:00:00$" | sort | tail -n 1)
  