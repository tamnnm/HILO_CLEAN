#!/bin/bash

source /home/tamnnm/load_func_vars.sh

for file in $prod_wrf/wrf_188110/merge_output/wrf_*; do
 runable var_cache.py $file
 echo "Run background for $file"
done
