#!/bin/bash

for i in {1881..1972}; do
    YEAR=$i
    VAR_NAME=pres.sfc
    ncl YEAR=$YEAR VAR_NAME=$VAR_NAME ncl_pres.ncl
done
