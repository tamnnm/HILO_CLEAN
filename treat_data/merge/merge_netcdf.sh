#!/bin/bash
for ((j=1900; j<2011; j++)); do
 #ncks --mk_rec_dmn time  "e20c.oper.an.sfc.6hr.128_167_2t.regn80sc.${j}010100_${j}123118.grb.nguyen658089.nc" "packed_era_167_${j}.nc"
 ncpdq -U "packed_e20c.oper.an.sfc.6hr.128_167_2t.regn80sc.${j}010100_${j}123118.grb.nguyen658089.nc" "unpack_era_167_${j}.nc"
done
wait
mergetime unpack_era_167_*.nc era_167.nc


