#!/bin/bash
#SBATCH -J wrf_tamnnm_root
#SBATCH --time=10:00:00
#SBATCH --nodes=2
#SBATCH --partition=scalable
#SBATCH --exclusive

folder="WPS_ERA5_194512"
ulimit -s unlimited
echo "Start $folder"
cd $folder
rm nohup.out
rm geo_em*
wait
if ./geogrid.exe; then
 echo "geogrid.exe success"
else
 echo "geogrid.exe failed"
fi
 wait
if ./ungrib.exe; then
 echo "ungrib.exe success"
else
 echo "ungrib.exe failed"
fi
wait

if ./metgrid.exe; then
 echo "metgrid.exe success"
else
 echo "metgrid.exe failed"
fi
cd ..

