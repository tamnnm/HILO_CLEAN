#!/bin/bash
### Job name
#PBS -N DETECTING
#PBS -q serial_q 
#PBS -l nodes=1:ppn=1
#PBS -o o.out
#PBS -e e.err
#PBS -V

module load netcdf/4.1.3_intel_64 
cd /work/users/longtt/DATA_DETECT/detecting
#
./tcdetect
mv TCs.txt exp.0604
exit
#
