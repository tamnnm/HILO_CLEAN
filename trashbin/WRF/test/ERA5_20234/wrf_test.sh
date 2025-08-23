#!/bin/bash
#SBATCH -J wrf_tam
#SBATCH --time=5-00:00:00
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --output=wrf.out
#SBATCH --partition=scalable
#SBATCH --exclude=compute-1-[12-13]
#SBATCH --exclusive
date
ulimit -s unlimited
srun -N 5 ./wrf.exe
wait
date

