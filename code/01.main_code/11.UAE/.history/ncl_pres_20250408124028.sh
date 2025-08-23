#!/bin/bash
#SBATCH --job-name=ncl_pres
#SBATCH --output=ncl_pres_%A_%a.out
#SBATCH --error=ncl_pres_%A_%a.err
#SBATCH --array=1881-1972
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --ntasks=1

# Load required modules (adjust as needed for your system)
module load ncl_ncarg/6.4.0_gnu_64

# Get the year from the array task ID
YEAR=${SLURM_ARRAY_TASK_ID}
VAR_NAME=pres.sfc
ncl YEAR=$YEAR VAR_NAME=$VAR_NAME pres_dif.ncl
