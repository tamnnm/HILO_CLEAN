#!/bin/bash
#SBATCH --job-name=ncl_pres
#SBATCH --output=ncl_pres_%A_%a.out
#SBATCH --error=ncl_pres_%A_%a.err
#SBATCH --array=0-91    # Changed to 0-91 (92 years from 1881 to 1972)
#SBATCH --time=02:00:00
#SBATCH --nodelist=compute-1-[18
#SBATCH --partition=scalable
#SBATCH --ntasks=1

# Load required modules
module load ncl_ncarg/6.4.0_gnu_64

# Calculate the actual year (1881 + array_index)
YEAR=$((1881 + SLURM_ARRAY_TASK_ID))
VAR_NAME=pres.sfc

echo "Processing year: $YEAR"
ncl YEAR=$YEAR VAR_NAME=$VAR_NAME pres_dif.ncl
