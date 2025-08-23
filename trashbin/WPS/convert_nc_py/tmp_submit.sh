#!/bin/bash
#SBATCH --job-name=convert_nc_im
#SBATCH --output=convert_nc_im.out
#SBATCH --error=convert_nc_im.err
#SBATCH --partition=broadwell
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=20
#SBATCH --time=10:00:00
python "$@"

# --------------------------- submit job --------------------------- #

