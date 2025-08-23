#!/bin/bash
#SBATCH --job-name=var_cache
#SBATCH --output=var_cache.out
#SBATCH --error=var_cache.err
#SBATCH --partition=scalable
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --time=10:00:00

python "$@"

# --------------------------- submit job --------------------------- #

