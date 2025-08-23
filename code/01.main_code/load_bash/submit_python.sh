#!/bin/bash
#SBATCH --job-name=JOBNAME
#SBATCH --output=JOBNAME.out
#SBATCH --error=JOBNAME.err
#SBATCH --partition=PARTITION
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --time=10:00:00

python "$@"

# --------------------------- submit job --------------------------- #

