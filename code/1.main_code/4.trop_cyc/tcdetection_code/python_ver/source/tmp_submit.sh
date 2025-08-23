#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=test_output.txt
#SBATCH --partition=broadwell
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=20
#SBATCH --time=10:00:00
python "$@"

# --------------------------- submit job --------------------------- #

