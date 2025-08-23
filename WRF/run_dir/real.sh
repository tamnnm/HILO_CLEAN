#!/bin/bash
#SBATCH --time=10-00:00:00
#SBATCH --output=real.out
#SBATCH --error=real.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --partition=broadwell
#SBATCH --nodelist=compute-2-7
#SBATCH --exclusive

# Use srun to run the executable with the specified number of nodes
mpirun ./real.exe