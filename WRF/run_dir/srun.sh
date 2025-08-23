#!/bin/bash
#SBATCH -J submit_srun
#SBATCH --time=10-00:00:00
#SBATCH -o submit_srun.out
#SBATCH -e submit_srun.err
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=56
#SBATCH --nodelist=compute-1-[7,8]
#SBATCH --partition=scalable

# !!!! REMEMBER: number_nodes <= the number of ntasks (It will automatically change)

# Case 1:
# If you do this: nodelist = {compute-1-[1-5] (5 nodes)} + {nodes = 2 (choose 2 nodes)} + {ntasks = 4} + {cpus-per-task=10}
#!!sbatch: error: invalid number of nodes (-N 5-2)

# Case 2
# If you do this: nodelist = {compute-1-[1,2] (2 nodes)} + {nodes = 2 (choose 2 nodes)} + {ntasks = 4} + {cpus-per-task=10}
#!! sbatch: error: Batch job submission failed: Requested node configuration is not available

ulimit -s unlimited
mpirun ./wrf.exe
