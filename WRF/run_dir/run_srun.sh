#!/bin/bash -l
#SBATCH --time=10-00:00:00
<<<<<<< HEAD
#SBATCH --ntasks=2
#SBATCH --error=/work/users/tamnnm/WRF/run_dir/report/NOAA_188110/NOAA_188110_07.err
#SBATCH --output=/work/users/tamnnm/WRF/run_dir/report/NOAA_188110/NOAA_188110_07.out
#SBATCH --cpus-per-task=56
#SBATCH --nodelist=compute-1-[13,14]
#SBATCH --partition=scalable
#SBATCH --exclusive

=======
#SBATCH --ntasks=3
#SBATCH --error=/work/users/tamnnm/WRF/run_dir/report/ERA5_194412/ERA5_194412_34.err
#SBATCH --output=/work/users/tamnnm/WRF/run_dir/report/ERA5_194412/ERA5_194412_34.out
#SBATCH --cpus-per-task=40
#SBATCH --nodelist=compute-1-[]
#SBATCH --partition=scalable
#SBATCH --exclusive

python file
>>>>>>> c80f4457 (First commit)
# !!!! REMEMBER: number_nodes <= the number of ntasks (It will automatically change)

# Case 1:
# If you do this: nodelist = {compute-1-[1-5] (5 nodes)} + {nodes = 2 (choose 2 nodes)} + {ntasks = 4} + {cpus-per-task=10}
#!!sbatch: error: invalid number of nodes (-N 5-2)

# Case 2
# If you do this: nodelist = {compute-1-[1,2] (2 nodes)} + {nodes = 2 (choose 2 nodes)} + {ntasks = 4} + {cpus-per-task=10}
#!! sbatch: error: Batch job submission failed: Requested node configuration is not available

source /home/tamnnm/load-baseEnv.sh
<<<<<<< HEAD
=======
module unload ncview/2.1.7_gnu_64
>>>>>>> c80f4457 (First commit)

ulimit -s unlimited
# Extract the list of nodes allocated by SLURM
nodes=$(scontrol show hostnames $SLURM_NODELIST | tr '\n' ',' | sed 's/,$//')

echo "Nodes allocated: $nodes"
<<<<<<< HEAD
=======
echo "Number of total CPUs: $((SLURM_NTASKS * SLURM_CPUS_PER_TASK))"

>>>>>>> c80f4457 (First commit)
# Use mpirun with the -hosts option to run the executable with the specified nodes
mpirun -hosts $nodes ./wrf.exe
