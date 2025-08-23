#!/bin/bash -l
#SBATCH --time=10-00:00:00
#SBATCH --output=real.out
#SBATCH --error=real.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --ntasks-per-node=1
#SBATCH --partition=scalable
#SBATCH --exclusive


source /home/tamnnm/load-baseEnv.sh
<<<<<<< HEAD
=======
module unload ncview/2.1.7_gnu_64
>>>>>>> c80f4457 (First commit)

ulimit -s unlimited
        
# Extract the list of nodes allocated by SLURM
nodes=$(scontrol show hostnames $SLURM_NODELIST | tr '\n' ',' | sed 's/,$//')

echo "Nodes allocated: $nodes"
# Use mpirun with the -hosts option to run the executable with the specified nodes
mpirun -hosts $nodes ./real.exe
