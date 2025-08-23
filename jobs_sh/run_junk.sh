#!/bin/bash

#BATCH --partition=scalable
#SBATCH --job-name=junk
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=240:00:00
#SBATCH --output=junk.out
#SBATCH --error=junk.err

source /home/student6/.bashrc

NCORE=$SLURM_NTASKS      ; [[ -z $NCORE ]] && NCORE=2

# #####################################################

pyDir="/work/users/student6/tam/code"
## Chay lan luot
#python "$pyDir/main.py"
python "$pyDir/junk.py"
#$pythonDir/python code1.sh 
#$pythonDir/python code1.sh 
#$pythonDir/python code1.sh 
#$pythonDir/python code1.sh 


## Chay song song nhieu code: --ntasks-per-node=[so luong code]

#
