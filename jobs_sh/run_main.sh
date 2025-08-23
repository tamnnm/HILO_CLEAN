#!/bin/bash

#BATCH --partition=scalable
#SBATCH --job-name=suicide
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=240:00:00
#SBATCH --output=suicide.out
#SBATCH --error=suicide.err

source /home/student6/.bashrc

NCORE=$SLURM_NTASKS      ; [[ -z $NCORE ]] && NCORE=2

# #####################################################
export PYTHONPATH=/work/users/student6/tam/code/pkg_twcr:${PYTHONPATH}
pyDir="/work/users/student6/tam/code/main_code"

## Chay lan luot
#python "$pyDir/main.py"
python "$pyDir/gpi_main.py"
#$pythonDir/python code1.sh 
#$pythonDir/python code1.sh 
#$pythonDir/python code1.sh 
#$pythonDir/python code1.sh 


## Chay song song nhieu code: --ntasks-per-node=[so luong code]

#ncode=` ls *codelist.py | wc -l ` 
