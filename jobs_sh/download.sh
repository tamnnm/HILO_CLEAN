#!/bin/bash
#SBATCH --partition=broadwell
#SBATCH --job-name=download
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=20
#SBATCH --time=240:00:00
#SBATCH --output=download.out
#SBATCH --error=download.err

source /home/student6/.bashrc
module load scrapy
module list
NCORE=$SLURM_NTASKS      ; [[ -z $NCORE ]] && NCORE=20


# #####################################################

pyDir="/work/users/student6/tam/download_code/wrf_download/twcr"
## Chay lan luot
cat $pyDir/urls.csv | tr -d '\r' | nohup xargs -n 10 -P 20 wget -nc --no-check-certificate -P $data/wrf_data
#srun python "$pyDir/download3.py" --pty bash
#$pythonDir/python code1.sh 
#$pythonDir/python code1.sh 
#$pythonDir/python code1.sh 
#$pythonDir/python code1.sh 


## Chay song song nhieu code: --ntasks-per-node=[so luong code]

#ncode=` ls *codelist.py | wc -l ` 

