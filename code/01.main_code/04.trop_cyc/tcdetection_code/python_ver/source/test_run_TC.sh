#!/bin/bash
#SBATCH --output=test_run.txt
#SBATCH --error=/dev/null
source /home/tamnnm/load_func_vars.sh
#runbroadwell "test" main_search.py #1005.0 4.0 1.45 -1.0 
# Submit the first job and capture the job ID
job_id=$(runbroadwell "test" main_search.py | awk '{print $NF}')
# Function to check job status
check_job_status() {
    squeue -j $job_id > /dev/null 2>&1
    return $?
