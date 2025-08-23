#!/bin/bash
#SBATCH --output=run_output.txt
#SBATCH --time=1-00:00:00 
source /home/tamnnm/load_func_vars.sh

root_folder="../grid_0.7"

for anom in $(seq -1.0 -2.0 -5.0); do
 if [ ! -d "$root_folder/output_$anom" ]; then
   mkdir "$root_folder/output_$anom" 
 fi

 if [ ! -d "$root_folder/res_$anom" ]; then
   mkdir "$root_folder/res_$anom"
 fi

 for i in $(seq 1005.0 0.5 1010.5); do
  for j in $(seq 2.0 0.25 4.0); do
   for k in $(seq 0.8 0.05 1.5); do
    for l in $(seq 0.2 0.2 0.6); do
       while true; do
           job_count=$(squeue -u tamnnm --noheader | wc -l)
           if ((job_count < 60)); then
               break
           fi
           echo "60 jobs are running, waiting..."
           sleep 60
       done 
       runable "$root_folder/output_$anom/test_${i}_${j}_${k}_${l}" main_search.py $i $j $k $anom $l
      sleep 5
      echo test_${i}_${j}_${k}_${anom}_${l}
    done
   done
 done
done
