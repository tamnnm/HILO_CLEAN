#!/bin/bash
#SBATCH -J wrf_tamnnm_root
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=broadwell
#SBATCH --exclusive

#year=(1968 2016 2023)
#month=(2 3 4)

year=(1944 1968)
month=(12 2)
nodes=4

# Need to finish for 1945 and 1946
# Need to run from 48 back
for ((j=1; j<43; j++)); do

  folder=$wrf/WRF/test/RUN_ERA5_"${year[$i]}${month[$i]}_${j}"
  if ((${year[$i]} == 1944)); then
  	fin_year=$((year[$i]+1))
 	  fin_month=2
	  srun="big"
    no_nodes="7-10"
  else
	  fin_year=${year[$i]}
 	  fin_month=$((month[$i]+1))
	  srun="small"
    no_nodes="1-4"
  fi
  cd $folder
  ulimit -s unlimited

  if [ -f $folder/"wrfout_d02_${fin_year}-0${fin_month}-01_00:00:00" ]; then
  #if ls $folder; then
    echo Skip $folder
    cd ..
    continue
  else
    while true; do
      # Check if any nodes are in idle state
      idle_nodes=$(sinfo --noheader --states=IDLE --nodes=compute-1-${no_nodes} --format="%N" | wc -l)
      # If there are idle nodes, submit the job and exit the loop
      if [ $idle_nodes -eq ${nodes} ]; then
        sbatch -J --nodes=${nodes} "tam_${year[$i]}$_${j}" ../run_srun_"${srun}".sh
        break
      else
        echo "Waiting for nodes. Waiting till death..."
        sleep 1800  # wait for 60 seconds before checking again
      fi
    done
    echo "RUN folder $folder"
    cd ..
  fi
  done
done
#wait
