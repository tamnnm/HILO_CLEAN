#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=broadwell
#SBATCH -o submit_wrf.txt

year=$1
month=$2
nodes=$3

# Need to finish for 1945 and 1946
# Need to run from 48 back
for ((j=0; j<43; j++)); do
  folder=$wrf/WRF/test/RUN_ERA5_"${year}${month}_${j}"
  if ((${year} == 1944)); then
  	fin_year=$((year+1))
 	fin_month=2
        no_nodes="8-11"
  else
	fin_year=${year}
 	fin_month=$((month+1))
        no_nodes="1-5"
  fi
  cd $folder
  ulimit -s unlimited
while true; do
  if [ -f $folder/"wrfout_d02_${fin_year}-0${fin_month}-01_00:00:00" ]; then
  #if ls $folder; then
    echo Skip $folder
    cd ..
    break
  else
      # Check if any nodes are in idle state
      idle_nodes=$(sinfo --noheader --states=IDLE --nodes=compute-1-\[${no_nodes}\] --format="%D")
      idle_nodes=${idle_nodes:-0}
      # If there are idle nodes, submit the job and exit the loop
      if [ $idle_nodes -ge $nodes ]; then
        sbatch -J "${year}_${j}" --nodes=${nodes} --nodelist=compute-1-\[${no_nodes}\] ../run_srun.sh
	echo "RUN folder $folder"
        break
      else
        echo "Waiting for $folder"
        echo "Waiting for nodes. Waiting till death..."
        sleep 86400  # wait for 1 day before checking again
      fi
  fi
  done
  cd ..
 done
#wait
