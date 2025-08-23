#!/bin/bash
#SBATCH --time=10-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=broadwell
#SBATCH -o submit_wrf.txt

year=$1
month=$2
nodes=$3

# Need to finish for 1945 and 1946
# Need to run from 48 back
for ((j=10; j<40; j++)); do
  folder=$wrf/WRF/test/RUN_ERA5_"${year}${month}_${j}"
  if ((${year} == 1944)); then
  	fin_year=$((year+1))
 	fin_month=2
        no_nodes="8-14"
  else
	fin_year=${year}
 	fin_month=$((month+1))
        no_nodes="1-5"
  fi
  cd $folder
  ulimit -s unlimited
  while true; do
    if [ -f $folder/"wrfout_d02_${fin_year}-0${fin_month}-01_00:00:00" ]; then
      #echo Skip RUN_ERA5_"${year}${month}_${j}"
      break
    else
        # Check if any nodes are in idle state

	# format %D: print the number, %N: Node hostnames e.g.compute-1-[1-6,13-14]
        idle_nodes=$(sinfo --noheader --states=IDLE --nodes=compute-1-\[${no_nodes}\] --format="%D")
	# If no idle_nodes is available, change empty string -> 0
        idle_nodes=${idle_nodes:-0}
        # If there are idle nodes, submit the job and exit the loop
        if [ $idle_nodes -ge $nodes ]; then
	  # if nodes<nodelist e.g. nodes=3 & nodelist=compute-1-[10-15] => invalid
	  # solution: print the actual list of nodes e.g. compute-1-10,compute-1-13,....
	  # -l: print seperate nodes
	  # awk: 'condition/pattern {action}' file (e.g. check if 4th field is idle; print name'
	  # paste: merge line; -s: single line; -d: delimiter
	  used_nodes=$(sinfo -N -l --noheader --nodes=compute-1-\[${no_nodes}\] | awk '$4=="idle" {print $1}' | head -3 | paste -sd ",")
          echo Number of available nodes: $idle_nodes, $no_nodes, $used_nodes
          sbatch -J "${year}_${j}" --nodes=${nodes} --nodelist=${used_nodes} ../run_srun.sh
          echo "RUN "${year}${month}_${j}""
          break
        else
          echo "Waiting for nodes with $folder. Waiting till death..."
          sleep 7200  # wait for 1 day before checking again
        fi
      fi
  done
  cd ..
 done
#wait
