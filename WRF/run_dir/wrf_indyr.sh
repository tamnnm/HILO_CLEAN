#!/bin/bash
#SBATCH -J submit_wrf
#SBATCH --time=10-00:00:00
#SBATCH --nodes=1
#SBATCH --partition=broadwell
#SBATCH -o report/submit_wrf.out
#SBATCH -e report/submit_wrf.err

source /home/tamnnm/load_func_vars.sh

option="y"
re_real="n"

# Function to display usage information
usage() {
<<<<<<< HEAD
    echo "Usage: $0 -dataset <DATASET> -ym <YYYYMM> [--re_real <y|n>] [--option <y|n>]"
    echo " --option <y|n>: y - use broadwell+scalable nodes, n - use scalable nodes"
=======
    echo "Usage: $0 -dataset <DATASET> -ym <YYYYMM> [--re_real <y|n>] [--option <y|n>] [--last_day <last_day>]"
    echo " --option <y|n>:        y - use broadwell+scalable nodes, n - use scalable nodes"
    echo " --last_day (optional): eg: 2023-01-31"
>>>>>>> c80f4457 (First commit)
    exit 1
}

# Parse command-line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -dataset)
            dataset="$2"
            shift 2
            ;;
        -ym)
            year_month="$2"
            shift 2
            ;;
        --option)
            option="$2"
            shift 2
            ;;
        --re_real)
            re_real="$2"
            shift 2
            ;;
<<<<<<< HEAD
=======
        --last_day)
            last_day="$2"
            shift 2
            ;;
>>>>>>> c80f4457 (First commit)
        *)
            echo "Invalid option: $1"
            usage
            ;;
    esac
done


# Check if dataset and year_month are provided
if [ -z "$dataset" ] || [ -z "$year_month" ]; then
    usage
fi

<<<<<<< HEAD
=======

if [ -z "$last_day" ]; then
    wps_path="${wps}/WPS_${dataset}_${year_month}"
    last_infile=$(ls "${wps_path}" | grep "^met_em.d01" | sort | tail -n 1)
    last_day=$(echo $last_infile | cut -d'.' -f3 | cut -d'_' -f1)
fi

last_outfile="wrfout_d01_${last_day}"

>>>>>>> c80f4457 (First commit)
# All directories
root_folder=${dataset}_${year_month}
org_folder="${wrf_test}/${root_folder}"
prod_folder="${prod_wrf}/wrf_${year_month}"
run_script=$wrf_run/run_srun.sh


# get_end_time() {
#     get_value() {
#         local input_string="$1"
#         local value=$(echo "$input_string" | grep -oP '(?<=\=)[^,]+' | tr -d '[:space:]')
#         echo "$value"
#     }
# }

# Global variables to store job parameters
name_part=""
nodes=""
cpus=""
no_nodes=""

# FUNC: Check the configuration of the jobs
check_line() {
  local key=$1
  local new_part=$2
  local line_number
  local ntask_number

  # Find the line number that starts with the key
  line_number=$(grep -n "#SBATCH --${key}=" "${run_script}" | cut -d: -f1)
  ntask_number=$(grep -n "#SBATCH --ntasks=" "${run_script}" | cut -d: -f1)
  # If the line number is found, replace the line
  if [ -n "$line_number" ]; then
    sed -i "${line_number}s#=.*#=${new_part}#" "$run_script"
  else
    # Add the new line after the ntasks line
    sed -i "${ntask_number}a #SBATCH --${key}=${new_part}" "$run_script"
  fi
}

run_real_exe() {
    if [ -f namelist.input ]; then
        # Move the namelist.input
        mv namelist.input temp.input
        cp "$prod_ind_folder/namelist.input" ./
        # Run the real.exe
        sbatch -J "${ind_base_folder}_real" "$wrf_run/run_real.sh"
        echo "Running real.exe for $ind_base_folder"
        # Return the changed namelist.input
        mv temp.input namelist.input
    fi
}

# ------------------------------ MAIN ------------------------------ #
cd $wrf_test
folders=(${prod_folder}/${root_folder}_*)
last_folder=${folders[-1]}
count=$(ls "${prod_folder}/${root_folder}_"* 2>/dev/null | wc -l)
check_id=0
<<<<<<< HEAD
check_id_2=0
=======
>>>>>>> c80f4457 (First commit)

# Iterate through all the loops until all the folders are run
while [ $check_id -lt $count ]; do
    
    # List through all the folders in the prod folder
    for ind_folder in ${prod_folder}/${root_folder}_*; do
        
        # Create a blank line
        echo
    
        if [ -d "$ind_folder" ]; then
            # DEFINE: Basename of the folder: [dataset]_[year_month]_[num]
            ind_base_folder=$(basename "$ind_folder")
            # DEFINE: Temporary folder in the test folder
            sub_ind_folder="${wrf_test}/${ind_base_folder}"
            # DEFINE: Final folder in the /prod
            prod_ind_folder="${prod_folder}/${ind_base_folder}"
<<<<<<< HEAD
=======
            
>>>>>>> c80f4457 (First commit)
            echo "Checking $ind_base_folder"
        else
            continue
        fi
        
<<<<<<< HEAD
        # If the sub_ind_folder exists, this case needs to be re_run/re_start
        if [ -d "$sub_ind_folder" ]; then
            cd $sub_ind_folder
        else
            # Complete one folder
            check_id=$((check_id+1))
=======
        echo $last_outfile
        ls "$last_outfile"* 1> /dev/null 2>&1
        ls "$prod_ind_folder/$last_outfile"* 1> /dev/null 2>&1
    
        if ls "$last_outfile"* 1> /dev/null 2>&1 || ls "$prod_ind_folder/$last_outfile"* 1> /dev/null 2>&1; then
            if [ -d "${ind_base_folder}" ]; then
                echo "Archiving the base folder....."
                mv "${ind_base_folder}" "/work/users/tamnnm/trashbin/WRF_archive/"
                sleep 15
                echo
            fi
            echo "WRF for $ind_base_folder has already been run."
>>>>>>> c80f4457 (First commit)
            continue
        fi
        
        # Check if the job is already exist, if yes => skip
        job_exists "$ind_folder|$ind_base_folder"
        wrf_job_exists=$?
        if [ $wrf_job_exists -eq 0 ]; then
            echo "Job with name '$ind_base_folder' is already in the queue."
            # Complete one folder
            check_id=$((check_id+1))
            continue
        fi
        
<<<<<<< HEAD
=======
        # If the sub_ind_folder exists, this case needs to be re_run/re_start
        if [ -d "$sub_ind_folder" ]; then
            cd $sub_ind_folder
        else
            # Complete one folder
            check_id=$((check_id+1))
            continue
        fi
        
>>>>>>> c80f4457 (First commit)
        # ----------------- TREAT WRFBDY AND WRFINPUT FILES ---------------- #
        job_exists "${ind_base_folder}_real"
        real_job_exists=$?
        
        if [ "$re_real" == "y" ]; then
            if [ $real_job_exists -eq 0 ]; then
                echo "Job with name '${ind_base_folder}_real' is already in the queue."
            else
                run_real_exe
            fi
            cd ..
            continue
        else
            wrfbdy_files=($(ls "$sub_ind_folder/wrfbdy"* 2>/dev/null))
            wrfinput_files=($(ls "$sub_ind_folder/wrfinput"* 2>/dev/null))
            
            if [ ${#wrfbdy_files[@]} -eq 0 ] || [ ${#wrfinput_files[@]} -eq 0 ]; then
                prod_wrfbdy_files=($(ls "$prod_ind_folder/wrfbdy"* 2>/dev/null))
                prod_wrfinput_files=($(ls "$prod_ind_folder/wrfinput"* 2>/dev/null))
                
                # If the input files is not in sub folder, copy them from prod folder
                #? Meaning: real.exe has been run => No need to run again
                if [ ${#prod_wrfbdy_files[@]} -gt 0 ] || [ ${#prod_wrfinput_files[@]} -gt 0 ]; then
                    cp -n "$prod_ind_folder/wrfbdy"* "$prod_ind_folder/wrfinput"* "$sub_ind_folder/"
                    # check_id=$((check_id+1))
                else
                # If prod folder has no input files neither, run the real.exe
                    # If the job is already in the queue, skip
                    if [ $real_job_exists -eq 0 ]; then
                        echo "Job with name '${ind_base_folder}_real' is already in the queue."
                    else
                        # Just check again for the existence of the namelist.input
                        run_real_exe
                        # Skip this case to check other cases first until it finishes
                    fi
                    cd ..
                    continue
                fi
            fi
        fi
        
        # Archive the input file if not yet
        if ([ ${#prod_wrfbdy_files[@]} -eq 0 ] || [ ${#prod_wrfinput_files[@]} -eq 0 ]) || [ "$re_real" == "y" ]; then
            cp "${sub_ind_folder}/wrfbdy"* "${sub_ind_folder}/wrfinput"* "$prod_ind_folder/"
        fi
        
        # --------------------------- MAIN SUBMIT -------------------------- #
        i=0
        
        while true; do
        
        # Check if any nodes are in idle state
            # Return idle_nodes: number of idle nodes
            result=$(set_job_parameters "$option") # 2>&1 >/dev/null)
            status=$?
            
            if [ $status -ne 0 ]; then
                if [[ $i -eq 1 ]]; then
                    echo "Start waiting..."
                fi

                if (( i % 60 == 0 )) ; then
                    echo "Waiting for $((i / 60)) hours...."
                fi
                sleep 60
                i=$((i+1))
            
            # If reach maximum simutaneous job, wait for 2 mins
            elif ! check_jobs; then
                echo "Maximum jobs achieved. Wait for 10 minutes..."
                sleep 600 # Wait for 10 minutes
            else
                params_line=$(echo "$result" | grep "^PARAMS:" | tail -n1)
                IFS=':' read -r _ name_part full_part nodes no_nodes cpus used_nodes <<< "$params_line"
                # if nodes<nodelist e.g. nodes=3 & nodelist=compute-1-[10-15] => invalid
                # solution: print the actual list of nodes e.g. compute-1-10,compute-1-13,....
                # -l: print seperate nodes
                # awk: 'condition/pattern {action}' file (e.g. check if 4th field is idle; print name'
                # paste: merge line; -s: single line; -d: delimiter
                
<<<<<<< HEAD
=======
                mkdir -p ${wrf_run}/report/${root_folder}
                
>>>>>>> c80f4457 (First commit)
                # Print the number of idle nodes, number of nodes to be used, and number of cpus per task
                echo "Number of cpus per task: $cpus"
                echo "Submitted partition: $full_part"
                echo "The nodes used: compute-${name_part}-[${used_nodes}]"
                # Replace the nodelist and partition in the temporary script
                check_line "nodelist" "compute-${name_part}-[${used_nodes}]"
                check_line "partition" "$full_part"
                check_line "cpus-per-task" "$cpus"
                check_line "ntasks" "$no_nodes"
                check_line "output" "$wrf_run/report/${root_folder}/${ind_base_folder}.out"
                check_line "error" "$wrf_run/report/${root_folder}/${ind_base_folder}.err"
                
<<<<<<< HEAD
                echo "Submitted job: $ind_base_folder"
                echo "Submitted time: $(date)"
                sbatch -J "$ind_base_folder" $run_script
=======
                job_id=$(sbatch -J "$ind_base_folder" $run_script | grep -o '[0-9]\+')
                echo "Submitted job: $ind_base_folder with ID: $job_id"
                echo "Submitted time: $(date)"
>>>>>>> c80f4457 (First commit)
                echo
                sleep 30
                break
                
                # Complete one folder
                check_id=$((check_id+1))
            fi
        done
        cd ..
    done
    
    # Complete one round check
    if [ "$ind_folder" == "$last_folder" ]; then
        # Stop after rerun all the real.exe for all folder
        re_real="n"
        sleep 120
    fi
done

echo "All jobs have been submitted."