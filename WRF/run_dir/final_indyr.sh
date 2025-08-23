#!/bin/bash
#SBATCH -J submit_wrf
#SBATCH --time=10-00:00:00
#SBATCH --nodes=1
#SBATCH --partition=broadwell
#SBATCH -o report/submit_final.out
#SBATCH -e report/submit_final.err

source /home/tamnnm/load_func_vars.sh

option="n"
mode="3"
re_real="n"

# Function to display usage information
usage() {
<<<<<<< HEAD
    echo "Usage: $0 -mode <MODE> -dataset <DATASET> -ym <YYYYMM> [--re_real <y|n>] [--check <y|n>] [--last_day <last_day>] [--option <y|n>]"
    echo "  -mode <MODE> : Mode of the run [1 - namelist only, 2 - wrf only, 3 - both - no check namelist, 4 - both - check namelist]. Default: 3"
    echo "  -dataset <DATASET> : Dataset name"
    echo "  -ym <YYYYMM> : Year and month"
    echo "  --re_real <y|n> : y - run real.exe, n - do not run real.exe. Default: n"
    echo "  --check <y|n> : y - check the namelist, n - modify the namelist. Default: y"
=======
    echo "Usage: $0 -mode <MODE> -dataset <DATASET> -ym <YYYYMM> [--re_real <y|n>] [--last_day <last_day>] [--option <y|n>]"
    echo "  -mode <MODE> : Mode of the run [0 - check namelist only, 1 - namelist only, 2 - wrf only, 3 - both - only check namelist, 4 - both - modify namelist]. Default: 3"
    echo "  -dataset <DATASET> : Dataset name"
    echo "  -ym <YYYYMM> : Year and month"
    echo "  --re_real <y|n> : y - run real.exe, n - do not run real.exe. Default: n"
>>>>>>> c80f4457 (First commit)
    echo "  --last_day <last_day> : Last day of the month"
    echo "  --option <y|n> : y - use broadwell+scalable nodes, n - use scalable nodes. Default: y"
    echo
    exit 1
}

# Parse command-line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -mode)
            mode="$2"
            shift 2
            ;;
        -dataset)
            dataset="$2"
            shift 2
            ;;
        -ym)
            year_month="$2"
            shift 2
            ;;
        --re_real)
            re_real="$2"
            shift 2
            ;;
        --last_day)
            last_day="$2"
            shift 2
            ;;
<<<<<<< HEAD
        --check)
            check="$2"
            shift 2
            ;;
=======
>>>>>>> c80f4457 (First commit)
        --option)
            option="$2"
            shift 2
            ;;
        *)
            echo "Invalid option: $1"
            usage
            ;;
    esac
done


# Check if dataset and year_month are provided
if [ -z "$mode" ] || [ -z "$dataset" ] || [ -z "$year_month" ]; then
    usage
fi

<<<<<<< HEAD
if [[ "$mode" != "1" && "$mode" != "2" && "$mode" != "3" ]]; then
    echo "Invalid mode. Mode must be 1, 2 or 3."
=======
if [[ "$mode" != "0" && "$mode" != "1" && "$mode" != "2" && "$mode" != "3" && "$mode" != "4" ]]; then
    echo "Invalid mode. Mode must be 0-4."
>>>>>>> c80f4457 (First commit)
    usage
fi

echo "Mode: $mode"

<<<<<<< HEAD
if [[ "$mode" == "4" ]]; then
=======
if [[ "$mode" == "4"  || "$mode" == "1" ]]; then
>>>>>>> c80f4457 (First commit)
    check="n"
else
    check="y"
fi

# NAMELIST RUN
if [[ "$mode" != "2" ]]; then
    echo "Running namelist..."
<<<<<<< HEAD
    
    if [[ -z "$last_day" ]]; then
        sh $wrf_run/namelist_indyr_v2.0.sh -dataset $dataset -ym $year_month --check $check
    else
        sh $wrf_run/namelist_indyr_v2.0.sh -dataset $dataset -ym $year_month --check $check --last_day $last_day
    fi
    
=======
    echo

    if [[ -z "$last_day" ]]; then
        #sh $wrf_run/namelist_indyr_v2.0.sh -dataset $dataset -ym $year_month --check $check
        wait 100
        echo
        echo "Waiting for running namelist to finish..."
        sleep 5
    else
        #sh $wrf_run/namelist_indyr_v2.0.sh -dataset $dataset -ym $year_month --check $check --last_day $last_day
        echo
        echo "Waiting for running namelist to finish..."
        wait 100
        sleep 5
    fi


>>>>>>> c80f4457 (First commit)
    # Check the exit status of the previous command
    if [ $? -eq 1 ]; then
        echo "namelist_indyr.sh exited with status 1. Exiting the script."
        exit 1
    fi
<<<<<<< HEAD
    echo "All the subfolders have been created."
fi

if [[ "$mode" == "1" ]]; then
=======
    echo
    echo "All the subfolders have been created."
fi

if [[ "$mode" == "1" || "$mode" == "0" ]]; then
>>>>>>> c80f4457 (First commit)
    echo "Mode 1: namelist only. Exiting the script."
    exit 0
fi

while true; do
    echo "Check the folder now. Do you want to proceed? [y/n]"
    read -r user_input
    if [[ "$user_input" == "y" || "$user_input" == "Y" ]]; then
        echo "Proceeding..."
        break
    elif [[ "$user_input" == "n" || "$user_input" == "N" ]]; then
        echo "Operation cancelled."
        exit 1
    else
        echo "Invalid input. Please enter 'y' or 'n'."
    fi
done

<<<<<<< HEAD
# SUBMIT THE JOB
if [ -z "$option" ]; then
    sbatch $wrf_run/wrf_indyr.sh -dataset $dataset -ym $year_month --re_real $re_real
else
    sbatch $wrf_run/wrf_indyr.sh -dataset $dataset -ym $year_month --option $option --re_real $re_real
fi
 
=======
timestamp=$(date +"%Y%m%d_%H%M%S")
log_dir="$wrf_run/logs"
mkdir -p "$log_dir"
# Create a log file with timestamp
log_file="${log_dir}/submit_wrf_${dataset}_${year_month}_${timestamp}.log"

echo "=== Job submission at $(date) ===" | tee -a "$log_file"
echo "Dataset: $dataset" | tee -a "$log_file"
echo "Year-Month: $year_month" | tee -a "$log_file"
echo "Mode: $mode" | tee -a "$log_file"
echo "Re-run real.exe: $re_real" | tee -a "$log_file"
echo "Option: $option" | tee -a "$log_file"
echo "----------------------------------------" | tee -a "$log_file"


# SUBMIT THE JOB
if [ -z "$last_day" ]; then
    job_id=$(sbatch $wrf_run/wrf_indyr.sh -dataset $dataset -ym $year_month --option $option --re_real $re_real | grep -o '[0-9]\+')
else
    job_id=$(sbatch $wrf_run/wrf_indyr.sh -dataset $dataset -ym $year_month --option $option --re_real $re_real --last_day $last_day | grep -o '[0-9]\+')
fi

echo "Job submitted with ID: $job_id" | tee -a "$log_file"
echo "Log saved to: $log_file" | tee -a "$log_file"

# Set up a monitor to copy the output file when it's updated
(
    # Wait for the job to start
    sleep 30

    # Keep track of what we've already logged
    out_lines=0
    err_lines=0

    # Monitor the output file and append to our log
    while squeue -j $job_id &>/dev/null; do
        sleep 300  # Check every 5 minutes
    done

    # Final copy after job completes
    echo "=== Job completed at $(date) ===" >> "$log_file"

    # Copy full output file
    if [ -f "submit_wrf.out" ]; then
        echo "=== FULL OUTPUT ===" >> "$log_file"
        cat "submit_wrf.out" >> "$log_file"
        cp "submit_wrf.out" "${log_dir}/submit_wrf_${dataset}_${year_month}_${timestamp}.out"
    fi

    # Copy full error file
    if [ -f "submit_wrf.err" ]; then
        echo "=== FULL ERROR LOG ===" >> "$log_file"
        cat "submit_wrf.err" >> "$log_file"
        cp "submit_wrf.err" "${log_dir}/submit_wrf_${dataset}_${year_month}_${timestamp}.err"
    fi

) &

>>>>>>> c80f4457 (First commit)
