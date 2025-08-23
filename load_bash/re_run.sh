echo "Deprecate: This script is deprecated. Use the wrf_indyr.sh instead."
exit 0

#!/bin/bash
#SBATCH -J submit_wrf
#SBATCH --time=10-00:00:00
#SBATCH --nodes=1
#SBATCH --partition=broadwell
#SBATCH -o submit_wrf.out
#SBATCH -e submit_wrf.err

## Example command:
usage()
{
    echo "Usage: $0 year_month dataset last_time option"
    echo "year_month: The year and month of the dataset"
    echo "dataset: The dataset name (All caps)"
    echo "last_time: The ending time of the dataset"
    echo "option: The option to run the WRF"
    echo "option: mix: Run the WRF from both compute-1 and compute-2"
    echo "option: nomix: Run the WRF from compute-1"
    exit 1
}

source /home/tamnnm/load_func_vars.sh

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
        -last_time)
            last_time="$2"
            shift 2
            ;;
        --raw_flag)
            raw_flag="$2"
            shift 2
            ;;
        --option)
            option2="$2"
            shift 2
            ;;
        *)
            echo "Invalid option: $1"
            usage-
            ;;
    esac
done

# The model folder for each year
org_folder="${wrf_test}/${dataset}_${year_month}"
# Folder for all the cases of that year
prod_folder="${prod_wrf}/wrf_${year_month}"

#Get the name of the last output if the running finished
#! Be careful if it has restarted once before.
#? The final file in that case is like this: e.g. The real last file: wrfout_d02_2019-06-01_00:00:00 => 2019-05-31_01:00:00
#last_infile=$(ls "${wps}/WPS_${dataset}_${year_month}" | grep "^met_em.d01" | sort | tail -n 1)
last_outfile="wrfout_d01_${last_time}_00:00:00"
run_script=$wrf_run/srun.sh


# get_end_time() {
#     get_value() {
#         local input_string="$1"
#         local value=$(echo "$input_string" | grep -oP '(?<=\=)[^,]+' | tr -d '[:space:]')
#         echo "$value"
#     }
# }

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
    sed -i "${line_number}s/=.*/=${new_part}/" "$run_script"
else
    # Add the new line after the ntasks line
    sed -i "${ntask_number}a #SBATCH --${key}=${new_part}" "$run_script"
fi
}


# Global variables to store job parameters
name_part=""
nodes=""
cpus=""
no_nodes=""

# FUNC: Check if the nodes are available for each partition
check_node_availability() {
    local nodes=$1
    local no_nodes=$2
    
    # format %D: print the number, %N: Node hostnames e.g.compute-1-[1-6,13-14]
    local idle_nodes=$(sinfo --noheader --states=IDLE --nodes=compute-${name_part}-[${nodes}] --format="%D")
    
    # If no idle_nodes is available, change empty string -> 0
    idle_nodes=${idle_nodes:-0}
    
    if [ $idle_nodes -ge $no_nodes ]; then
        return 0 # True
    else
        return 1 # False
    fi
}

# FUNC: Check the availability of the nodes and set the job parameters
set_job_parameters() {
    name_part=1
    no_nodes=2
    cpus=56

    # Check for scalable nodes
    nodes="7-14"
    if check_node_availability "$nodes" $((no_nodes+1)); then
        echo "Using scalable nodes: $nodes"
        return
    elif check_node_availability "$nodes" "$no_nodes"; then
        echo "Using scalable nodes: $nodes"
        return
    fi

    # Check for scaltmp nodes
    nodes="15-18"
    if check_node_availability "$nodes" $((no_nodes+1)); then
        echo "Using scalable nodes: $nodes"
        return
    elif check_node_availability "$nodes" "$no_nodes"; then
        echo "Using scaltmp nodes: $nodes"
        return
    fi

    # Check for scalable small nodes
    nodes="1-5"
    cpus=40
    no_nodes=3
    if check_node_availability "$nodes" "$no_nodes"; then
        echo "Using scalable nodes: $nodes"
        return
    fi

    if [ "$no_option" == "mix" ]; then
        # Default to broadwell nodes
        name_part=2
    
        nodes="6-7"
        no_nodes=2
        cpus=36
        if check_node_availability "$nodes" "$no_nodes"; then
            echo "Using broadwell nodes: $nodes"
            return
        fi
        
        # Default to broadwell nodes
        nodes="1-5"
        no_nodes=4
        cpus=20
        if check_node_availability "$nodes" "$no_nodes"; then
            echo "Using broadwell nodes: $nodes"
            return
        fi
    fi
    
    echo "No available nodes found"
    return 1
}

count=$(ls "${prod_folder}/${dataset}"* 2>/dev/null | wc -l)
check_id=0
check_id_2=0

# ------------------------------ MAIN ------------------------------ #
# Iterate through all the loops until all the folders are run
while [ $check_id -lt $count ]; do
    
    # Wait every time it loops through all the folders
    if [ $check_id_2 -eq $count ]; then
        sleep 60
        check_id_2=0
    fi
    
    # Iterate through all the folders
    for ind_folder in "${prod_folder}/${dataset}"*; do
        check_id_2=$((check_id_2+1))
        # Create a blank line
        echo
        
        # Check if the directory exists
        if [ -d "$ind_folder" ]; then
            # DEFINE: Basename of the folder: [dataset]_[year_month]_[num]
            ind_base_folder=$(basename "$ind_folder")
            # DEFINE: Temporary folder in the test folder
            sub_ind_folder="${wrf_test}/${ind_base_folder}"
            # DEFINE: Final folder in the /prod
            prod_ind_folder="${prod_folder}/${ind_base_folder}"
            echo "$ind_base_folder"
        else
            continue
        fi
        
        # Check if the job is already exist
        job_exists=$(squeue --noheader --format="%j" | grep -E "^($ind_folder|$ind_base_folder)$")

        if [ -n "$job_exists" ]; then
            echo "Job with name '$ind_base_folder' is already in the queue."
            check_id=$((check_id+1))
            continue
        fi
        
        # Check if the WRF has already been run
            # If yes, skip the folder
        if ls "$last_outfile"* 1> /dev/null 2>&1 || ls "$prod_ind_folder/$last_outfile"* 1> /dev/null 2>&1; then
            echo "WRF for $ind_base_folder has already been run"
            check_id=$((check_id+1))
            continue
        fid
        
        # Create temporary folder back in wrf_test if it has not existed yet
        if [ ! -d "$sub_ind_folder" ]; then
            cp -r "$org_folder" "$sub_ind_folder"
            echo "Creating $ind_base_folder"
            # DEFINE: The flag to rewrite the namelist.input and input files
            raw_flag=0
        fi
        
        # Go to the temporary folder
        cd "$sub_ind_folder"
        
        # List all files matching the pattern
            #? Check if the input files are available (i.e. wrfbdy and wrfinput)
            #! If you change surface scheme (sf), you need to RERUN "REAL.exe" to produce wrfinput. But you can use the same wrfbdy files no matter what physics schemes you will use.
            #? wrfinput is only needed for initial time but wrfbdy is needed for every time.
        
        if raw_flag -eq 1; then
            rm -f wrfbdy* wrfinput*
        fi
        
        # Check if the input files are ava ilable in the prod folder
        wrfbdy_files=($(ls "$prod_ind_folder/wrfbdy"* 2>/dev/null))
        wrfinput_files=($(ls "$prod_ind_folder/wrfinput"* 2>/dev/null))
        
        # If there is input in prod folder, copy them to the temporary folder
        if [ ${#wrfbdy_files[@]} -gt 0 ] && [ ${#wrfinput_files[@]} -gt 0 ]; then
            cp -n --preserve=context "$prod_ind_folder/wrfbdy"* "$prod_ind_folder/wrfinput"* "$sub_ind_folder/"
            check_id=$((check_id+1))
        # If input files are not available, run the real.exe
        else
            # Fix the namelist.input to match the physics scheme
            
            
            real_job_exists=$(squeue --noheader --format="%j" | grep -E "^(${ind_base_folder}_real)$")
            
            if [ -n "$real_job_exists" ]; then
                echo "Job with name '${ind_base_folder}_real' is already in the queue."
                continue
            fi
            
            cp -n --preserve=context "${sub_ind_folder}/wrfbdy"* "${sub_ind_folder}/wrfinput"* "$prod_ind_folder/"
            
        fi
        
        
        # TODO: Match the physics scheme for the namelist.input
        
        # Archive the old namelist.inputto prod folder
        cp "$sub_ind_folder/namelist.input" "$prod_ind_folder/"
        
        # ----------------------------- RESTART ---------------------------- #
        last_rst=$(ls $prod_ind_folder | grep "^wrfrst_d01_.*00:00:00$" | sort | tail -n 1)
        cp namelist.input namelist_org.input
        cp namelist_org.input "$prod_ind_folder/namelist.input"
        
        #TODO: Check these lines
        #TODO: Use the rerun/restart.txt files instead of these
        
        if [ -n "$last_rst" ] ; then
            last_folder="$prod_ind_folder"

            last_out=$(ls $last_folder | grep "^wrfout_d01_.*00:00:00$" | sort | tail -n 1)
            sub_rst=$(ls $last_folder | grep "^wrfrst_d01_.*00:00:00$" | sort | tail -n 2 | head -n 1)

            # Get the creation times of the files
            rst_creation_time=$(stat -c %Y "$last_folder/$last_rst")
            out_creation_time=$(stat -c %Y "$last_folder/$last_out")
            
            restart_flag=True
            rst_datetime=$(echo "$last_rst" | awk -F'_' '{print $3}')
            out_datetime=$(echo "$last_out" | awk -F'_' '{print $3}')
            sub_rst_datetime=$(echo "$sub_rst" | awk -F'_' '{print $3}')

            # Compare the date of wrf_rst and wrf_out
            if [ $(date -d "$rst_datetime" +%Y%m%d) -gt $(date -d "$out_datetime" +%Y%m%d) ]; then
                # First check if rst has finished
                if [ -n "$sub_rst" ]; then
                    rst_datetime=$sub_rst_datetime
                else
                    restart_flag=False
                fi
            else
                # Compare the creation times if the dates is avaliable
                if [ "$rst_creation_time" -gt "$out_creation_time" ]; then
                # Second test: The restart file has not finished
                    if [ -n "$sub_rst" ]; then
                        rst_datetime=$sub_rst_datetime
                    else
                        restart_flag=False
                    fi
                fi
            fi


            if [[ $restart_flag == "True" ]]; then
                hour_part=$(echo "$last_rst" | awk -F'_' '{print $4}')
                year=$(echo "$rst_datetime" | cut -d'-' -f1)
                month=$(echo "$rst_datetime" | cut -d'-' -f2)
                day=$(echo "$rst_datetime" | cut -d'-' -f3)
                hour=$(echo "$hour_part" | cut -d':' -f1)
                check_line_input "start_year" "$year"
                check_line_input "start_month" "$month"
                check_line_input "start_day" "$day"
                check_line_input "start_hour" "$hour"
                check_line_input "restart" ".true."
                check_line_input "override_restart_timers" ".true."
                
                check_line_input "history_outname" "${prod_ind_folder}/wrfout_d<domain>_<date>"
                check_line_input "rst_outname" "${prod_ind_folder}/wrfrst_d<domain>_<date>"
                echo "Restarting from ${rst_datetime}_${hour_part} for $ind_base_folder"
                cp -n $last_folder/wrfrst_d0*_${rst_datetime}_${hour_part} ./
            else
                echo "No restart file is available"
                # This put in case of errors. If not, there might be case that it rerun the first day.
                exit 1
            fi
        else
            check_line_input "restart_interval" "7200"
            echo Run from the beginning
            cp -f namelist_org.input namelist.input
        fi

        ulimit -s unlimited
        i=0

        case $option in
        "mix")
            no_option=4 # Run scalable
            ;;
        "nomix")
            no_option=3 # Run broadwell
            ;;
        *)
            echo "Invalid option: $option"
            exit 1
            ;;
        esac
        

        while true; do
            
        
            # Check if any nodes are in idle state

            idle_nodes=${idle_nodes:-0}

            # If there are idle nodes, submit the job and exit the loop
            if [ $idle_nodes -ge $no_nodes ]; then

                # if nodes<nodelist e.g. nodes=3 & nodelist=compute-1-[10-15] => invalid
                # solution: print the actual list of nodes e.g. compute-1-10,compute-1-13,....
                # -l: print seperate nodes
                # awk: 'condition/pattern {action}' file (e.g. check if 4th field is idle; print name'
                # paste: merge line; -s: single line; -d: delimiter

                # Get the list of available nodes in the specified range and partition
                used_nodes=$(sinfo -N -h --nodes=compute-${part}-[${nodes}] -o "%n %P %t" | awk '$3 == "idle" {print $1}' | head -${no_nodes} | sed "s/compute-${part}-//g" | paste -sd "," -)
                # Print the number of idle nodes, number of nodes to be used, and number of cpus per task
                echo "Number of idle nodes: $idle_nodes"
                echo "Number of cpus per task: $cpus"
                echo "Submitted partition: $name_part"
                echo "The nodes used: $used_nodes"
                # Replace the nodelist and partition in the temporary script
                check_line "nodelist" "compute-${part}-[${used_nodes}]"
                check_line "partition" "$name_part"
                check_line "cpus-per-task" $cpus
                check_line "ntasks" $no_nodes

                #sbatch -J "$ind_base_folder" $run_script
                echo "Submitted job with name '$ind_base_folder'"
                check_id=$((check_id+1))
                break
            else
                if [ $i == 1 ] || [ $i -gt 18000 ]; then
                    echo "Waiting for nodes with $ind_folder. Waiting till death..."
                # sleep 60  # wait for 1 hour before checking again
                fi
                sleep 1
            fi
            i=$((i+1))
        done
        cd ..
    done
done