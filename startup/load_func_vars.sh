#!/bin/bash
# load_func_vars.sh

source /home/tamnnm/load_env_vars.sh

#$1: GG = My Drive (e.g.GG/Code/noaa)

#$2: Destination

crclonesync(){
 nohup rclone sync "$1" "$2" --retries=9999 &
}

crclonecopy() {
 nohup rclone copy "$1" "$2" --retries=9999 &
}

runout(){
    { time python "$1"; } 2>&1 | tee $code/output.txt
}

run() {
    if [[ "$1" == *.py ]]; then
        time python "$@"
    elif [[ "$1" == *.sh ]]; then
        sbatch "$@"
    else
        echo "Unsupported file type: $1"
    fi
}

# ------------------------- SUBMITIING JOB ------------------------- #

runtmp(){
 # Check if at least one argument is provided
    if [ "$#" -lt 1 ]; then
        echo "Usage: runtmp [partition] <python_script> [arguments...]"
        return 1
    fi

    # Extract the job name from the first argument
    job_name=$(basename "$1" .py)

    partition="scaltmp"

    # Create a temporary job script with replaced placeholders
    sed -e "s/JOBNAME/${job_name}/" -e "s/PARTITION/${partition}/" "$code/submit_python.sh" > tmp_submit.sh
    sbatch tmp_submit.sh "$@"
}

runable(){
  # Check if at least one argument is provided
    if [ "$#" -lt 1 ]; then
        echo "Usage: runtmp [partition] <python_script> [arguments...]"
        return 1
    fi

    # Extract the job name from the first argument
    job_name=$(basename "$1" .py)

    partition="scalable"

    # Create a temporary job script with replaced placeholders
    sed -e "s#JOBNAME#${job_name}#" -e "s/PARTITION/${partition}/" "$code/submit_python.sh" > tmp_submit.sh
    sbatch tmp_submit.sh "$@"
}

runbroadwell(){
  # Check if at least one argument is provided
    if [ "$#" -lt 1 ]; then
        echo "Usage: runtmp <job_name> [partition] <python_script> [arguments...]"
        return 1
    fi

    # Extract the job name from the first argument
    job_name=$(basename "$1" .py)

    partition="broadwell"

    # Create a temporary job script with replaced placeholders
    sed -e "s#JOBNAME#${job_name}#" -e "s/PARTITION/${partition}/" "$code/submit_python.sh" > tmp_submit.sh
    sbatch tmp_submit.sh "$@"
}

# ----------------- ADDITIONAL FORMATTING FUNCTIONS ---------------- #
month_value(){
    local month=$1
    if [ "$month" -lt 10 ]; then
        echo "0$month"
    else
        echo "$month"
    fi
}

read_file() {
    local file=$1
    local line
    while IFS= read -r line; do
        echo "$line"
    done < "$file"
}



cf2pyi(){
    local fortran_file="$1"
    local base_name=$(basename "$fortran_file" .f90)
    f2py -c -m "$base_name" --fcompiler=intelem "$fortran_file"
}

# -------------------- WRF FORMATIING FUNCTIONS -------------------- #
formatted_namelist(){
  local input_file=$1
  tmp_file=temp.input

  while IFS= read -r line; do
      # Trim whitespace from the beginning and end of the line
      trimmed_line=$(echo "$line" | sed 's/^[ \t]*//;s/[ \t]*$//')

      # Check if the line contains an '=' character
      if [[ "$trimmed_line" == *"="* ]]; then
          # Split the line at the '=' character
          left_part=$(echo "$trimmed_line" | cut -d'=' -f1)
          right_part=$(echo "$trimmed_line" | cut -d'=' -f2)

          # Format the right part to have a length of 40 characters
          formatted_left_part=$(printf "%-40s" "$left_part")

          # Split the right part by ',' and assign each part 9 spaces

          #? If it has only one part, print it with ","
          #? If it has more than one part, print the each part with "," as 9 spaces

          formatted_right_part=$(echo "$right_part" | awk -F, '{
            if (NF == 1) {
              gsub(/^[ \t]+|[ \t]+$/, "", $i);
              printf $1",";
              }
            else {
              for(i=1;i<NF;i++) {
                gsub(/^[ \t]+|[ \t]+$/, "", $i);
                printf "%-9s", $i",";
              }
              print "";
            }
          }'
          )
          # Combine the left and formatted right parts
          formatted_line="$formatted_left_part= $formatted_right_part"
          trimmed_formatted_line=$(echo "$formatted_line" | sed 's/^[ \t]*//;s/[ \t]*$//')
      else
          # If the line doesn't contain an '=', keep it as is
          formatted_line="$trimmed_line"
      fi

      # Write the formatted line to the output file
    echo "$formatted_line" >> "$tmp_file"
  done < "$input_file"
  mv "$tmp_file" "$input_file"
  }

remove_dot_slash_prefix() {
local folder=$1
if [[ $folder == ./* ]]; then
  echo "${folder:2}"
else
  echo "$folder"
fi
}

# ------------------------ WRF INPUT REPLACE ----------------------- #

check_line_input() {
  local key=$1
  local new_part=$2
  local line_number

  # Find the line number that starts with the key
  line_number=$(grep -n "^[[:space:]]*${key}[[:space:]]*=" "namelist.input" | cut -d: -f1)

  if [[ "$key" == *"outname"* ]]; then
    input="\'${new_part}\'"
  elif  [[ "$key" == *"restart"* || "$key" == *"output"* ]]; then
    input="${new_part}"
  else
    input="${new_part}, ${new_part}"
  fi
  
  # If the line number is found, replace the line
  if [ -n "$line_number" ]; then
    sed -i "${line_number}s#=.*#= ${input},#" "namelist.input"
  else
    first_slash_line=$(grep -n "^[[:space:]]*/" "namelist.input" | cut -d: -f1 | head -n 1)
    sed -i "${first_slash_line}i ${key} = ${input}," "namelist.input"
  fi
 }

## ----------------------- CHECK NODE AVAILABILITY ----------------------- ##
pending_nodes() {
    squeue -u tamnnm -h -t PENDING -o "%i" | \
        while read jobid; do
            scontrol show jobs $jobid | grep "ReqNodeList" | cut -d'=' -f2 | cut -d' ' -f1
        done | tr ',' '\n'
}


check_node_availability() {
    local nodes=$1
    local no_nodes=$2
    
    # format %D: print the number, %N: Node hostnames e.g.compute-1-[1-6,13-14]
    #? "%D" is already the counting => No-need to use wc -l
    local idle_nodes=$(sinfo --noheader --states=IDLE --nodes=compute-${name_part}-[${nodes}] --format="%D")
    # If no idle_nodes is available, change empty string -> 0
    idle_nodes=${idle_nodes:-0}
    if [ $idle_nodes -ge $no_nodes ]; then
        # echo out the value instead of setting a variable
        sinfo -N -h --nodes=compute-${name_part}-[${nodes}] -o "%n %P %t" | \
            awk '$3 == "idle" {print $1}' | head -${no_nodes} | \
            sed "s/compute-${name_part}-//g" | paste -sd "," -
        return 0 # True
    fi
    return 1 # False
}


check_node_availability_mix() {
    local nodes=$1
    local no_nodes=$2
    local required_cpus=$3
    
    # Get nodes in MIXED state with their CPU info
    # %C: CPU count in ALLOCATED/IDLE/OTHER/TOTAL format
    # %n: Node name
    local available_nodes=$(sinfo --noheader --states=MIX,IDLE --nodes=compute-${name_part}-[${nodes}] --format="%n %C" | \
        while read node cpu_info; do
            # Extract idle CPUs (second number in CPU format A/I/O/T)
            # Convert to integer for comparison
            idle_cpus=$(($(echo $cpu_info | cut -d'/' -f2)))
            if [ "$idle_cpus" -ge "$required_cpus" ]; then
                echo "$node"
            fi
        done| wc -l)
    
    # If no available_nodes is found, change empty string -> 0
    available_nodes=${available_nodes:-0}
    
    ##? NOTICE
    # Don't think about check IDLE and MIX the same time by >= mixed_cpus
    # => We should use the same configuration for all nodes
    # => We should prioritize IDLE nodes
    
    if [ $available_nodes -ge $no_nodes ]; then
        sinfo --noheader --states=MIX,IDLE --nodes=compute-${name_part}-[${nodes}] --format="%n %C %t" | \
            while read node cpu_info state; do
                idle_cpus=$(($(echo $cpu_info | cut -d'/' -f2)))
                node_num=$(echo $node | cut -d'-' -f3)
                if [ "$idle_cpus" -ge "$required_cpus" ]; then
                    echo "$node"
                fi
            done | head -${no_nodes} | sed "s/compute-${name_part}-//g" | paste -sd "," -
        return 0 # True
    fi
    return 1 # False
}

# FUNC: Check the availability of the nodes and set the job parameters
# return $node $no_nodes $cpus $full_part(name) $name_part(number)
# return 0 if the nodes are available, 1 if not

try_node_configuration() {
    local name_part=$1
    local full_part=$2
    local nodes=$3
    local no_nodes=$4
    local cpus=$5
    local mix_cpus=${6:-0}  # Optional parameter for mixed mode
    local pending_nodes
    
    # Running nodes but already pending
    pending_nodes=$(pending_nodes)
    
    used_nodes=$(check_node_availability "$nodes" "$no_nodes")
        
    echo "Used nodes: $used_nodes"
    if [ $? -eq 0 ] && [ -n "$used_nodes" ]; then
        echo "Using ${full_part} nodes: $nodes"
        echo "PARAMS:${name_part}:${full_part}:${nodes}:${no_nodes}:${cpus}:${used_nodes}"
        return 0
    fi
    
    # Only try mixed mode if mix_cpus is provided
    if [ $mix_cpus -gt 0 ]; then
        used_nodes=$(check_node_availability_mix "$nodes" "$((no_nodes + 1))" "$mix_cpus")
        echo "Used nodes: $used_nodes"
        if [ $? -eq 0 ] && [ -n "$used_nodes" ]; then
            no_nodes=$((no_nodes + 1))
            echo "Using mixed ${full_part} nodes: $nodes"
            echo "PARAMS:${name_part}:${full_part}:${nodes}:${no_nodes}:${mix_cpus}:${used_nodes}"
            return 0
        fi
    fi

    return 1
}

set_job_parameters() {
    local name_part=1
    local no_nodes=2
    local option=$1
    
    # Try scaltmp nodes (15-18)
    try_node_configuration "$name_part" "scaltmp" "15-18" "$no_nodes" "56" "40" && return 0
    
    # Try scalable small nodes (1-6)
    try_node_configuration "$name_part" "scalable" "1-6" "3" "40" && return 0
    
    # Scalable nodes is having some trouble making it frezze the jobs
    
    # Try scalable nodes (7-14)
    try_node_configuration "$name_part" "scalable" "7-14" "$no_nodes" "56" "40" && return 0
    
    # Only try broadwell nodes if option is "y"
    if [ "$option" == "y" ]; then
        name_part=2
        # Try broadwell nodes (6-7)
        try_node_configuration "$name_part" "broadwell" "6-7" "2" "36" && return 0
        
        # Try broadwell nodes (1-5)
        try_node_configuration "$name_part" "broadwell" "1-5" "4" "20" && return 0
    fi
    
    return 1
}



# --------------------- CHECK USER'S JOB COUNT --------------------- #
check_jobs() {
    local max_jobs=5 # Actual maximum: 4 = 5 - 1 (file controlling the job submission)
    local job_count=$(squeue -u tamnnm --states=R,PD | wc -l)
    # Subtract 1 from job_count to account for the header line in squeue output
    job_count=$((job_count - 1))
    echo "Current job count: $job_count"
    if [ "$job_count" -gt "$max_jobs" ]; then
        return 1 # More than max_jobs
    else
        return 0 # Less than or equal to max_jobs
    fi
}

# --------------------- CHECK JOB EXISTS --------------------- #
job_exists() {
    local job_name=$1
    # If jobs is being cancelled, it will still appear as CG in status => Add status
    if squeue --noheader --format="%j %T" | grep -E "^(${job_name}) (RUNNING|PENDING)$" > /dev/null; then
        return 0
    else
        return 1
    fi
}
