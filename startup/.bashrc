# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
  . /etc/bashrc
fi

# User specific aliases and functions
# VISUALISATION
source ~/load-baseEnv.sh
source ~/load_func_vars.sh
## This has been source in load_func_vars.sh
##source ~/load_env_vars.sh

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
force_color_prompt=yes


cnohup() {
    timestamp=$(date +"%Y%m%d_%H%M%S")  # Generate a timestamp
    log_file="nohup_${timestamp}.log"  # Create a log file name with the timestamp
    nohup "$@" >> "$log_file" 2>&1 &   # Redirect stdout and stderr to the log file
    echo "Output is being logged to $log_file"
}

conohup(){
nohup "$@" > /dev/null 2>&1 &
}

mcode(){
 local prefix=$1
 local base_dir="$code/01.main_code"

 for dir in "$base_dir"/"$prefix"*; do
  if [ -d "$dir" ]; then
  cd "$dir"
   return
  else
   cd "$base_dir"
  fi
 done
 # If no directory is found, change to the base directory
}

hisncdump() {
 local input_file=$1
 ncdump -h "$input_file" | grep history
}

link() {
 local folder=$1
 local name_folder=$2
 ln -rsf $1 $name_folder
}

crename() {
    # Usage: crename "old_pattern" "new_pattern" file1 file2...
    if [ $# -lt 3 ]; then
        echo "Usage: crename old_pattern new_pattern files..."
        return 1
    fi

    old="$1"
    new="$2"
    shift 2

    for file in "$@"; do
        if [ ! -e "$file" ]; then
            echo "Error: File '$file' does not exist"
            continue
        fi

        newname="${file//$old/$new}"
        if [ "$file" != "$newname" ]; then
            echo "Renaming: $file -> $newname"
            mv -- "$file" "$newname"
        fi
    done
}


alias up="cd .."
alias back="cd -"
alias upls="cd .. & ls"
alias cdw="cd $work"
alias prodw="cd $myprod"
alias codew="cd $code"
alias mcodew="cd $mcode"
alias ccodew="cd $current_code"
alias supw="cd $sup"
alias pkgw="cd $pkg"
alias dataw="cd $data"
alias cdataw="cd $current_data"
alias paraw="cd $data/wrf_data/netcdf/para"
alias obsw="cd $data/obs"
alias data45w="cd $data/pap25_QA_1945"
alias calw="cd $data/pap25_QA_1945/cal"
alias wgetcc="wget --no-check-certificate"
alias junkw="cd ~/.conda/envs/tamnnm/lib/python3.9/site-packages/my_junk/"
alias brvi="vi ~/.bashrc"
alias brcat="cat ~/.bashrc"
alias brs="source ~/.bashrc"
alias loadvi="vi ~/load-baseEnv.sh"
alias loadcat="cat ~/load-baseEnv.sh"
alias funcvi="vi ~/load_func_vars.sh"
alias funccode="code ~/load_func_vars.sh"
alias funcat="cat ~/load_func_vars.sh"
alias envvi="vi ~/load_env_vars.sh"
alias envcat="cat ~/load_env_vars.sh"
alias envcode="code  ~/load_env_vars.sh"
alias changevi="vi $work/change.txt"
alias changecode="code $work/change.txt"
alias changecat="cat $work/change.txt"
alias twcrw="cd $data/pap25_QA_1945/cal/twcr"
alias wrfw="cd $wrf"
alias wrft="cd $wrf_test/"
alias wpsp="cd $wrf/WPS_All"
alias wrfnc="cd $wrf_data/netcdf"
alias wrfg="cd $wrf_data/gribfile"
alias wrfp="cd $prod_wrf"
alias wrfr="cd $wrf/run_dir"
alias treatw="cd $work/treat_data"
alias datathay="cd /work/users/thanhnd/dat_OBS_grid"
alias datatung="cd /data/projects/LOTUS/tungnd/"
alias trashw="cd $trash"
alias imgw="cd $img"
#command
alias ctest="python $test/test.py"
alias otest="code $test/test.py"
alias itest="cd $test/image_test"
alias cpython='function _cpython(){ python "$1" >> output.txt; };_cpython'
alias csrun="srun -p scalable -J 'python' -n 40 -N 1 --time 24:00:00 --pty bash"
alias csrunb="srun -p broadwell -J 'python' -n 20 -N 1 --time 24:00:00 --pty bash"
alias crcloneecera="nohup rclone sync GG:Code/wrf/ecmwf/cera $data/wrf_data/netcdf/cera_6h_1900_2010 --retries=9999 &"
 ##retries tell rclone to retry indefinitely even though the connection failed
 ## if you used rsync, you can use parallel to use as many nodes as you want
alias rclonewc="rclone lsf GG:PhD/Code/wrf/hourly | wc -l"
#some command for list
alias la="ls -A"
alias ll="ls -alFhX"
alias ls="ls -X"
alias lwc="ls -l | wc -l"
alias df="df -h"
alias du="du -sh"
alias duw="du -sh $work"
alias dudata="du -sh /data/projects/REMOSAT/tamnnm"
alias dumax="du -sh | sort -h | tail -25"
alias trash="mv -t $work/trashbin"
alias grib_to_netcdf="sh $work/treat_data/grib_to_netcdf.sh"
alias sshow="scontrol show jobid -dd"
alias ssqueue="squeue -o\"%.18i %.9P %.15j %.8u %.1T %.12M %.12l %.6D %.4C\" -S u+j"
alias agit="sh ~/gitsync/auto_commit.sh"
alias cgit="sh ~/gitsync/custom_commit.sh"
alias hncdump="ncdump -h"
alias usqueue="squeue --user tamnnm -o '%.18i %.5t %.15j %.5C %.6D %.20N %.15P %.10r %.19S %.10M'"
alias pdscancel="scancel --user tamnnm --state=PENDING"
alias rscancel="scancel --user tamnnm --state=RUNNING"
alias ascancel="scancel --user tamnnm"
alias ups="ps ux -U tamnnm"
alias cpa="cp -a"
alias cpp="cp -P"
alias squeue="squeue -o '%.18i %.9u %.5t %.15j %.5C %.6D %.20N %.15P %.10r %.19S %.10M'"
alias run_real="sh $wrf_run/real.sh"

# IF you want to show to username: \u
# export PS1="\[$(tput bold)$(tput setaf 2)\]\u@\h:\[$(tput setaf 4)\]\W\[$(tput sgr0)\]\$ "

# Current terminal prompt
export PS1="\[$(tput bold)$(tput setaf 2)\]\h: \[$(tput setaf 4)\]\W\[$(tput sgr0)\]\$ "
  alias ls='ls --color'
  alias dir='dir --color'
  alias vdir='vdir --color'
  alias grep='grep --color'
  alias fgrep='fgrep --color'
  alias egrep='egrep --color'

alias mcw_cat="export VSCODE_IPC_HOOK_CLI=$(cat ~/.vscode_ipc_hook_cli) && ulimit -s unlimited"

## >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/work/apps/gnu_4.8.5/anaconda3/2021.11/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
 if [ -f "/work/apps/gnu_4.8.5/anaconda3/2021.11/etc/profile.d/conda.sh" ]; then
  . "/work/apps/gnu_4.8.5/anaconda3/2021.11/etc/profile.d/conda.sh"  # commented out by conda initialize
 else
   export PATH="/work/apps/gnu_4.8.5/anaconda3/2021.11/bin:$PATH"  # commented out by conda initialize
 fi
fi

unset __conda_setup

# <<< conda initialize <<<
conda config --set auto_activate_base False

# Prevent redundant activation of the 'tamnnm' environment
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "tamnnm" ]; then
 conda deactivate
 conda activate tamnnm
fi

tmux_init() {
    # Skip if already in tmux or not in VSCode
    if [[ -z "$VSCODE_IPC_HOOK_CLI" ]]; then
        return
    fi

    # Save VSCode IPC hook for recovery
    echo "$VSCODE_IPC_HOOK_CLI" > ~/.vscode_ipc_hook_cli
    sync

    local session_name="${PWD##*/}"

    # Create session if it doesn't exist
    if ! tmux has-session -t "$session_name" 2>/dev/null; then
        tmux new-session -s "$session_name"
    fi

    # Update ALL panes (existing + new) with VSCODE_IPC_HOOK_CLI
    tmux list-panes -t "$session_name" -F '#{pane_id}' | while read -r pane_id; do
        # 1. Export VSCODE_IPC_HOOK_CLI (always update)
        tmux send-keys -t "$pane_id" "export VSCODE_IPC_HOOK_CLI='$VSCODE_IPC_HOOK_CLI'" C-m
        tmux send-keys -t "$pane_id" "clear" C-m

    done

    # Attach if not already in tmux
    [[ -z "$TMUX" ]] && tmux attach -dt "$session_name"
}

# Check if the shell is non-login and interactive
if [[ $- == *i* ]] && [[ -z "$LOGIN_SHELL" ]]; then
    # Only run auto_commit.sh if NOT in tmux
    if [[ -z "$TMUX" ]]; then
        conohup bash -c "sh ~/gitsync/auto_commit.sh"

    fi

    # Extract the value of VSCODE_IPC_HOOK_CLI if it exists
    if [[ -z "$TMUX" ]] && [[ -n "$VSCODE_IPC_HOOK_CLI" ]]; then
        tmux_init  # Your existing function
    fi
fi


cd .
ulimit -s unlimited
