#!/bin/bash

<<<<<<< HEAD
# Define the source and destination directories
source_dir="/work/users/tamnnm/"
destination_dir="/home/tamnnm/git_code"

# Create the destination directory if it does not exist
mkdir -p "$destination_dir"

# Use rsync to copy all Python and Bash files while preserving the directory structure
rsync -avm --include='*.py' --include='*.sh' --include='*/' --exclude='*' "$source_dir/" "$destination_dir/"

rsync -av --include='*.sh' --include='*.py' --exclude='*' ../ /home/tamnnm/git_code/load_bash/

## Sync the cut_dataset file to monitor its change
rsync -avz /home/tamnnm/.conda/envs/tamnnm/lib/python3.9/site-packages/my_junk/ /home/tamnnm/git_code/private_package/

echo "All Python and Bash files have been copied to $destination_dir while preserving the directory structure."
=======
if [ -n "$COPY_LOAD" ]; then
    return 0  # For sourcing
    # exit 0  # For direct execution
fi

# Define the source and destination directories
SOURCE_DIR="/work/users/tamnnm/"
DES_DIR="/home/tamnnm/gitsync"

# Create the destination directory if it does not exist
mkdir -p "$DES_DIR"

rsync -avm --include='*.py' --include='*.sh' \
    --exclude='*.png' --exclude='*.jpg' --exclude='*.jpeg' \
    --exclude='*.gif' --exclude='*.shp' --exclude='*.bmp' \
    --include='*/' --exclude='*' --exclude='trashbin/wrf/*' \
    "$SOURCE_DIR" "$DES_DIR/"

rsync -av --include='*.sh' --include='*.py' --exclude='*' ../ ${DES_DIR}/load_bash/

rsync -av --include='*.sh' --include='*.py' --exclude='*' /data/projects/REMOSAT/tamnnm/ ${DES_DIR}/data/

## Sync the cut_dataset file to monitor its change
rsync -av --include='*.sh' --include='.*' --include='.tmux*/' --exclude='.*/' --exclude='*' /home/tamnnm/ ${DES_DIR}/startup/

rsync -avz /home/tamnnm/.conda/envs/tamnnm/lib/python3.9/site-packages/my_junk/ ${DES_DIR}/private_package/

echo "All Python and Bash files have been copied to $DES_DIR while preserving the directory structure."

export COPY_LOAD=1
>>>>>>> c80f4457 (First commit)
