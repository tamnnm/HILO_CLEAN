#!/bin/bash

# Define the source and destination directories
source_dir="/work/users/tamnnm/"
destination_dir="/home/tamnnm/git_code"

# Create the destination directory if it does not exist
mkdir -p "$destination_dir"

# Use rsync to copy all Python and Bash files while preserving the directory structure
rsync -avm --include='*.py' --include='*.sh' --include='*/' --exclude='*' "$source_dir/" "$destination_dir/"

echo "All Python and Bash files have been copied to $destination_dir while preserving the directory structure."
