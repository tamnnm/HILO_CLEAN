#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 <last_file>"
    echo
    echo "Parameters:"
    echo "  <last_file>  The name of the last file to check for in each folder."
    echo "Use the format YYYYMMDD e.g. 20190101"
    echo
    echo "Options:"
    echo "  --help       Display this help message and exit."
    echo
    echo "Description: Quick check for folder needs to rerun or restart."
    exit 1
}

# Check for --help option
if [[ "$1" == "--help" ]]; then
    usage
fi

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    usage
fi

last_file=$1

for folder in ./*; do 
 if [ ! -f "${folder}/wrfout_d02_${1}_00:00:00" ]; then
  if [ $(ls "${folder}" | grep "^wrfout_d02" | wc -l) -lt 2 ]; then
   echo $folder >> re_run.txt
  else
   echo $folder >> re_rst.txt
  fi
 fi
done
