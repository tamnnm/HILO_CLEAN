#!/bin/bash
#SBATCH -J namelist
#SBATCH --time=10-00:00:00
#SBATCH --nodes=1
#SBATCH --partition=broadwell
#SBATCH -o namelist_wrf.txt

# Read the file
file="years_months.txt"

# Extract the list of years and months
years=$(grep "^years:" "$file" | cut -d' ' -f2-)
months=$(grep "^months:" "$file" | cut -d' ' -f2-)

# Convert the lists to arrays
years_array=($years)
months_array=($months)
dataset=$1

for  ((i=0; i<${#year[@]}; i++)); do
  sbatch -J "namelist_${years_array[$i]}" run_namelist_indyr.sh ${years_array[$i]} ${months_array[$i]} $dataset
done
