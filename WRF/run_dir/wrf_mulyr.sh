#!/bin/bash
#SBATCH -J wrf_submit
#SBATCH --time=10-00:00:00
#SBATCH --nodes=1
#SBATCH --partition=broadwell
#SBATCH -o submit_wrf.txt

usage() {
  echo "Usage: $0 -dataset <dataset>"
  echo "  dataset: ERA5, ERA, NOAA"
  exit 1
}

echo "List of years and months to run WRF for"
cat years_months.txt
echo "Do you want to want to change? (y/n)"
read answer
if [ "$answer" == "y" ]; then
  nano years_months.txt
elif [ "$answer" == "n" ]; then
  echo "Proceeding with the current list"
else
  echo "Invalid input"
  exit 1
fi

while [ "$1" != "" ]; do
  case $1 in
    -dataset )
        shift
        dataset=$1
        ;;
    * )
        usage
  esac
  shift
done

# Read the file
file="years_months.txt"

# Extract the list of years and months
years=$(grep "^years:" "$file" | cut -d' ' -f2-)
months=$(grep "^months:" "$file" | cut -d' ' -f2-)

# Convert the lists to arrays
years_array=($years)
months_array=($months)
dataset=$1

for ((i=0; i<${#year[@]}; i++)); do
<<<<<<< HEAD
  sbatch -J "wrf_${years_array[$i]}" wrf_indyr.sh -ym "${years_array[$i]}${months_array[$i]}" -dataset ${dataset}
=======
  wait 200
  #sbatch -J "wrf_${years_array[$i]}" wrf_indyr.sh -ym "${years_array[$i]}${months_array[$i]}" -dataset ${dataset}
>>>>>>> c80f4457 (First commit)
done
