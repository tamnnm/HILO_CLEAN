#!/bin/bash
#SBATCH -J wrf_tamnnm_root
#SBATCH --time=10:00:00
#SBATCH --output=report/submit_wps.out
#SBATCH --error=report/submit_wps.err
#SBATCH --nodes=1
#SBATCH --partition=scalable
#SBATCH --exclusive

usage() {
  echo "Usage: $0 -dataset <dataset_name> -ym <year_month>"
  exit 1
}

option = 1
grib_option = true

geo_option = true
ungrib_option = true
metgrid_option = true

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
        -h|--help)
            usage
            ;;
        -option)
            option="$2"
            shift 2
            ;;
        -grib_option)
            grib_option="$2"
            shift 2
            ;;
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

if [ $option -eq 1 ]; then
    geo_option = true
    ungrib_option = true
    metgrid_option = true
elif [ $option -eq 2 ]; then
    geo_option = false
    ungrib_option = true
    metgrid_option = true
elif [ $option -eq 3 ]; then
    geo_option = false
    ungrib_option = false
    metgrid_option = true
fi

folder="WPS_ERA5_194512"
ulimit -s unlimited
echo "Start $folder"
cd $folder
rm nohup.out
rm geo_em*
wait
if ./geogrid.exe; then
 echo "geogrid.exe success"
else
 echo "geogrid.exe failed"
fi
 wait
if ./ungrib.exe; then
 echo "ungrib.exe success"
else
 echo "ungrib.exe failed"
fi
wait

if ./metgrid.exe; then
 echo "metgrid.exe success"
else
 echo "metgrid.exe failed"
fi
cd ..