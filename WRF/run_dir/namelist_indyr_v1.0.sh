#!/bin/bash
#SBATCH --job-name=submit_namelist
#SBATCH --output=submit_namelist.out
#SBATCH --error=submit_namelist.err
#SBATCH --time=10-00:00:00
#SBATCH --nodes=1
#SBATCH --exclusive

source /home/tamnnm/load_func_vars.sh
source /home/tamnnm/.bashrc

# mp_scheme=( 6 8 )
# bl_scheme=( 1 2 5 )
# sf_scheme=( 1 2 5 )
# cu_scheme=( 1 16 )
# ra_scheme=( 4 24 )

mp_scheme=( 6 8 )
bl_scheme=( 1 2 5 )
sf_scheme=( 1 2 91 5 )
cu_scheme=( 1 16 3 )


remove_dot_slash_prefix() {
 local folder=$1
 if [[ $folder == ./* ]]; then
   echo "${folder:2}"
 else
   echo "$folder"
 fi
}

year_month=$1
dataset=$2

wps_path="${wps}/WPS_${dataset}_${year_month}"
root_folder="${dataset}_${year_month}"
prod_folder="${prod_wrf}/wrf_${year_month}"


year_month_day(){
  local infile="$1"
  local datetime=$(echo "$infile" | cut -d'.' -f3)
  local year=$(echo "$datetime" | cut -d'-' -f1)
  local month=$(echo "$datetime" | cut -d'-' -f2)
  local day=$(echo "$datetime" | cut -d'-' -f3 | cut -d'_' -f1)
  local hour=$(echo "$datetime" | cut -d'_' -f2 | cut -d':' -f1)
  echo "$datetime $year $month $day $hour"
}

create_root_folder() {
  last_infile=$(ls "$wps_path" | grep "^met_em.d01" | sort | tail -n 1)
  first_infile=$(ls "$wps_path" | grep "^met_em.d01" | sort | head -n 1)
  re_infile=$(echo "$last_infile" | sed 's/_00:/_01:/')

  local start_time start_year start_month start_day start_hour
  read start_time start_year start_month start_day start_hour <<< "$(year_month_day ${first_infile})"
  read end_time end_year end_month end_day end_hour <<< "$(year_month_day ${last_infile})"

  echo $start_year
  if [ ! -d ${root_folder} ]; then
    cp -r "MODEL/" "${root_folder}"
  fi

  if [ ! -d "${prod_folder}" ]; then
    mkdir -p "${prod_folder}"
  fi


  cd "${root_folder}"

  # rm -f met_em*
  # rm -f rsl*

  # rm -f wrfbdy*
  # rm -f wrfinput*

  # ln -rsf $wps_path/met_em* .

  check_line_input "start_year" "$start_year"
  check_line_input "start_month" "$start_month"
  check_line_input "start_day" "$start_day"
  check_line_input "start_hour" "$start_hour"

  check_line_input "end_year" "$end_year"
  check_line_input "end_month" "$end_month"
  check_line_input "end_day" "$end_day"
  check_line_input "end_hour" "$end_hour"

  echo $start_year $start_month $start_day $start_hour
  formatted_namelist "namelist.input"
  cd ..
}

create_sub_folder(){
  local number=$1
  local mp_scheme=$2
  local bl_scheme=$3
  local sf_scheme=$4
  local cu_scheme=$5
  local ra_scheme=$6

  sub_folder="${root_folder}_${number}"

  if [ ! -d "${sub_folder}" ]; then
    cp -r "${root_folder}" "${sub_folder}"
  else
    cp "${root_folder}/namelist.input" "${sub_folder}/namelist.input"
  fi

  if [ ! -d "${prod_folder}/${sub_folder}" ]; then
    mkdir ${prod_folder}/${sub_folder}
  fi

  cd "${sub_folder}"

  # rm -f met_em*
  # ln -rsf $wps_path/met_em* .

  check_line_input "history_outname" "${prod_folder}/${sub_folder}/wrfout_d<domain>_<date>"
  check_line_input "rst_outname" "${prod_folder}/${sub_folder}/wrfrst_d<domain>_<date>"

  check_line_input "mp_physics" "${mp_scheme}"
  check_line_input "bl_pbl_physics" "${bl_scheme}"
  check_line_input "sf_sfclay_physics" "${sf_scheme}"
  check_line_input "cu_physics" "${cu_scheme}"
  check_line_input "ra_lw_physics" "${ra_scheme}"
  check_line_input "ra_sw_physics" "${ra_scheme}"

  if [[ $cu_scheme != 1 ]]; then
    check_line_input "cu_rad_feedback" ".false."
  fi

  formatted_namelist "namelist.input"
  
  sbatch ../run_dir/run_real.sh

  cp namelist.input "${prod_folder}/${sub_folder}/namelist.input"

  cd ..
}

# Initialize the job number
j=0

# Create the root folder

cd $wrf_test
#create_root_folder

# Loop through the schemes
for mp in "${mp_scheme[@]}"; do
  for bl in "${bl_scheme[@]}"; do
    for sf in "${sf_scheme[@]}"; do
      for cu in "${cu_scheme[@]}"; do
          if [ "$j" -lt 10 ]; then
            no_job="0$j"
          else
            no_job="$j"
          fi
          
          if
          { [ "$bl" -eq 1 ] && [ "$sf" -ne 1 ] && [ "$sf" -ne 91 ]; } || { [ "$bl" -eq 2 ] && [ "$sf" -ne 2 ]; }; then
            continue
          fi
          
          echo $ra
          echo "${no_job}"
          create_sub_folder "${no_job}" "${mp}" "${bl}" "${sf}" "${cu}" "${ra}"
          j=$((j+1))
        done
      done
    done
  done
done