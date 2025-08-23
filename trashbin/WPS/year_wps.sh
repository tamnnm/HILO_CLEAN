#!/bin/bash

#year=(1943 1945 1964 1968 2016 2022 2023 1972)
#month=(12 12 11 2 3 2 4 )

year=(1945 ) 
month=(12 )
replace_line() {
 local line_number=$1
 local new_part=$2
 local file=$3
 sed -i "${line_number}s/=.*/= '${new_part}','${new_part}', /" "$file"
}

month_format() {
 local mon=$1
 if [[ $mon -lt 10 ]]; then
    mon="0$mon"
 fi
 echo "$mon"
}

create_new_folder() {
 local year=$1
 local month=$2

 if [ -d WPS_ERA5_"$year$month" ]; then
  #rm -r WPS_ERA5_"$year$month"
  break
 else
  cp -r WPS_ERA5/ WPS_ERA5_"$year$month"
  
  wait 
  cd WPS_ERA5_"$year$month"
  rm GRIBFILE.A*
  rm PFILE*
  for file in "$wrfdata/era5_6h_1940_2023/*(${year})*.grib"
   do ./link_grib.csh $file
  done
  if [ $year -gt 1947 ]; then
   if [ $year != 1972 ]; then
    range=1
   else
    range=2
   fi
   month_start=$(month_format (($month + $range)))
   month_end=$(month_format (($month + $range)))
   year_end=${year}
  else
   month_start="09"
   month_end="03"
   year_end=$((year +1))
 fi  
 replace_line 4 "${year}-${month_start}-01_00:00:00" namelist.wps
 replace_line 5 "${year_end}-${month_end}-01_00:00:00" namelist.wps
 sed -i "27s/WPS\/WPS_1971\/geogrid/WPS\/WPS_ERA5\/geogrid/" namelist.wps 
 #sed -i '34s/grib/grb/' namelist.wps 
 cd ..
 fi
 }

for ((i=0; i<${#year[@]};i++)); do
	if ((${year[$i]} != 1972)); then
		create_new_folder "${year[$i]}" "${month[$i]}"
		echo "Finish WPS_ERA_${year[$i]}${month[$i]}"
        else 
		create_new_folder "${year[$i]}" 4
		create_new_folder "${year[$i]}" 5
		echo "Finish WPS_ERA5_${year[$i]}4 and WPS_ERA5_${year[$i]}5"
	fi 
done
