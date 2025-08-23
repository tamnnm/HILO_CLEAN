#!/bin/bash

for folder in WPS_ERA5_*; do
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

	if [[ "$folder" != "WPS_ERA5_196411" && "$folder" != "WPS_ERA5_19682" ]]; then

        	if ./ungrib.exe; then
	  		echo "ungrib.exe success"
		else
			echo "ungrib.exe failed"
		fi
		wait
	else
		echo "Skip ungrib ${folder}"
	fi
	wait

	if ./metgrid.exe; then
		echo "metgrid.exe success"
	else
		echo "metgrid.exe failed"
	fi
	cd ..
done
