#!/bin/bash

#Create a list of times

time_start=('1968-01-01' '2008-01-01' '2023-03-01' '2016-02-01' '1987-09-01' '1971-07-01' '1964-10-01')

time_end=('1968-03-01' '2008-03-01' '2023-05-01' '2016-04-01' '1988-03-01' '1973-08-01' '1964-12-01')

#Get the length of time_start
len=${#time_start[@]}

#Iterate over and replace time 
for ((i=0; i<len; i++)); do
	year_name=$(echo time_start[i] | cut -d'-' -f1)
	month_name=$(echo time_start[i] | cut -d'-' -f2)
	day_name=$(echo time_start[i] | cut -d'-' -f3)
	
	cp $wrf/WPS $wrf/${folder_name}
	
	sed -i "s/

	
