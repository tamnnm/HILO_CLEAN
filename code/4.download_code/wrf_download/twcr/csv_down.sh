#!/bin/bash

# Set the delimiter to comma
IFS=','

filename=$work/d

# Read the CSV file and loop over each line
while read url filename
do
    # Download the file using wget
    wget "$url" -O "$filename"
done < urls.csv

