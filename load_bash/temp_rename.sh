#!/bin/bash
# Navigate to the directory containing the files

for folder in ./NOAA_*; do
cd $folder
# Find and rename files
find . -type f -name '*00:*' | while read -r file; do
   new_name="${file}00:00"
   mv "$file" "$new_name"
done
cd ..
done
