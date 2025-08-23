#!/bin/bash
files=$1
start=$2
end=$3

# Initialize the output file
output_file="${files}_clim.nc"

# Remove the output file if it already exists
if [[ -f "$output_file" ]]; then
    rm "$output_file"
fi

# Loop over the specified range of files
for i in $(seq $start $end); do
    # Construct the file name
    file_path="${files}.${i}.nc"
    
    # Check if the file exists
    if [[ -f "$file_path" ]]; then
        echo "Processing $file_path"
        
        # Merge the current file into the output file
        if [[ -f "$output_file" ]]; then
            cdo mergetime "$file_path" "$output_file" temp.nc
            mv temp.nc "$output_file"
        else
            cp "$file_path" "$output_file"
        fi
    else
        echo "File $file_path does not exist."
    fi
done

echo "Merged files into $output_file"
