#!/bin/bash

FOLDER_DIR="$1"
DAILY_DIR="$2"
DAILY_UNMERGE_DIR="$3"

for file in "$FOLDER_DIR"/*.nc; do
    output_file="$DAILY_DIR/$(basename "$file" .nc).nc"
    unmerge_output_file="$DAILY_UNMERGE_DIR/$(basename "$file" .nc).nc"
    if ! file_exists "$output_file" && ! file_exists "$unmerge_output_file"; then
        echo "Creating daily sum for $file"
        cdo daysum "$file" "$output_file" &
        fi
done

# nums is the number of variable
nums=($(find "$FOLDER_DIR" -type f -name "${dataset}_*_*.nc" | \
grep -oP "${dataset}_\K[0-9]+(?=_)" | \
sort -n | uniq))

for num in "${nums[@]}"; do
    if [ -z "$(ls -A "$DAILY_UNMERGE_DIR/*${num}*.nc" 2>/dev/null)" ]; then
        DAILY_MERGE_DIR=$DAILY_DIR
    else
        DAILY_MERGE_DIR=$DAILY_UNMERGE_DIR
    fi

    if [[ "$num" == 167 ]]; then
        if ! file_exists "$DAILY_DIR/min_${dataset}_${num}.nc" || \
            ! file_exists "$DAILY_DIR/max_${dataset}_${num}.nc" || \
            ! file_exists "$DAILY_DIR/mean_${dataset}_${num}.nc"; then

            types=("min" "max" "mean")
            for type in "${types[@]}"; do
                (
                if [[ -f temp_${num}_${type}.nc ]]; then
                    rm temp_${num}_${type}.nc  # Remove any existing temp.nc to start fresh
                fi
                # Process files in batches of 4
                batch=()
                counter=0
                for file in "$DAILY_MERGE_DIR/${type}_${dataset}_${num}_*.nc"; do
                    if [[ $file == *_unfixed.nc ]]; then
                        continue
                    fi
                    batch+=("$file")
                    echo $file
                    ((counter++))
                    if ((counter == 4)); then
                        if [[ -f temp.nc ]]; then
                            # Merge the current batch with temp.nc
                            cdo mergetime temp_${num}_${type}.nc "${batch[@]}" temp_merged_${num}_${type}.nc
                            mv temp_merged_${num}_${type}.nc temp_${num}_${type}.nc
                        else
                            # Merge the first batch into temp.nc
                            cdo mergetime "${batch[@]}" temp_${num}_${type}.nc
                        fi
                        batch=()  # Reset the batch
                        counter=0
                    fi
                done
                # Process remaining files in the batch
                if ((counter > 0)); then
                    if [[ -f temp.nc ]]; then
                        # Merge the remaining files with temp.nc
                        cdo mergetime temp_${num}_${type}.nc "${batch[@]}" temp_merged_${num}_${type}.nc
                        mv temp_merged_${num}_${type}.nc temp_${num}_${type}.nc
                    else
                        # Merge the remaining files into temp.nc
                        cdo mergetime "${batch[@]}" temp_${num}_${type}.nc
                    fi
                fi

                if [[ $DAILY_MERGE_DIR == $DAILY_DIR ]]; then
                    mv $DAILY_DIR/${type}_${dataset}_${num}_*.nc "$DAILY_DIR/unmerge/"
                fi
                # Turn K into C
                cdo subc,273.15 temp_${num}_${type}.nc "$DAILY_DIR/${type}_${dataset}_${num}.nc"
                ) &
            done
            wait
        else
            cdo subc,273.15 "$DAILY_DIR/min_${dataset}_${num}.nc" "$DAILY_DIR/min_${dataset}_${num}.nc"
            cdo subc,273.15 "$DAILY_DIR/max_${dataset}_${num}.nc" "$DAILY_DIR/max_${dataset}_${num}.nc"
            cdo subc,273.15 "$DAILY_DIR/mean_${dataset}_${num}.nc" "$DAILY_DIR/mean_${dataset}_${num}.nc"
        fi

    else
        if ! file_exists "$DAILY_DIR/${dataset}_${num}.nc"; then
            mkdir -p "$DAILY_DIR/unmerge"
            if [[ -f temp.nc ]]; then
                rm temp_${num}.nc  # Remove any existing temp.nc to start fresh
            fi
            (
            # Process files in batches of 4
            batch=()
            counter=0
            for file in "$DAILY_MERGE_DIR/${dataset}_${num}_*.nc"; do
                if [[ $file == *_unfixed.nc ]]; then
                    continue
                fi
                batch+=("$file")
                ((counter++))
                if ((counter == 4)); then
                    if [[ -f temp.nc ]]; then
                        # Merge the current batch with temp.nc
                        cdo mergetime temp.nc "${batch[@]}" temp_merged_${num}.nc
                        mv temp_merged_${num}.nc temp_${num}.nc
                    else
                        # Merge the first batch into temp.nc
                        cdo mergetime "${batch[@]}" temp_${num}.nc
                    fi
                    batch=()  # Reset the batch
                    counter=0
                fi
            done

            # Process remaining files in the batch
            if ((counter > 0)); then
                if [[ -f temp.nc ]]; then
                    # Merge the remaining files with temp.nc
                    cdo mergetime temp_${num}.nc "${batch[@]}" temp_merged_${num}.nc
                    mv temp_merged_${num}.nc temp_${num}.nc
                else
                    # Merge the remaining files into temp.nc
                    cdo mergetime "${batch[@]}" temp_${num}.nc
                fi
            fi


            if [[ $DAILY_MERGE_DIR == $DAILY_DIR ]]; then
                mv $DAILY_DIR/${dataset}_${num}_*.nc "$DAILY_DIR/unmerge"
            fi

                # Check if filename contains "228" AND dataset is era or era5
            if [[ "$num" == 228 && ("$dataset" == "era" || "$dataset" == "era5") ]]; then
                cdo mulc,1000 temp_${num}.nc "temp.nc"
                mv temp.nc "$DAILY_DIR/${dataset}_${num}.nc"
            else
                mv temp_${num}.nc "$DAILY_DIR/${dataset}_${num}.nc"
            fi
            ) &
        elif [[ "$num" == 228 && ("$dataset" == "era" || "$dataset" == "era5") ]]; then
                #? Shift time already multiplied by 1000
                # cdo mulc,1000 "$DAILY_DIR/${dataset}_${num}.nc" "temp.nc"
                # mv temp.nc "$DAILY_DIR/${dataset}_${num}.nc"
                echo "Skipping daily mean for $(basename "$file") - output already exists"
        fi
    fi
done