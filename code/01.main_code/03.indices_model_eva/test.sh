#!/bin/bash

pending_nodes=$(squeue -u tamnnm -h -t PENDING -o "%i" | \
            while read jobid; do
                scontrol show jobs $jobid | grep "ReqNodeList" | cut -d' ' -f1 | cut -d'=' -f2
            done | tr ',' '\n' | sed 's/compute-[0-9]-//g' | \
            # Expand ranges like [12-14] to individual numbers
            sed 's/\[\([0-9]*\)-\([0-9]*\)\]/\1 \2/g' | \
            while read line; do
                if [[ $line =~ ([0-9]+)[[:space:]]([0-9]+) ]]; then
                    # If it's a range, expand it
                    seq ${BASH_REMATCH[1]} ${BASH_REMATCH[2]}
                else
                    # If it's a single number or list
                    echo $line | tr ' ' '\n'
                fi
            done | sort -n | uniq)
        
echo $pending_node
