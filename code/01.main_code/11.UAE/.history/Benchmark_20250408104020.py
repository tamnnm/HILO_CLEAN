import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import subprocess
import sys

def sh_gettime(folder: str) -> int:
    """
    Run a bash command and show output in real-time
    
    Parameters
    ----------
    command : str
        Command to execute
        
    Returns
    -------
    int
        Return code from the command
    """
    process = subprocess.run(
        [f"sh time_extract.sh {folder}"],
        capture_output=True,
        shell=True,
        text=True,
        bufsize=1
    )
    
    # Process output in real-time
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
            sys.stdout.flush()
    return process.poll()

def process_benchmark_results(case_dir: str) -> None:
    """Process the benchmark results"""
    # Read the output files
    try:
        with open(f"{case_dir}/rsl.out.0000", 'r') as f:
            output = f.read()
            # Extract timing information
            # Process results
            # Save to DataFrame
            pass
    except FileNotFoundError:
        print(f"Could not find output file in {case_dir}")

def ext_benchmark(dataset, yyyymm, model = 'wrf'):
    prod_path = os.getenv("wrf_prod")
    for folder in os.path.join(os.listdir(prod_path),f'{model}_{yyyymm}'):
        print("Starting WRF benchmark...")
        return_code = run_bash_realtime(f"./run_wrf.sh {case_dir}")
        
        if return_code == 0:
            print("Benchmark completed successfully")
            # Process the results
            process_benchmark_results(case_dir)
        else:
            print(f"Benchmark failed with return code {return_code}")



# Read the data
data = pd.read_csv('benchmark_data.txt', sep=' ', names=['Config', 'NCPUs', 'AvgTime'])

# Create the plot
plt.figure(figsize=(12, 6))

# Bar plot
x = np.arange(len(data['Config']))
plt.bar(x, data['AvgTime'])

# Customize the plot
plt.xlabel('Model Configuration')
plt.ylabel('Average Time (seconds)')
plt.title('WRF Model Performance Benchmark')

# Rotate x-axis labels for better readability
plt.xticks(x, data['Config'], rotation=45)

# Add CPU count as text on top of each bar
for i, (time, cpus) in enumerate(zip(data['AvgTime'], data['NCPUs'])):
    plt.text(i, time, f'{cpus} CPUs', ha='center', va='bottom')

# Add grid for better readability
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('wrf_benchmark.png', dpi=300, bbox_inches='tight')
plt.close()