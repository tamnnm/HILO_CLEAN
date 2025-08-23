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
    result = subprocess.run(
        [f"sh time_extract.sh {folder}"],
        capture_output=True,
        shell=True,
        text=True,
        bufsize=1
    )
    
    if result.returncode != 0:
        raise Exception(f"Error running command: {result.stderr}")
 
    try:
        # Return the single value
        return float(result.stdout.strip())
    except ValueError:
        raise ValueError(f"Invalid output: {result.stdout}")

def ext_benchmark(dataset, yyyymm, model = 'wrf'):
    prod_path = os.getenv("prod_wrf")
    big_folder = os.path.join(prod_path,f'{model}_{yyyymm}')
    for folder in os.listdir(big_folder):
        print("Starting WRF benchmark...")
        return_code = sh_gettime(big_folder+'/'+folder)
        print(folder)
        print(return_code)

ext_benchmark('NOAA',188110)

# # Read the data
# data = pd.read_csv('benchmark_data.txt', sep=' ', names=['Config', 'NCPUs', 'AvgTime'])

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