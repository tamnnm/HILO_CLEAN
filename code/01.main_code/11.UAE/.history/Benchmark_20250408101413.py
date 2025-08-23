import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def ext_benchmark(dataset, model = 'wrf'):
    prod_path = os.getenv("wrf_prod")
        



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