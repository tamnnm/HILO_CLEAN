mport matplotlib.pyplot as plt
import numpy as np

# Sample data: x, y, and z values
x = np.random.rand(100)
y = np.random.rand(100)
z = np.random.randint(0, 101, 100)  # Random integers between 0 and 100 for z values

# Create a scatter plot with colorbar and custom ticks
plt.scatter(x, y, c=z, cmap='viridis', s=50, edgecolors='k', vmin=0, vmax=100)

# Define the custom colorbar ticks
colorbar_ticks = [0, 20, 40, 60, 80, 100]

# Create the colorbar with custom ticks
cbar = plt.colorbar(ticks=colorbar_ticks)
cbar.set_label('Colorbar Label')

# Add labels and title
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Scatter Plot with Custom Colorbar Ticks')

# Show the plot
plt.show()


