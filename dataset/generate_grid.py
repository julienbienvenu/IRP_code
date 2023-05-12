import rasterio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Open the GeoTIFF file
with rasterio.open("dataset/tif/n47_e001_1arc_v3.tif") as dataset:

    # Read the raster data into a NumPy array
    raster = dataset.read(1)

    # Convert the data type to float and set nodata values to NaN
    raster = raster.astype(float)
    raster[raster == dataset.nodata] = np.nan



# Get a 10% sample of the array
sampled_raster = raster

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Get the x, y, and z coordinates
x, y = np.meshgrid(np.arange(sampled_raster.shape[1]), np.arange(sampled_raster.shape[0]))
z = sampled_raster

# Plot the surface
ax.plot_surface(x, y, z)

# Show the plot
plt.show()
