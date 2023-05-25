import glob
import os
import random
import rasterio
import numpy as np
from sklearn.model_selection import train_test_split

from dataset.shortest_path_algos import dijkstra

def retrieve_tif():

    '''
    Input : no input
    Output : list of 
    '''
    
    ''' Get the data '''
    # Get the files
    directory = 'dataset/tif/'
    files = os.listdir(directory)
    tif_files = [os.path.join(directory, f) for f in files if f.endswith('.tif')]

    # Create a list to store the 100x100 grids
    grids = []
    
    for file in tif_files:

        # Open the GeoTIFF file
        with rasterio.open(file) as dataset:

            # Read the raster data into a NumPy array
            raster = dataset.read(1)

            # Convert the data type to float and set nodata values to NaN
            raster = raster.astype(float)
            raster[raster == dataset.nodata] = np.nan

        ''' Cut the data'''

        # Create a random 3601x3601 grid for testing purposes
        original_grid = np.array(raster)

        # Determine the number of 100x100 grids that can fit within the original grid
        num_horizontal_grids = original_grid.shape[1] // 100
        num_vertical_grids = original_grid.shape[0] // 100        

        # Loop through each horizontal and vertical grid
        for i in range(num_vertical_grids):
            for j in range(num_horizontal_grids):
                # Determine the indices for the current 100x100 grid
                start_row = i * 100
                end_row = start_row + 100
                start_col = j * 100
                end_col = start_col + 100
                
                # Create a new 100x100 grid and copy the values from the original grid
                new_grid = np.zeros((100, 100))
                new_grid[:,:] = original_grid[start_row:end_row, start_col:end_col]
                
                # Normalize the new data
                new_grid_norm = (new_grid - np.nanmin(new_grid)) / (np.nanmax(new_grid) - np.nanmin(new_grid))

                # Add the new grid to the list of grids
                grids.append(new_grid_norm)    

    print(f"Max values : {len(grids)}")        

    return grids

import concurrent.futures
from tqdm import tqdm

def generate_labels(grids):
    label_grids = []

    def process_grid(grid):
        # Define the start and end points
        max_range = grid.shape[0]
        # start = (random.randint(0, max_range-1), random.randint(0, max_range-1))
        # end = (random.randint(0, max_range-1), random.randint(0, max_range-1))
        start = (25, 25)
        end = (75, 75)
        
        # Run algorithm
        path_dijkstra = dijkstra(grid, start, end)

        # Generate the label
        label_grid = np.zeros((max_range, max_range))
        for cell in path_dijkstra:
            x, y = cell
            label_grid[y][x] = 0.5

        return label_grid

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Process each grid concurrently
        futures = [executor.submit(process_grid, grid) for grid in grids]

        # Create a progress bar
        progress_bar = tqdm(total=len(futures))

        # Retrieve the results
        for future in concurrent.futures.as_completed(futures):
            label_grid = future.result()
            label_grids.append(label_grid)

            # Update the progress bar
            progress_bar.update(1)

        # Close the progress bar
        progress_bar.close()

    return label_grids

def generate_tif_dataset(max_values = 0):

    print("Retrieve the data")
    grids = retrieve_tif()

    if max_values > 10 :
        grids = grids[:max_values]

    print("Generate the path")
    labels = generate_labels(grids)

    print(len(grids))
    print(len(labels))

    X_train, X_test, y_train, y_test = train_test_split(grids, labels, test_size=0.2, random_state=42)

    X_train = np.stack(X_train)
    y_train = np.stack(y_train)
    X_test = np.stack(X_test)
    y_test = np.stack(y_test)

    return X_train, X_test, y_train, y_test