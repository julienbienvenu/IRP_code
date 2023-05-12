import heapq
import numpy as np

def get_neighbors(node, height, width):
    
    row, col = node
    neighbors = []
    if row > 0:
        neighbors.append((row-1, col))
    if row < height-1:
        neighbors.append((row+1, col))
    if col > 0:
        neighbors.append((row, col-1))
    if col < width-1:
        neighbors.append((row, col+1))
    return neighbors

def get_weight(node1, node2, grid):
    return abs(grid[node1] - grid[node2])

def dijkstra(grid, start, end):
    # Get the dimensions of the grid
    height, width = grid.shape

    # Initialize the distances to infinity
    distances = np.full((height, width), np.inf)

    # Set the distance to the start node to 0
    distances[start] = 0

    # Initialize the priority queue with the start node
    pq = [(0, start)]

    # Initialize the previous nodes to None
    prev = np.full((height, width), None)

    # While the priority queue is not empty
    while pq:
        # Get the node with the smallest distance
        current_dist, current_node = heapq.heappop(pq)

        # If we have reached the end node, stop
        if current_node == end:
            break

        # For each neighbor of the current node
        for neighbor in get_neighbors(current_node, height, width):
            # Calculate the distance to the neighbor
            neighbor_dist = distances[current_node] + get_weight(current_node, neighbor, grid)

            # If the new distance is shorter than the old distance
            if neighbor_dist < distances[neighbor]:
                # Update the distance and previous node
                distances[neighbor] = neighbor_dist
                prev[neighbor] = current_node

                # Add the neighbor to the priority queue
                heapq.heappush(pq, (neighbor_dist, neighbor))

    # Backtrack from the end node to the start node to get the path
    path = [end]
    current_node = end
    while current_node != start:
        current_node = prev[current_node]
        path.append(current_node)
    path.reverse()

    return path