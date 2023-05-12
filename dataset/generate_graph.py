import networkx as nx
import matplotlib.pyplot as plt
import random

# Create a figure to plot the graphs on
fig = plt.figure(figsize=(10, 10))

# Generate four random graphs with 20 to 30 nodes and edges between them
graphs = []
for i in range(4):
    n = random.randint(20, 30)
    m = random.randint(n-1, n*(n-1)//2)
    G = nx.gnm_random_graph(n, m)
    graphs.append(G)

# Plot the graphs on the same figure
for i in range(4):
    ax = fig.add_subplot(2, 2, i+1)
    
    # Draw the graph
    # nx.draw_networkx(graphs[i], pos=nx.spring_layout(graphs[i]), with_labels=False, ax=ax)
    
    # Choose two random nodes
    start = random.choice(list(graphs[i].nodes()))
    end = random.choice(list(graphs[i].nodes()))
    
    # Find the shortest path between the nodes
    shortest_path = nx.shortest_path(graphs[i], start, end)
    
    # Highlight the start and end nodes
    node_colors = ["green" if node == start or node == end else "red" for node in graphs[i].nodes()]
    nx.draw_networkx_nodes(graphs[i], pos=nx.spring_layout(graphs[i]), node_color=node_colors, ax=ax)
    
    # Highlight the edges in the shortest path
    edge_colors = ["red" if (u, v) in zip(shortest_path[:-1], shortest_path[1:]) else "black" for u, v in graphs[i].edges()]
    nx.draw_networkx_edges(graphs[i], pos=nx.spring_layout(graphs[i]), edge_color=edge_colors, ax=ax)
    
    # Add a title to the subplot
    ax.set_title("Graph {}".format(i+1))

# Adjust the spacing between the subplots
fig.tight_layout()

plt.show()
