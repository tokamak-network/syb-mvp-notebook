import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from contract_interface_mvp import VouchMinimal, DEFAULT_RANK
from utils.utils import generate_alphabetical_names

# ------------------ 1. Setup (User's Script) ------------------

# Create the Custom Graph (Scale-Free)
scale_free_graph = nx.barabasi_albert_graph(20, 5)
directed_network = scale_free_graph.to_directed()

# Initialize the contract
contract = VouchMinimal(directed_network)

# Setup User Metadata
users = {}
names = generate_alphabetical_names(len(contract.nodes))

for addr in contract.nodes:
    idx = contract.address_to_idx[addr]
    name = names[idx]
    users[addr] = {'name': name, 'address': addr}

# Create address_to_name map for easy lookup
address_to_name = {data['address']: data['name'] for data in users.values()}

# ------------------ 2. Console Printout ------------------

# Collect data for console and plotting
user_data = []

# Sort by score descending for readability
sorted_addresses = sorted(users.keys(), key=lambda addr: contract.get_score(addr), reverse=True)

for addr in sorted_addresses:
    name = address_to_name[addr]
    score = contract.get_score(addr)
    rank = contract.get_rank(addr)
    
    user_data.append({
        'User': name,
        'Address': addr,
        'Rank': 'DEFAULT' if rank >= DEFAULT_RANK else str(rank),
        'Raw Rank': rank,
        'Score': score
    })

df = pd.DataFrame(user_data)

print("\n--- Final Ranks and Scores (Sorted by Score) ---")
print(df[['User', 'Rank', 'Score', 'Address']].to_string(index=False))

# ------------------ 3. Plotting the Graph ------------------

# Prepare data for NetworkX drawing
# Need to map scores/ranks/names back to NetworkX integer nodes (0-14)
graph = contract.network
pos = nx.spring_layout(graph, seed=42) # Fixed seed for reproducibility

node_scores = [df[df['Address'] == contract.idx_to_address.get(node_idx)]['Score'].iloc[0] for node_idx in graph.nodes()]
node_ranks = [df[df['Address'] == contract.idx_to_address.get(node_idx)]['Raw Rank'].iloc[0] for node_idx in graph.nodes()]
labels = {node_idx: address_to_name[contract.idx_to_address.get(node_idx)] for node_idx in graph.nodes()}

# Calculate node sizes based on scores
scores_array = np.array(node_scores)
min_score = scores_array.min()
max_score = scores_array.max()

min_size = 500
max_size = 2000

if max_score > min_score:
    normalized_scores = (scores_array - min_score) / (max_score - min_score)
    node_sizes = min_size + normalized_scores * (max_size - min_size)
else:
    node_sizes = [min_size + (max_size - min_size) / 2] * len(scores_array)
    
# Determine node colors based on Rank (lower rank is better/different color)
colors = []
non_default_ranks = [r for r in node_ranks if r < DEFAULT_RANK]

if non_default_ranks:
    min_r = min(non_default_ranks)
    max_r = max(non_default_ranks)
    # Use 'plasma' colormap, reversing normalization so lower rank (better) gets a more distinct color
    cmap = plt.colormaps.get_cmap('plasma')
else:
    # Fallback if all nodes are DEFAULT
    cmap = plt.colormaps.get_cmap('plasma')


for rank in node_ranks:
    if rank >= DEFAULT_RANK:
        colors.append('#cccccc') # Light grey for DEFAULT
    else:
        if max_r > min_r:
            normalized_rank = (rank - min_r) / (max_r - min_r)
            colors.append(cmap(1.0 - normalized_rank))
        else:
            colors.append(cmap(0.5))

# Create the plot
plt.figure(figsize=(20, 15))

# Draw nodes
nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color=colors, alpha=0.9)

# Draw edges (arrows for directed graph)
nx.draw_networkx_edges(graph, pos, edge_color='gray', arrowsize=20, width=1.5)

# Draw labels (names)
nx.draw_networkx_labels(graph, pos, labels=labels, font_size=10, font_color='black')

plt.title('VouchMinimal Scale-Free Network Status (Size=Score, Color=Rank)', fontsize=16)
plt.axis('off')

# Manually create a simple legend for Rank
legend_handles = []

# Handle DEFAULT rank
legend_handles.append(plt.scatter([], [], s=100, color='#cccccc', label='DEFAULT RANK'))

# Handle unique non-default ranks
for rank in sorted(list(set(non_default_ranks))):
    if max_r > min_r:
        normalized_rank = (rank - min_r) / (max_r - min_r)
        color = cmap(1.0 - normalized_rank)
    else:
        color = cmap(0.5)

    legend_handles.append(plt.scatter([], [], s=100, color=color, label=f'Rank {rank}'))

# Draw a legend for Score (size)
score_legend_points = [min_size, max_size]
score_legend_labels = [f"Score $\\approx$ {scores_array.min():,.0f} (Min)",
                       f"Score $\\approx$ {scores_array.max():,.0f} (Max)"]

# Use a common color for size legend handles (e.g., grey)
score_legend_color = '#999999'

# Scale the size for the legend marker appearance
size_scale = 1/5 

for i, size in enumerate(score_legend_points):
    legend_handles.append(plt.scatter([], [], s=size*size_scale, color=score_legend_color, edgecolors='k', label=score_legend_labels[i]))
    
plt.legend(handles=legend_handles, title="Node Color = Rank (Lower is Better)\nNode Size = Score (Higher is Better)", loc='upper left', bbox_to_anchor=(0.0, 1.0))

# Save the plot
plt.savefig("scale_free_network_status.png")
print("\nGraph saved to scale_free_network_status.png")