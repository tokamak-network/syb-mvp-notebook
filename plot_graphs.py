# plot_graphs.py

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math
import os

def plot_graph_evolution_with_scores(graphs, scores_list, title_prefix, layout_type="circular", filename=None):
    """
    Plots a sequence of graphs with fixed node positions and captions,
    displaying node labels and scores, with each graph inside a box.
    
    Styling is updated to match the "notebook" style:
    - 'skyblue' nodes
    - Node size is proportional to score
    """
    if not graphs:
        print(f"No graphs to plot for {title_prefix}.")
        return

    num_graphs = len(graphs)
    cols = min(num_graphs, 5)
    rows = math.ceil(num_graphs / cols)
    
    # --- Find all nodes present across all graphs for stable layout ---
    all_nodes = set()
    for G in graphs:
        all_nodes.update(G.nodes())
    
    if not all_nodes:
        print("All graphs are empty.")
        return
    
    # Create a dummy graph with all nodes that ever appear
    dummy_graph = nx.Graph()
    dummy_graph.add_nodes_from(sorted(list(all_nodes)))
    
    if layout_type == "circular":
        pos = nx.circular_layout(dummy_graph)
    elif layout_type == "spring":
        # Use a fixed seed for reproducible layout
        pos = nx.spring_layout(dummy_graph, seed=42, k=0.5, iterations=50)
    elif layout_type == "random":
        pos = nx.random_layout(dummy_graph, seed=42)
    elif layout_type == "spectral": # Added spectral layout
        pos = nx.spectral_layout(dummy_graph)
    else:
        # Default to circular layout if an unknown type is provided
        pos = nx.circular_layout(dummy_graph)
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if num_graphs == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    is_random_evolution = "Random Evolution" in title_prefix

    for i, G in enumerate(graphs):
        ax = axes[i]
        
        # Get nodes *actually in this graph*
        nodes_in_g = sorted(list(G.nodes()))
        
        # Get scores for the current step
        current_scores_dict = scores_list[i] if i < len(scores_list) else {}
        
        # --- Start: Styling Logic (like notebook) ---
        
        # 1. Get scores for current nodes
        scores_for_nodes = [current_scores_dict.get(node, 0.0) for node in nodes_in_g]

        # 2. Create Labels (Node: Score)
        labels = {}
        for node in nodes_in_g:
            score = current_scores_dict.get(node, 0.0)
            # Format large scores scientifically to avoid overlap
            if score > 1_000_000:
                labels[node] = f"{node}: {score:.2e}"
            else:
                labels[node] = f"{node}: {score:,.0f}"

        # 3. Calculate Node Sizes based on scores
        node_sizes = []
        min_size = 300
        max_size = 1500
        
        if scores_for_nodes and max(scores_for_nodes) > 0:
            min_s = min(scores_for_nodes)
            max_s = max(scores_for_nodes)
            
            if max_s > min_s:
                for score in scores_for_nodes:
                    # Normalize score (0 to 1)
                    normalized = (score - min_s) / (max_s - min_s)
                    # Map to size range
                    node_sizes.append(min_size + normalized * (max_size - min_size))
            else:
                # All nodes have same score
                node_sizes = [min_size + 0.5 * (max_size - min_size)] * len(nodes_in_g)
        else:
            # All nodes have 0 score
            node_sizes = [min_size] * len(nodes_in_g)

        # 4. Set Node Color
        node_colors = 'skyblue'
        
        # --- End: Styling Logic ---
        
        # Get caption for the evolution
        caption = ""
        if is_random_evolution:
            caption = f"Iteration {i}"
        else:
            if i == 0:
                caption = "Initial Graph"
            else:
                prev_edges = set(graphs[i-1].edges())
                current_edges = set(G.edges())
                added_edges = list(current_edges - prev_edges)
                removed_edges = list(prev_edges - current_edges)
                
                # Check for node changes
                prev_nodes = set(graphs[i-1].nodes())
                current_nodes = set(G.nodes())
                added_nodes = list(current_nodes - prev_nodes)

                if added_nodes:
                     caption = f"Added node {added_nodes[0]}"
                elif added_edges:
                    caption = f"Added edge {tuple(sorted(added_edges[0]))}"
                elif removed_edges:
                    caption = f"Removed edge {tuple(sorted(removed_edges[0]))}"
                else:
                    caption = "No change"
        
        # Draw the graph with fixed layout, node labels, and a single edge color
        nx.draw(
            G, pos, ax=ax, 
            with_labels=False, 
            node_color=node_colors,  # <-- Use new style
            node_size=node_sizes,    # <-- Use new style
            edge_color='gray',
            nodelist=nodes_in_g,     # <-- IMPORTANT: ensure order
            edgelist=list(G.edges()) # Only draw edges in G
        )
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8)
        
        ax.set_title(f"Step {i}")
        ax.text(0.5, -0.15, caption, ha='center', transform=ax.transAxes, fontsize=10)
        
        # Add the delimiting box
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        
    for j in range(num_graphs, len(axes)):
        axes[j].axis('off')

    plt.suptitle(title_prefix, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # --- Add save logic ---
    if filename:
        output_dir = "plots"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Graph evolution plot saved to {filepath}")
        plt.close(fig)
    else:
        plt.show()

def display_graph_state(graph, scores, title):
    """
    Prints the graph's nodes, scores, and neighbors to the console.
    """
    print(f"\n--- {title} ---")
    if not graph.nodes():
        print("Graph is empty.")
        return
        
    for node in sorted(graph.nodes()):
        score = scores.get(node, 0.0) # Use .get() for safety
        neighbors = sorted(list(graph.neighbors(node)))
        # Use .2e for score to match plotting
        print(f"Node {node:<2} | Score: {score:<10.2e} | Neighbors: {neighbors}")
    print("-" * (len(title) + 6))