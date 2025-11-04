# plot_utils.py

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

# --- Helper Function to Generate Random Graphs ---

def generate_random_graph(n, m):
    """
    Generates a random simple undirected graph with n vertices and m edges.
    """
    if n < 0 or m < 0:
        raise ValueError("Number of vertices and edges must be non-negative.")
    max_edges = n * (n - 1) // 2
    if m > max_edges:
        raise ValueError(f"Edge count {m} exceeds the maximum possible of {max_edges}.")

    G = nx.Graph()
    G.add_nodes_from(range(n))

    possible_edges = []
    if n > 1:
        for i in range(n):
            for j in range(i + 1, n):
                possible_edges.append((i, j))
    
    random.shuffle(possible_edges)
    edges_to_add = possible_edges[:m]
    G.add_edges_from(edges_to_add)
    
    return G

# --- Helper Function for Plotting a Single Graph ---

def plot_graph_with_scores(G, pos, scores, title, ax=None):
    """
    Plots a graph on a given matplotlib axes 'ax' or creates a new figure.
    Uses a fixed layout 'pos' to ensure nodes are in the same place.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        show_plot = True
    else:
        show_plot = False

    nodes = sorted(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    labels = {}
    for node in nodes:
        # Added a check for safety
        if node_to_idx.get(node, -1) >= len(scores):
             labels[node] = f"{node}: N/A"
             continue
        score_val = scores[node_to_idx[node]]
        if np.isnan(score_val):
            labels[node] = f"{node}: N/A"
        else:
            labels[node] = f"{node}: {score_val:.3f}"
    
    nx.draw(G, pos, ax=ax, with_labels=False, node_color='skyblue', node_size=700, edge_color='gray', width=1.5)
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=10, font_color='black')
    ax.set_title(title, fontsize=14)
    
    if show_plot:
        plt.show()

# --- Helper Function for Text Output ---

def print_score_comparison(nodes, initial, rw, pr, es, am):
    """Prints a formatted table of all scores to the console."""
    print("\n\n" + "="*70)
    print(" " * 25 + "SCORE COMPARISON")
    print("="*70)
    
    header = f"{'Node':<6} | {'Input':<10} | {'RandomWalk':<10} | {'PageRank':<10} | {'EqualSplit':<10} | {'Argmax':<10}"
    print(header)
    print("-" * len(header))
    
    for i, node in enumerate(nodes):
        s_init = f"{initial[i]:.4f}"
        s_rw = f"{rw[i]:.4f}"
        s_pr = f"{pr[i]:.4f}"
        
        s_es = f"{es[i]:.4f}" if not np.isnan(es[i]) else "N/A"
        s_am = f"{am[i]:.4f}" if not np.isnan(am[i]) else "N/A"
        
        print(f"{node:<6} | {s_init:<10} | {s_rw:<10} | {s_pr:<10} | {s_es:<10} | {s_am:<10}")
        
    print("-" * len(header))
    sum_init = np.sum(initial)
    sum_rw = np.sum(rw)
    sum_pr = np.sum(pr)
    sum_es = np.nansum(es)
    sum_am = np.nansum(am)

    s_sum_es = f"{sum_es:<10.2f}" if sum_es > 0 else "N/A       "
    s_sum_am = f"{sum_am:<10.2f}" if sum_am > 0 else "N/A       "

    print(f"{'SUM':<6} | {sum_init:<10.2f} | {sum_rw:<10.2f} | {sum_pr:<10.2f} | {s_sum_es} | {s_sum_am}")
    print("=" * 70 + "\n")

# --- Helper Function for Bar Plot ---

def plot_score_barplot(nodes, initial, rw, pr, es, am):
    """Plots a grouped bar chart comparing all scores for each node."""
    print("Generating score comparison bar plot...")
    
    n_nodes = len(nodes)
    
    algos_data = {
        'Input': initial,
        'RandomWalk': rw,
        'PageRank': pr
    }
    if not np.isnan(es[0]):
        algos_data['EqualSplit'] = es
    if not np.isnan(am[0]):
        algos_data['Argmax'] = am
        
    algo_names = list(algos_data.keys())
    n_algos_to_plot = len(algo_names)

    x = np.arange(n_nodes)
    width = 0.8 / n_algos_to_plot
    fig, ax = plt.subplots(figsize=(max(12, n_nodes * 1.5), 7))
    
    offsets = np.linspace(-width * (n_algos_to_plot - 1) / 2, 
                           width * (n_algos_to_plot - 1) / 2, 
                           n_algos_to_plot)

    for i, (name, data) in enumerate(algos_data.items()):
        offset = offsets[i]
        rects = ax.bar(x + offset, data, width, label=name)
        ax.bar_label(rects, padding=3, fmt='%.3f', rotation=90, fontsize=8)

    ax.set_ylabel('Scores')
    ax.set_title('Score Comparison by Node and Algorithm')
    ax.set_xticks(x)
    ax.set_xticklabels(nodes)
    ax.set_xlabel('Node ID')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    all_scores = np.concatenate(list(algos_data.values()))
    ax.set_ylim(bottom=0, top=np.nanmax(all_scores) * 1.25)

    fig.tight_layout()
    plt.show()
