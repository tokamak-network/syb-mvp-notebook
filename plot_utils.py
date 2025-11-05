# plot_utils.py

from IPython.display import display, HTML
import plotly.graph_objects as go
import networkx as nx
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import random
from contract_interface_mvp import DEFAULT_RANK


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


# Helper Functions for Plotting

# We'll define two helper functions to display the network state at each step, using the same logic as the interactive UI.

def show_network_status(contract, users, ui):
    """Displays the network status table (logic from ui._update_status)."""
    num_users = len(contract.nodes)
    num_edges = contract.network.number_of_edges()
    
    # Get all users' stats
    user_stats = []
    for addr, data in users.items():
        rank = contract.get_rank(addr)
        score = contract.get_score(addr)
        prev_rank = contract.get_previous_rank(addr)
        prev_score = contract.get_previous_score(addr)
        outdeg = contract.get_outdegree(addr)
        indeg = contract.get_in_count(addr)
        
        rank_display = "DEFAULT" if rank >= DEFAULT_RANK else str(rank)
        
        node = contract.nodes.get(addr)
        has_previous_rank = node and node.previous_rank > 0
        if not has_previous_rank:
            prev_rank_display = "-"
        else:
            prev_rank_display = "DEFAULT" if prev_rank >= DEFAULT_RANK else str(prev_rank)
        
        display_name = ui._get_display_name(data['name'])
        user_stats.append({
            'name': display_name,
            'rank': rank_display,
            'prev_rank': prev_rank_display,
            'score': score,
            'prev_score': prev_score,
            'outdeg': outdeg,
            'indeg': indeg
        })
    
    user_stats.sort(key=lambda x: x['name'])

    stats_html = f"""
    <div style='padding: 10px; font-size: 13px;'>
        <div style='padding: 5px;'><strong>👥 Total Users:</strong> {num_users}</div>
        <div style='padding: 5px;'><strong>🔗 Total Vouches:</strong> {num_edges}</div>
        <div style='padding: 10px; margin-top: 10px;'><strong>📈 User Rankings:</strong></div>
        <div style='max-height: 300px; overflow-y: auto;'>
            <table style='width: 100%; border-collapse: collapse; font-size: 11px;'>
                <tr style='background: #f0f0f0;'>
                    <th style='padding: 5px; text-align: left; border: 1px solid #ddd;'>User</th>
                    <th style='padding: 5px; text-align: left; border: 1px solid #ddd;'>Rank</th>
                    <th style='padding: 5px; text-align: left; border: 1px solid #ddd;'>Prev Rank</th>
                    <th style='padding: 5px; text-align: right; border: 1px solid #ddd;'>Score</th>
                    <th style='padding: 5px; text-align: right; border: 1px solid #ddd;'>Prev Score</th>
                    <th style='padding: 5px; text-align: center; border: 1px solid #ddd;'>Out</th>
                    <th style='padding: 5px; text-align: center; border: 1px solid #ddd;'>In</th>
                </tr>
    """
    
    for stat in user_stats:
        rank_changed = stat['rank'] != stat['prev_rank'] and stat['prev_rank'] != "-"
        score_changed = stat['score'] != stat['prev_score'] and stat['prev_score'] != 0
        
        rank_cell_style = "background: #fff3cd;" if rank_changed else ""
        score_cell_style = "background: #fff3cd;" if score_changed else ""
        prev_rank_cell_style = "color: #666; font-style: italic;" if stat['prev_rank'] != "-" else "color: #999;"
        prev_score_cell_style = "color: #666; font-style: italic;" if stat['prev_score'] != 0 else "color: #999;"
        
        stats_html += f"""
                <tr>
                    <td style='padding: 5px; border: 1px solid #ddd;'>{stat['name']}</td>
                    <td style='padding: 5px; border: 1px solid #ddd; {rank_cell_style}'><strong>{stat['rank']}</strong></td>
                    <td style='padding: 5px; border: 1px solid #ddd; {prev_rank_cell_style}'>{stat['prev_rank']}</td>
                    <td style='padding: 5px; text-align: right; border: 1px solid #ddd; {score_cell_style}'><strong>{stat['score']:,}</strong></td>
                    <td style='padding: 5px; text-align: right; border: 1px solid #ddd; {prev_score_cell_style}'>{stat['prev_score']:,}</td>
                    <td style='padding: 5px; text-align: center; border: 1px solid #ddd;'>{stat['outdeg']}</td>
                    <td style='padding: 5px; text-align: center; border: 1px solid #ddd;'>{stat['indeg']}</td>
                </tr>
        """
    
    stats_html += """
            </table>
        </div>
    </div>
    """
    display(HTML(stats_html))

def show_network_graph(contract, users, ui):
    """Displays the network graph (logic from ui._update_graph)."""
    if contract.network.number_of_nodes() == 0:
        print("No nodes in network")
        return

    scores = contract.compute_scores_for_display()

    focus_node_idx = ui.focus_node_idx
    
    if focus_node_idx is not None and focus_node_idx in contract.network.nodes():
        focus_node = focus_node_idx
        connected_components = list(nx.weakly_connected_components(contract.network))
        
        connected_nodes = None
        for component in connected_components:
            if focus_node in component:
                connected_nodes = component
                break
        
        if connected_nodes:
            G = contract.network.subgraph(list(connected_nodes)).copy()
        else:
            G = contract.network.subgraph([focus_node]).copy()
    else:
        G = contract.network.copy()

    pos = nx.shell_layout(G)

    node_order = sorted(G.nodes())
    labels = {}
    node_scores = {}
    node_ranks = {}
    node_rank_values = {}
    node_names = {}
    
    for idx in node_order:
        address = contract.idx_to_address.get(idx)
        if address:
            user_name = users[address]['name']
            display_name = ui._get_display_name(user_name)
            full_node_order = sorted(contract.network.nodes())
            score_idx = full_node_order.index(idx) if idx in full_node_order else 0
            score_val = scores[score_idx] if score_idx < len(scores) else 0.0
            rank = contract.get_rank(address)
            rank_display = "Default rank" if rank >= DEFAULT_RANK else str(rank)
            
            labels[idx] = f"{display_name}<br>Rank: {rank_display}<br>Score: {score_val:,.0f}"
            node_scores[idx] = score_val
            node_ranks[idx] = rank_display
            node_rank_values[idx] = rank
            node_names[idx] = display_name
        else:
            labels[idx] = f"Node {idx}"
            node_scores[idx] = 0.0
            node_ranks[idx] = "N/A"
            node_rank_values[idx] = DEFAULT_RANK
            node_names[idx] = f"Node {idx}"
    
    rank_values_list = [node_rank_values[idx] for idx in node_order]
    non_default_ranks = [r for r in rank_values_list if r < DEFAULT_RANK]
    has_default_ranks = any(r >= DEFAULT_RANK for r in rank_values_list)
    
    if non_default_ranks:
        min_rank = min(non_default_ranks)
        max_rank = max(non_default_ranks)
    else:
        min_rank = DEFAULT_RANK
        max_rank = DEFAULT_RANK
    
    score_values_list = [node_scores[idx] for idx in node_order]
    
    if score_values_list:
        min_score = min(score_values_list)
        max_score = max(score_values_list)
    else:
        min_score = 0
        max_score = 1
    
    colormap = cm.get_cmap('viridis')
    min_size = 15
    max_size = 35
    
    node_colors_final = []
    node_sizes_final = []
    for idx in node_order:
        score = node_scores[idx]
        rank_val = node_rank_values[idx]
        
        if rank_val >= DEFAULT_RANK:
            normalized_rank = 0.0
        elif max_rank > min_rank:
            normalized_rank = (max_rank - rank_val) / (max_rank - min_rank)
        else:
            normalized_rank = 0.5
        
        rgba = colormap(normalized_rank)
        color = mcolors.rgb2hex(rgba)
        node_colors_final.append(color)
        
        if max_score > min_score:
            normalized_score = (score - min_score) / (max_score - min_score)
        else:
            normalized_score = 0.5
        
        base_size = min_size + normalized_score * (max_size - min_size)
        
        if idx == focus_node_idx:
            node_sizes_final.append(base_size + 5)
        else:
            node_sizes_final.append(base_size)
    
    edge_x = []
    edge_y = []
    arrow_annotations = []
    
    if not isinstance(G, nx.DiGraph):
        G = nx.DiGraph(G)
    
    if len(pos) > 0:
        all_x = [p[0] for p in pos.values()]
        all_y = [p[1] for p in pos.values()]
        layout_scale = max(max(all_x) - min(all_x), max(all_y) - min(all_y)) if all_x and all_y else 1.0
        node_radius = layout_scale * 0.03
    else:
        node_radius = 0.03
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        dx = x1 - x0
        dy = y1 - y0
        edge_length = np.sqrt(dx**2 + dy**2)
        
        if edge_length > 0:
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            dx_norm = dx / edge_length
            dy_norm = dy / edge_length
            arrow_length = edge_length * 0.55
            arrow_start_x = mid_x - dx_norm * arrow_length / 2
            arrow_start_y = mid_y - dy_norm * arrow_length / 2
            arrow_end_x = mid_x + dx_norm * arrow_length / 2
            arrow_end_y = mid_y + dy_norm * arrow_length / 2
            
            arrow_annotations.append(
                dict(
                    ax=arrow_start_x,
                    ay=arrow_start_y,
                    x=arrow_end_x,
                    y=arrow_end_y,
                    xref='x',
                    yref='y',
                    axref='x',
                    ayref='y',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.2,
                    arrowwidth=1.8,
                    arrowcolor='#888'
                )
            )
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    node_x_focus, node_y_focus, node_text_focus, node_colors_focus, node_sizes_focus = [], [], [], [], []
    node_x_normal, node_y_normal, node_text_normal, node_colors_normal, node_sizes_normal = [], [], [], [], []
    
    for idx in node_order:
        x, y = pos[idx]
        address = contract.idx_to_address.get(idx)
        if address:
            rank = contract.get_rank(address)
            rank_display = "D" if rank >= DEFAULT_RANK else str(rank)
            score_val = node_scores[idx]
            user_name = node_names[idx]
            label = f"{user_name}<br>Rank: {rank_display}<br>Score: {score_val:,.0f}"
        else:
            label = f"Node {idx}"
        
        if idx == focus_node_idx:
            node_x_focus.append(x)
            node_y_focus.append(y)
            node_text_focus.append(label)
            node_colors_focus.append(node_colors_final[node_order.index(idx)])
            node_sizes_focus.append(node_sizes_final[node_order.index(idx)])
        else:
            node_x_normal.append(x)
            node_y_normal.append(y)
            node_text_normal.append(label)
            node_colors_normal.append(node_colors_final[node_order.index(idx)])
            node_sizes_normal.append(node_sizes_final[node_order.index(idx)])
    
    node_trace_normal = go.Scatter(
        x=node_x_normal, y=node_y_normal,
        mode='markers',
        hoverinfo='text',
        hovertext=node_text_normal,
        marker=dict(
            color=node_colors_normal,
            size=node_sizes_normal,
            opacity=1.0,
            line=dict(width=0, color='black')
        )
    )
    
    node_trace_focus = go.Scatter(
        x=node_x_focus, y=node_y_focus,
        mode='markers',
        hoverinfo='text',
        hovertext=node_text_focus,
        marker=dict(
            color=node_colors_focus,
            size=node_sizes_focus,
            opacity=1.0,
            line=dict(width=4, color='#FFD700')
        )
    ) if node_x_focus else None
    
    legend_shapes = []
    legend_annotations = []
    legend_y_start = 0.98
    legend_x_start = 0.02
    legend_x_color = legend_x_start
    legend_x_text = legend_x_start + 0.025
    legend_height = 0.25
    legend_width = 0.015
    num_legend_steps = 10
    legend_y_bottom = legend_y_start - legend_height
    step_height = legend_height / num_legend_steps
    
    def format_rank(rank_val):
        if rank_val >= DEFAULT_RANK:
            return "DEFAULT RANK"
        else:
            return str(int(rank_val))
    
    for i in range(num_legend_steps):
        step_val = i / (num_legend_steps - 1) if num_legend_steps > 1 else 0
        rgba = colormap(step_val)
        step_color = mcolors.rgb2hex(rgba)
        y0 = legend_y_bottom + i * step_height
        y1 = legend_y_bottom + (i + 1) * step_height
        
        legend_shapes.append(
            dict(
                type="rect",
                xref="paper", yref="paper",
                x0=legend_x_color - legend_width/2,
                y0=y0,
                x1=legend_x_color + legend_width/2,
                y1=y1,
                fillcolor=step_color,
                line=dict(color="#000", width=0.5)
            )
        )
    
    legend_shapes.append(
        dict(
            type="rect",
            xref="paper", yref="paper",
            x0=legend_x_color - legend_width/2,
            y0=legend_y_bottom,
            x1=legend_x_color + legend_width/2,
            y1=legend_y_start,
            fillcolor="rgba(0,0,0,0)",
            line=dict(color="#000", width=1.5)
        )
    )
    
    if non_default_ranks and max_rank > min_rank:
        best_rank_display = format_rank(min_rank)
        worst_rank_display = format_rank(max_rank)
    elif non_default_ranks and has_default_ranks:
        best_rank_display = format_rank(min_rank)
        worst_rank_display = "DEFAULT RANK"
    elif non_default_ranks:
        best_rank_display = format_rank(min_rank)
        worst_rank_display = format_rank(min_rank)
    else:
        best_rank_display = "DEFAULT RANK"
        worst_rank_display = "DEFAULT RANK"
    
    legend_annotations.append(
        dict(
            text=best_rank_display,
            showarrow=False,
            xref="paper", yref="paper",
            x=legend_x_text, y=legend_y_start,
            xanchor="left", yanchor="top",
            font=dict(color="#000", size=11, family="Arial", weight="bold")
        )
    )
    legend_annotations.append(
        dict(
            text=worst_rank_display,
            showarrow=False,
            xref="paper", yref="paper",
            x=legend_x_text, y=legend_y_bottom,
            xanchor="left", yanchor="bottom",
            font=dict(color="#000", size=11, family="Arial", weight="bold")
        )
    )
    legend_annotations.append(
        dict(
            text="Node color represents rank (lighter = better rank), size represents score",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002,
            xanchor="left", yanchor="bottom",
            font=dict(color="#888", size=12)
        )
    )
    
    title_text = (
        f"SYB Network Graph - Focused on {node_names.get(focus_node_idx, 'Node')}"
        if focus_node_idx is not None and focus_node_idx in contract.network.nodes()
        else "SYB Network Graph (Rank & Score) - Full View"
    )
    
    figure_data = [edge_trace, node_trace_normal]
    if node_trace_focus:
        figure_data.append(node_trace_focus)
    
    all_annotations = arrow_annotations + legend_annotations
    
    fig = go.Figure(
        data=figure_data,
        layout=go.Layout(
            title=dict(
                text=title_text,
                font=dict(size=18)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=all_annotations,
            shapes=legend_shapes,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
    )
    
    display(fig)
