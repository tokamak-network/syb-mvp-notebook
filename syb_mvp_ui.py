"""SYB MVP User Interface - Simplified UI for VouchMinimal Contract"""

import ipywidgets as widgets
from IPython.display import display, clear_output
from ContractInterfaceMvp import VouchMinimal, DEFAULT_RANK
from utils import generate_mul_eth_addresses
import networkx as nx
from comparison import plot_graph_with_scores
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np


def create_random_mvp_network(num_users=8) -> tuple:
    """
    Create a random MVP network with initial vouches.
    
    Returns:
        tuple: (contract, users) where users is a dict mapping addresses to user info
    """
    print(f"📊 Creating MVP network with {num_users} users...")
    
    # NOTE: could change this into other random graph from networkx
    network = nx.erdos_renyi_graph(num_users, 0.2, directed=True)
    # filter the edge from node zero

    edges_to_remove = [edge for edge in network.edges() if 0 in edge]
    network.remove_edges_from(edges_to_remove)
    contract = VouchMinimal(network)
    
    print(f"✅ Created {contract.network.number_of_edges()} initial vouches")
    
    # Build users dictionary
    users = {}
    addresses = contract.idx_to_address.values()
    user_names = [f"User {i}" for i in range(num_users)]
    for i, addr in enumerate(addresses):
        users[addr] = {
            'name': user_names[i],
            'address': addr
        }
    
    return contract, users


class SYBMvpUserInterface:
    """Simplified UI for MVP contract - no batch processing, immediate updates."""
    
    def __init__(self, contract: VouchMinimal, users: dict, current_user_address: str):
        """
        Initialize MVP UI.
        
        Args:
            contract: VouchMinimal contract instance
            users: Dictionary of user info
            current_user_address: Address of the current user
        """
        self.contract = contract
        self.users = users
        self.current_user_address = current_user_address
        self._handlers_connected = False  # Track if handlers are already connected
        self._updating_graph = False  # Prevent duplicate graph updates
        self.focus_node_idx = None  # Node index to focus on (None = show all, or specific node index)
        
        # Set focus to current user's node by default
        if current_user_address in contract.address_to_idx:
            self.focus_node_idx = contract.address_to_idx[current_user_address]
        
        self._create_widgets()
        self._create_interface()
        self._connect_events()
    
    def _create_widgets(self):
        """Create UI widgets."""
        # Action buttons
        self.vouch_btn = widgets.Button(
            description='👍 Vouch',
            button_style='success',
            layout=widgets.Layout(width='150px')
        )
        self.unvouch_btn = widgets.Button(
            description='👎 Unvouch',
            button_style='danger',
            layout=widgets.Layout(width='150px')
        )
        
        # Input dropdowns
        self.vouch_target = widgets.Dropdown(
            options=self._get_vouchable_users(),
            description='Target:',
            layout=widgets.Layout(width='300px')
        )
        self.unvouch_target = widgets.Dropdown(
            options=self._get_unvouchable_users(),
            description='Target:',
            layout=widgets.Layout(width='300px')
        )
        
        # Display widgets
        self.status_display = widgets.HTML(
            value="<div style='padding: 10px;'>Loading...</div>"
        )
        self.graph_output = widgets.Output(layout=widgets.Layout(height='500px'))
        self.action_output = widgets.Output()
        
        # Info display
        self.user_info_display = widgets.HTML(
            value="<div style='padding: 10px;'>Loading...</div>"
        )
    
    def _get_vouchable_users(self):
        """Get list of users that can be vouched for."""
        options = []
        for addr, data in self.users.items():
            if addr != self.current_user_address:
                # Check if already vouched
                if not self.contract.has_vouch(self.current_user_address, addr):
                    label = f"{data['name']} ({addr[:10]}...)"
                    options.append((label, addr))
        return options if options else [("No users available", None)]
    
    def _get_unvouchable_users(self):
        """Get list of users that can be unvouched."""
        options = []
        for addr, data in self.users.items():
            if addr != self.current_user_address:
                # Only show if already vouched
                if self.contract.has_vouch(self.current_user_address, addr):
                    label = f"{data['name']} ({addr[:10]}...)"
                    options.append((label, addr))
        return options if options else [("No vouches to remove", None)]
    
    def _create_interface(self):
        """Create main interface layout."""
        # User info section
        user_info_box = widgets.VBox([
            widgets.HTML(value="<h3>👤 Current User</h3>"),
            self.user_info_display
        ], layout=widgets.Layout(border='2px solid #2E86AB', padding='15px'))
        
        # Actions section
        vouch_row = widgets.HBox([
            self.vouch_target,
            self.vouch_btn
        ], layout=widgets.Layout(margin='5px'))
        
        unvouch_row = widgets.HBox([
            self.unvouch_target,
            self.unvouch_btn
        ], layout=widgets.Layout(margin='5px'))
        
        actions_box = widgets.VBox([
            widgets.HTML(value="<h3>⚡ Actions</h3>"),
            vouch_row,
            unvouch_row,
            self.action_output
        ], layout=widgets.Layout(border='2px solid #4CAF50', padding='15px'))
        
        # Status section
        status_box = widgets.VBox([
            widgets.HTML(value="<h3>📊 Network Status</h3>"),
            self.status_display
        ], layout=widgets.Layout(border='2px solid #FF9800', padding='15px', width='100%'))
        
        # Graph section
        graph_box = widgets.VBox([
            widgets.HTML(value="<h3>🌐 Network Graph</h3>"),
            self.graph_output
        ], layout=widgets.Layout(border='2px solid #9C27B0', padding='15px'))
        
        # Main layout - side by side
        top_row = widgets.HBox([
            user_info_box,
            actions_box
        ], layout=widgets.Layout(justify_content='space-between'))
        
        self.interface = widgets.VBox([
            top_row,
            status_box,
            graph_box
        ])
    
    def _connect_events(self):
        """Connect button events to handlers."""
        # Clear any existing handlers first to prevent duplicates
        self.vouch_btn._click_handlers.callbacks.clear()
        self.unvouch_btn._click_handlers.callbacks.clear()
        
        # Now connect handlers
        self.vouch_btn.on_click(self._handle_vouch)
        self.unvouch_btn.on_click(self._handle_unvouch)
        
        self._handlers_connected = True
    
    def _handle_vouch(self, b):
        """Handle vouch action."""
        with self.action_output:
            clear_output(wait=True)
            try:
                target_addr = self.vouch_target.value
                if target_addr is None:
                    print("❌ Please select a valid target")
                    return
                
                self.contract.vouch(self.current_user_address, target_addr)
                
                # Update dropdowns
                self.vouch_target.options = self._get_vouchable_users()
                self.unvouch_target.options = self._get_unvouchable_users()
                

                self._update_all()
                
            except Exception as e:
                print(f"❌ Vouch failed: {e}")
    
    def _handle_unvouch(self, b):
        """Handle unvouch action."""
        with self.action_output:
            clear_output(wait=True)
            try:
                target_addr = self.unvouch_target.value
                if target_addr is None:
                    print("❌ Please select a valid target")
                    return
                
                self.contract.unvouch(self.current_user_address, target_addr)
                
                # Update dropdowns
                self.vouch_target.options = self._get_vouchable_users()
                self.unvouch_target.options = self._get_unvouchable_users()
                
                # Immediately update UI (don't print here - let UI updates speak for themselves)
                self._update_all()
                
            except Exception as e:
                print(f"❌ Unvouch failed: {e}")
    
    def _update_user_info(self):
        """Update current user information display."""
        user_name = self.users[self.current_user_address]['name']
        rank = self.contract.get_rank(self.current_user_address)
        score = self.contract.get_score(self.current_user_address)
        outdegree = self.contract.get_outdegree(self.current_user_address)
        in_count = self.contract.get_in_count(self.current_user_address)
        
        # Format rank for display
        if rank >= DEFAULT_RANK:
            rank_display = "DEFAULT"
        else:
            rank_display = str(rank)
        
        html = f"""
        <div style='padding: 10px; font-size: 14px;'>
            <div style='padding: 5px;'><strong>Name:</strong> {user_name}</div>
            <div style='padding: 5px;'><strong>Address:</strong> {self.current_user_address}</div>
            <div style='padding: 5px;'><strong>Rank:</strong> {rank_display}</div>
            <div style='padding: 5px;'><strong>Score:</strong> {score:,}</div>
            <div style='padding: 5px;'><strong>Outdegree:</strong> {outdegree}</div>
            <div style='padding: 5px;'><strong>In-degree:</strong> {in_count}</div>
        </div>
        """
        self.user_info_display.value = html
    
    def _update_status(self):
        """Update network status display."""
        num_users = len(self.contract.nodes)
        num_edges = self.contract.network.number_of_edges()
        
        # Get all users' stats
        user_stats = []
        for addr, data in self.users.items():
            rank = self.contract.get_rank(addr)
            score = self.contract.get_score(addr)
            prev_rank = self.contract.get_previous_rank(addr)
            prev_score = self.contract.get_previous_score(addr)
            outdeg = self.contract.get_outdegree(addr)
            indeg = self.contract.get_in_count(addr)
            
            rank_display = "DEFAULT" if rank >= DEFAULT_RANK else str(rank)
            
            # Handle previous rank display
            # Check if node actually has a previous rank set (by checking if it's not the initial 0)
            node = self.contract.nodes.get(addr)
            has_previous_rank = node and node.previous_rank > 0
            if not has_previous_rank:
                prev_rank_display = "-"  # No previous value set
            else:
                prev_rank_display = "DEFAULT" if prev_rank >= DEFAULT_RANK else str(prev_rank)
            
            user_stats.append({
                'name': data['name'],
                'rank': rank_display,
                'prev_rank': prev_rank_display,
                'score': score,
                'prev_score': prev_score,
                'outdeg': outdeg,
                'indeg': indeg
            })
        
        # Sort by score descending
        # TODO: add button to sort by rank, score, outdeg, indeg, username
        # default: sorted by names
        user_stats.sort(key=lambda x: x['name'])

        # Build HTML
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
            # Highlight if rank or score changed
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
        
        self.status_display.value = stats_html
    
    def _update_graph(self):
        """Update the network graph visualization."""
        # Prevent duplicate updates
        if self._updating_graph:
            return

        self._updating_graph = True
        try:
            with self.graph_output:
                clear_output(wait=True)

                if self.contract.network.number_of_nodes() == 0:
                    print("No nodes in network")
                    return  # Will still hit finally block

                # Get scores for visualization
                scores = self.contract.compute_scores_for_display()

                # Determine which graph to show: full graph or subgraph focused on a node
                if self.focus_node_idx is not None and self.focus_node_idx in self.contract.network.nodes():
                    # Find the weakly connected component containing the focus node
                    # This includes all transitively connected nodes
                    focus_node = self.focus_node_idx
                    connected_components = list(nx.weakly_connected_components(self.contract.network))
                    
                    # Find the component that contains the focus node
                    connected_nodes = None
                    for component in connected_components:
                        if focus_node in component:
                            connected_nodes = component
                            break
                    
                    # If found, create subgraph with the connected component
                    if connected_nodes:
                        G = self.contract.network.subgraph(list(connected_nodes)).copy()
                    else:
                        # Fallback: just the focus node itself
                        G = self.contract.network.subgraph([focus_node]).copy()
                else:
                    # Show full graph
                    G = self.contract.network.copy()

                # Get node positions using shell layout
                pos = nx.shell_layout(G)

                # Create labels with addresses and scores, and prepare node data
                node_order = sorted(G.nodes())
                labels = {}
                node_scores = {}
                node_ranks = {}
                node_names = {}
                
                for idx in node_order:
                    address = self.contract.idx_to_address.get(idx)
                    if address:
                        user_name = self.users[address]['name']
                        # Get score index from full network node order
                        full_node_order = sorted(self.contract.network.nodes())
                        score_idx = full_node_order.index(idx) if idx in full_node_order else 0
                        score_val = scores[score_idx] if score_idx < len(scores) else 0.0
                        rank = self.contract.get_rank(address)
                        rank_display = "D" if rank >= DEFAULT_RANK else str(rank)
                        
                        labels[idx] = f"{user_name}<br>Rank: {rank_display}<br>Score: {score_val:,.0f}"
                        node_scores[idx] = score_val
                        node_ranks[idx] = rank_display
                        node_names[idx] = user_name
                    else:
                        labels[idx] = f"Node {idx}"
                        node_scores[idx] = 0.0
                        node_ranks[idx] = "N/A"
                        node_names[idx] = f"Node {idx}"
                
                # Calculate node sizes based on scores (normalize to reasonable size range)
                score_values = [node_scores[idx] for idx in node_order]
                if score_values and max(score_values) > 0:
                    min_score = min(score_values)
                    max_score = max(score_values)
                    # Normalize to size range (15-40 for nodes)
                    node_sizes = []
                    for idx in node_order:
                        score = node_scores[idx]
                        if max_score > min_score:
                            normalized = (score - min_score) / (max_score - min_score)
                            size = 15 + normalized * 25  # Range from 15 to 40
                        else:
                            size = 20  # Default size if all scores are the same
                        node_sizes.append(size)
                else:
                    node_sizes = [20] * len(node_order)
                
                # Prepare node colors list first
                node_color_list = []
                for idx in node_order:
                    if idx == self.focus_node_idx:
                        node_color_list.append('gold')  # Bright gold
                    else:
                        node_color_list.append('skyblue')
                
                # Prepare node sizes and colors final (needed for arrow positioning)
                node_sizes_final = []
                node_colors_final = []
                for i, idx in enumerate(node_order):
                    if idx == self.focus_node_idx:
                        node_sizes_final.append(node_sizes[i] * 1.2)  # 20% larger
                        node_colors_final.append('gold')  # Gold color
                    else:
                        node_sizes_final.append(node_sizes[i])
                        node_colors_final.append(node_color_list[i])
                
                # Prepare edge traces with arrows (directed graph)
                edge_x = []
                edge_y = []
                arrow_annotations_list = []
                
                # Ensure G is a DiGraph to get directed edges
                if not isinstance(G, nx.DiGraph):
                    G = nx.DiGraph(G)
                
                # Calculate node radii for arrow positioning
                # Convert marker size to approximate coordinate space radius
                # Get the range of positions to estimate scale
                all_positions = [pos[node] for node in G.nodes()]
                if all_positions:
                    x_coords = [p[0] for p in all_positions]
                    y_coords = [p[1] for p in all_positions]
                    coord_range = max(max(x_coords) - min(x_coords), max(y_coords) - min(y_coords))
                    coord_range = coord_range if coord_range > 0 else 1.0
                    
                    # Estimate radius: node size in pixels -> fraction of coordinate range
                    node_radii = {}
                    for i, idx in enumerate(node_order):
                        size = node_sizes_final[i] if i < len(node_sizes_final) else 20
                        # Convert size (15-40 range) to coordinate units
                        # Rough estimate: 1% of coordinate range per 10 pixels of size
                        node_radii[idx] = (size / 100.0) * (coord_range * 0.01)
                else:
                    node_radii = {idx: 0.02 for idx in node_order}
                
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    
                    # Calculate direction vector
                    dx = x1 - x0
                    dy = y1 - y0
                    length = np.sqrt(dx**2 + dy**2)
                    
                    if length > 0:
                        # Normalize direction
                        dx_norm = dx / length
                        dy_norm = dy / length
                        
                        # Get target node radius
                        target_radius = node_radii.get(edge[1], 0.02)
                        
                        # Calculate arrow end point: stop at edge of target node
                        # Move back from center by the node radius
                        arrow_end_x = x1 - dx_norm * target_radius
                        arrow_end_y = y1 - dy_norm * target_radius
                        
                        # Calculate arrow start point: slightly before the end
                        arrow_start_offset = target_radius * 0.2  # Small gap before arrow
                        arrow_start_x = arrow_end_x - dx_norm * arrow_start_offset
                        arrow_start_y = arrow_end_y - dy_norm * arrow_start_offset
                        
                        # Draw the edge line (stop before target node)
                        edge_end_x = arrow_end_x - dx_norm * target_radius * 0.5
                        edge_end_y = arrow_end_y - dy_norm * target_radius * 0.5
                        edge_x.extend([x0, edge_end_x, None])
                        edge_y.extend([y0, edge_end_y, None])
                        
                        # Add arrow annotation pointing from source to target
                        arrow_annotations_list.append(
                            dict(
                                ax=arrow_start_x,
                                ay=arrow_start_y,
                                x=arrow_end_x,
                                y=arrow_end_y,
                                xref="x",
                                yref="y",
                                axref="x",
                                ayref="y",
                                showarrow=True,
                                arrowhead=2,
                                arrowsize=1.0,
                                arrowwidth=1.2,
                                arrowcolor="#666",
                                standoff=0,  # No standoff since we calculated it manually
                                startstandoff=0
                            )
                        )
                    else:
                        # Zero length edge (shouldn't happen)
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])
                
                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=2, color='#888'),
                    hoverinfo='none',
                    mode='lines'
                )
                
                # Prepare node traces with sizes based on scores
                node_x = []
                node_y = []
                node_text = []
                
                for idx in node_order:
                    x, y = pos[idx]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(labels[idx])
                
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers',
                    hoverinfo='text',
                    hovertext=node_text,
                    marker=dict(
                        color=node_colors_final,  # Gold for focus node, skyblue for others
                        size=node_sizes_final,  # Size based on score, larger for focus node
                        opacity=1.0,  # Fully opaque, not transparent
                        line=dict(width=2, color='black')
                    )
                )
                
                # Create figure
                title_text = (
                    f"SYB Network Graph - Focused on {node_names.get(self.focus_node_idx, 'Node')}"
                    if self.focus_node_idx is not None and self.focus_node_idx in self.contract.network.nodes()
                    else "SYB Network Graph (Rank & Score) - Full View"
                )
                
                fig = go.Figure(
                    data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=dict(
                            text=title_text,
                            font=dict(size=18)
                        ),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[
                            dict(
                                text="Node size represents score (larger = higher score)",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002,
                                xanchor="left", yanchor="bottom",
                                font=dict(color="#888", size=12)
                            )
                        ] + arrow_annotations_list,
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='white'
                    )
                )
                
                # Display plotly figure in Jupyter widget output
                from IPython.display import display
                display(fig)
                
                # TODO: an alternative to use the matplotlib code to display the graph
                # COMMENTED OUT: Original matplotlib code
                # # Create figure with interactive mode OFF
                # plt.ioff()
                # fig, ax = plt.subplots(figsize=(12, 10))
                # 
                # # Color the focus node differently
                # node_colors = []
                # node_sizes = []
                # for node in node_order:
                #     if node == self.focus_node_idx:
                #         node_colors.append('gold')
                #         node_sizes.append(2000)
                #     else:
                #         node_colors.append('skyblue')
                #         node_sizes.append(1500)
                # 
                # nx.draw(
                #     G,
                #     pos,
                #     ax=ax,
                #     with_labels=False,
                #     node_color=node_colors,
                #     node_size=node_sizes,
                #     edge_color='gray',
                #     width=2,
                #     arrows=True,
                #     arrowsize=20,
                #     arrowstyle='->'
                # )
                # nx.draw_networkx_labels(
                #     G,
                #     pos,
                #     labels=labels,
                #     ax=ax,
                #     font_size=9,
                #     font_color='black'
                # )
                # # Update title based on whether showing subgraph or full graph
                # if self.focus_node_idx is not None and self.focus_node_idx in self.contract.network.nodes():
                #     focus_name = labels.get(self.focus_node_idx, f"Node {self.focus_node_idx}")
                #     # Extract just the name part (before newline)
                #     focus_display = focus_name.split('\n')[0] if '\n' in focus_name else focus_name
                #     title = f"SYB Network Graph - Focused on {focus_display}"
                # else:
                #     title = "SYB Network Graph (Rank & Score) - Full View"
                # ax.set_title(title, fontsize=18, fontweight='bold')
                # ax.axis('off')
                # plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
                #
                # # Render to buffer and display as image to avoid matplotlib display hooks
                # import io
                # from IPython.display import display as ipy_display, Image as IPyImage
                #
                # buf = io.BytesIO()
                # fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                # buf.seek(0)
                # img_data = buf.getvalue()
                # buf.close()
                #
                # # Display the image directly - bypasses matplotlib's display system
                # ipy_display(IPyImage(data=img_data))
        finally:
            self._updating_graph = False
    
    def set_focus_node(self, node_idx: int = None):
        """
        Set the focus node for the graph visualization.
        
        Args:
            node_idx: Node index to focus on (None to show full graph)
        """
        if node_idx is None:
            self.focus_node_idx = None
        elif node_idx in self.contract.network.nodes():
            self.focus_node_idx = node_idx
        else:
            raise ValueError(f"Node {node_idx} not found in network")
        self._update_graph()
    
    def set_focus_by_address(self, address: str):
        """
        Set the focus node by user address.
        
        Args:
            address: User address to focus on
        """
        if address in self.contract.address_to_idx:
            self.focus_node_idx = self.contract.address_to_idx[address]
            self._update_graph()
        else:
            raise ValueError(f"Address {address} not found")

    def _update_all(self):
        """Update all UI elements."""
        self._update_user_info()
        self._update_status()
        self._update_graph()
    
    def display(self):
        """Display the MVP interface."""
        plt.close('all')
        print(f"🔐 Connected as: {self.users[self.current_user_address]['name']} ({self.current_user_address[:12]}...)")
        print("✅ MVP Contract Interface - Transactions process immediately!")
        
        # Initial updates
        self._update_all()
        
        # Display interface
        display(self.interface)

