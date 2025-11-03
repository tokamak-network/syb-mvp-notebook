"""SYB MVP User Interface - Simplified UI for VouchMinimal Contract"""

import ipywidgets as widgets
from IPython.display import display, clear_output
from ContractInterfaceMvp import VouchMinimal, DEFAULT_RANK
from utils import generate_mul_eth_addresses
import networkx as nx
from comparison import plot_graph_with_scores
import matplotlib.pyplot as plt


def create_random_mvp_network(num_users=8) -> tuple:
    """
    Create a random MVP network with initial vouches.
    
    Returns:
        tuple: (contract, users) where users is a dict mapping addresses to user info
    """
    print(f"📊 Creating MVP network with {num_users} users...")
    
    # Create some initial vouches (seed the network)
    print("\n🤝 Creating initial vouching relationships...")
    
    # NOTE: could change this into other random graph from networkx
    network = nx.erdos_renyi_graph(num_users, 0.25, directed=True)
    print(f"Network nodes: {network.nodes()}")
    contract = VouchMinimal(network)
    
    print(f"✅ Created {contract.network.number_of_edges()} initial vouches")
    
    # Build users dictionary
    users = {}
    addresses = contract.idx_to_address.values()
    print(f"idx_to_address: {contract.idx_to_address}")
    print(f"Addresses: {addresses}")
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
        self.refresh_graph_btn = widgets.Button(
            description='🔄 Refresh Graph',
            button_style='info',
            layout=widgets.Layout(width='150px')
        )
        self.refresh_status_btn = widgets.Button(
            description='📊 Refresh Status',
            button_style='info',
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
        
        refresh_row = widgets.HBox([
            self.refresh_graph_btn,
            self.refresh_status_btn
        ], layout=widgets.Layout(margin='5px'))
        
        actions_box = widgets.VBox([
            widgets.HTML(value="<h3>⚡ Actions</h3>"),
            vouch_row,
            unvouch_row,
            refresh_row,
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
        self.refresh_graph_btn._click_handlers.callbacks.clear()
        self.refresh_status_btn._click_handlers.callbacks.clear()
        
        # Now connect handlers
        self.vouch_btn.on_click(self._handle_vouch)
        self.unvouch_btn.on_click(self._handle_unvouch)
        self.refresh_graph_btn.on_click(self._handle_refresh_graph)
        self.refresh_status_btn.on_click(self._handle_refresh_status)
        
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
    
    def _handle_refresh_graph(self, b):
        """Refresh the network graph."""
        with self.action_output:
            clear_output(wait=True)
            print("🔄 Refreshing graph...")
        
        self._update_graph()
        
        with self.action_output:
            clear_output(wait=True)
            print("✅ Graph refreshed!")
    
    def _handle_refresh_status(self, b):
        """Refresh status displays."""
        with self.action_output:
            clear_output(wait=True)
            print("🔄 Refreshing status...")
        
        self._update_status()
        self._update_user_info()
        
        with self.action_output:
            clear_output(wait=True)
            print("✅ Status refreshed!")
    
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
        user_stats.sort(key=lambda x: x['score'], reverse=True)
        
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
        fig = None
        try:
            with self.graph_output:
                clear_output(wait=True)

                if self.contract.network.number_of_nodes() == 0:
                    print("No nodes in network")
                    return  # Will still hit finally block

                # Get scores for visualization
                scores = self.contract.compute_scores_for_display()

                # Get node positions
                pos = nx.spring_layout(self.contract.network, seed=42)

                # Create labels with addresses and scores
                node_order = sorted(self.contract.network.nodes())
                labels = {}
                for idx in node_order:
                    address = self.contract.idx_to_address.get(idx)
                    if address:
                        user_name = self.users[address]['name']
                        score_idx = node_order.index(idx)
                        score_val = scores[score_idx] if score_idx < len(scores) else 0.0
                        rank = self.contract.get_rank(address)
                        rank_display = "D" if rank >= DEFAULT_RANK else str(rank)
                        labels[idx] = f"{user_name}\nR:{rank_display}\nS:{score_val:.2f}"
                    else:
                        labels[idx] = str(idx)

                # Create figure with interactive mode OFF
                plt.ioff()
                fig, ax = plt.subplots(figsize=(12, 10))
                nx.draw(
                    self.contract.network,
                    pos,
                    ax=ax,
                    with_labels=False,
                    node_color='skyblue',
                    node_size=1500,
                    edge_color='gray',
                    width=2,
                    arrows=True,
                    arrowsize=20,
                    arrowstyle='->'
                )
                nx.draw_networkx_labels(
                    self.contract.network,
                    pos,
                    labels=labels,
                    ax=ax,
                    font_size=9,
                    font_color='black'
                )
                ax.set_title("Network Graph (Rank & Score)", fontsize=16, fontweight='bold')
                ax.axis('off')
                plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

                # Render to buffer and display as image to avoid matplotlib display hooks
                import io
                from IPython.display import display as ipy_display, Image as IPyImage

                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                img_data = buf.getvalue()
                buf.close()

                # Display the image directly - bypasses matplotlib's display system
                ipy_display(IPyImage(data=img_data))
        finally:
            # Always close the figure to prevent accumulation
            if fig is not None:
                plt.close(fig)
            else:
                plt.close('all')
            self._updating_graph = False
    
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

