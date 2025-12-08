"""SYB MVP User Interface - Simplified UI for VouchMinimal Contract"""

import ipywidgets as widgets
from IPython.display import display, clear_output
from contract_interface_mvp import VouchMinimal, DEFAULT_RANK
from utils.plot_utils import (
    build_network_status_html,
    build_network_figure,
)
import matplotlib.pyplot as plt
from utils.utils import generate_alphabetical_names
import networkx as nx
from typing import Dict, Tuple


def create_random_mvp_network(num_users: int = 8) -> Tuple[VouchMinimal, Dict[str, Dict[str, str]]]:
    """
    Create a random MVP network with initial vouches.

    Returns:
        tuple: (contract, users) where users is a dict mapping addresses to user info
    """
    print(f"📊 Creating MVP network with {num_users} users...")

    network = nx.erdos_renyi_graph(num_users, 0.2, directed=True)

    # # Filter out edges involving node zero for a cleaner starting graph
    # edges_to_remove = [edge for edge in network.edges() if 0 in edge]
    # network.remove_edges_from(edges_to_remove)

    contract = VouchMinimal(network)

    print(f"✅ Created {contract.network.number_of_edges()} initial vouches")

    users: Dict[str, Dict[str, str]] = {}
    addresses_by_idx = {idx: addr for idx, addr in contract.idx_to_address.items()}
    user_names = generate_alphabetical_names(num_users)

    for idx in sorted(addresses_by_idx.keys()):
        addr = addresses_by_idx[idx]
        name = user_names[idx] if idx < len(user_names) else f"User {idx}"
        users[addr] = {
            "name": name,
            "address": addr,
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
        
        # Generate name mapping for all users
        self._name_mapping = self._generate_name_mapping()
        
        self._create_widgets()
        self._create_interface()
        self._connect_events()
    
    def _generate_name_mapping(self) -> dict:
        """Generate mapping from 'User 0', 'User 1', etc. to real names."""
        # Get all user names
        user_names = [data['name'] for data in self.users.values()]
        # Extract user indices
        user_indices = {}
        for name in user_names:
            if name.startswith('User '):
                try:
                    idx = int(name.split()[1])
                    user_indices[name] = idx
                except (ValueError, IndexError):
                    pass
        
        # Generate real names
        if user_indices:
            max_idx = max(user_indices.values())
            real_names = generate_alphabetical_names(max_idx + 1)
            # Create mapping
            mapping = {}
            for name, idx in user_indices.items():
                mapping[name] = real_names[idx]
            return mapping
        return {}
    
    def _get_display_name(self, name: str) -> str:
        """Get display name (real name if available, otherwise original)."""
        return self._name_mapping.get(name, name)
    
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
                    display_name = self._get_display_name(data['name'])
                    label = f"{display_name} ({addr[:10]}...)"
                    options.append((label, addr))
        return options if options else [("No users available", None)]
    
    def _get_unvouchable_users(self):
        """Get list of users that can be unvouched."""
        options = []
        for addr, data in self.users.items():
            if addr != self.current_user_address:
                # Only show if already vouched
                if self.contract.has_vouch(self.current_user_address, addr):
                    display_name = self._get_display_name(data['name'])
                    label = f"{display_name} ({addr[:10]}...)"
                    options.append((label, addr))
        return options if options else [("No vouches to remove", None)]
    
    def _create_interface(self):
        """Create main interface layout."""
        # Actions section
        vouch_row = widgets.HBox([
            self.vouch_target,
            self.vouch_btn
        ], layout=widgets.Layout(margin='5px'))
        
        unvouch_row = widgets.HBox([
            self.unvouch_target,
            self.unvouch_btn
        ], layout=widgets.Layout(margin='5px'))
        
        # Graph section
        graph_box = widgets.VBox([
            widgets.HTML(value="<h3>🌐 Network Graph</h3>"),
            self.graph_output
        ], layout=widgets.Layout(padding='0px', width='100%'))
        
        # Controls section (current user info + actions)
        user_info_section = widgets.VBox([
            widgets.HTML(value="<h4>👤 Current User</h4>"),
            self.user_info_display,
        ], layout=widgets.Layout(flex='1 1 0%', padding='0 20px 0 0'))
        
        actions_section = widgets.VBox([
            widgets.HTML(value="<h4>⚡ Actions</h4>"),
            vouch_row,
            unvouch_row,
            self.action_output,
        ], layout=widgets.Layout(flex='1 1 0%', padding='0 0 0 20px'))
        
        controls_row = widgets.HBox(
            [user_info_section, actions_section],
            layout=widgets.Layout(width='100%', justify_content='space-between', align_items='flex-start')
        )
        
        controls_box = widgets.VBox([
            widgets.HTML(value="<h3>⚙️ Controls</h3>"),
            controls_row,
        ], layout=widgets.Layout(padding='10px 0', width='100%'))
        
        # Status section (table / network status)
        status_box = widgets.VBox([
            widgets.HTML(value="<h3>📊 Network Status</h3>"),
            self.status_display
        ], layout=widgets.Layout(padding='10px 0', width='100%'))
        
        self.interface = widgets.VBox([
            graph_box,
            controls_box,
            status_box,
        ], layout=widgets.Layout(width='100%'))
    
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
        display_name = self._get_display_name(user_name)
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
            <div style='padding: 5px;'><strong>Name:</strong> {display_name}</div>
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
        stats_html = build_network_status_html(self.contract, self.users, self._get_display_name)
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
                figure = build_network_figure(
                    self.contract,
                    self.users,
                    self._get_display_name,
                    self.focus_node_idx,
                )
                if figure is None:
                    print("No nodes in network")
                else:
                    display(figure)
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
        user_name = self.users[self.current_user_address]['name']
        display_name = self._get_display_name(user_name)
        print(f"🔐 Connected as: {display_name} ({self.current_user_address[:12]}...)")
        print("✅ MVP Contract Interface - Transactions process immediately!")
        
        # Initial updates
        self._update_all()
        
        # Display interface
        display(self.interface)

