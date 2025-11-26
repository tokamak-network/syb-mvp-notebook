import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import random
import os
import itertools # For generating random initial vouches

# --- Definitions from contract_interface_mvp.py ---

# Address Generation Functions (Crucial for VouchMinimal)
def generate_random_eth_address():
    return f"0x{random.randint(0, 0xffffffffffffffffffffffffffffffffffffffff):040x}"

def generate_mul_eth_addresses(n):
    return [generate_random_eth_address() for _ in range(n)]

# Constants
DEFAULT_RANK = 6
R = 5
SCOREBOOST_OUT = 1
SCOREBOOST_CAP = 15
SEEDVOUCHCOUNT = 5

@dataclass
class Node:
    """Node state matching the Solidity contract."""
    rank: int = 0
    score: int = 0
    outdegree: int = 0
    in_neighbors: List[str] = field(default_factory=list)
    previous_rank: int = 0
    previous_score: int = 0


# --- UPDATED VouchMinimal Class (Including Structured __init__) ---

class VouchMinimal:
    """Python implementation of VouchMinimal Solidity contract."""
    
    def __init__(self, network: nx.DiGraph=None):
        """
        Initializes contract with network using an Iterative Multi-Source Layered BFS Replay.
        
        The process starts with Node 0 (Alice). If the graph is disjoint, it automatically
        restarts the Layered BFS from the lowest-index source node in the remaining
        unprocessed edges until the entire graph is covered.
        """
        self.nodes: Dict[str, Node] = {}
        self.has_edge: Dict[str, Dict[str, bool]] = {}
        self.seed_vouch_count = 0
        self.address_to_idx: Dict[str, int] = {}
        self.idx_to_address: Dict[int, str] = {}
        self._next_idx = 0

        if network is None:
            raise ValueError("Network is required.")
        
        self.network = network.copy() 
        
        # 1. Address Generation and Initial Node Setup
        network_nodes = list(network.nodes())
        if not network_nodes:
            return
        
        num_nodes = len(network_nodes)
        addresses = generate_mul_eth_addresses(num_nodes)
        node_to_address = dict(zip(sorted(network_nodes), addresses))
        
        for node_idx, address in node_to_address.items():
            self.address_to_idx[address] = node_idx
            self.idx_to_address[node_idx] = address
            if address not in self.nodes:
                self.nodes[address] = Node()
            self._next_idx = max(self._next_idx, node_idx + 1)
        
        # --- 2. Full Graph Replay Strategy (Iterative Multi-Source BFS) ---
        TARGET_NODE_IDX = 0 # Alice, the primary starting point
        all_original_edges = list(network.edges())
        edges_to_process = set(all_original_edges)
        ordered_edges_to_replay = []

        initial_total_edges = len(all_original_edges)
        run_count = 0
        
        while edges_to_process:
            run_count += 1
            
            # a. Determine the current source/target for BFS
            if run_count == 1:
                # First pass: start with Alice (0)
                current_source_node = TARGET_NODE_IDX
            else:
                # Subsequent passes: find the lowest index source node among remaining edges
                current_source_node = min(e[0] for e in edges_to_process)
            
            print(f"Diffusion init (Run {run_count}): Starting Layered BFS from source Index {current_source_node}")

            # b. Create a subgraph containing only the remaining edges for BFS calculation
            subgraph = nx.DiGraph()
            subgraph.add_edges_from(edges_to_process)
            subgraph_nodes = set(subgraph.nodes())

            # c. Calculate BFS layers from the current source within the subgraph
            try:
                # Shortest path on the *remaining* graph
                # Note: nx.shortest_path_length is not strictly necessary 
                # but it is the most direct and efficient way
                distances = nx.shortest_path_length(subgraph, source=current_source_node)
            except nx.NetworkXNoPath:
                # If current_source_node is isolated in the remaining edges, this should not happen 
                # as it was chosen as an edge source, but we break to prevent infinite loops.
                break 

            nodes_by_layer = {}
            max_distance = 0
            for node, dist in distances.items():
                if dist != float('inf'):
                    if dist not in nodes_by_layer:
                        nodes_by_layer[dist] = []
                    nodes_by_layer[dist].append(node)
                    max_distance = max(max_distance, dist)

            processed_in_this_run = set()
            
            # d. Apply Layered BFS (Outward + Feedback)
            for k in range(max_distance + 1):
                current_sources = nodes_by_layer.get(k, [])
                
                # 1. PHASE A (Outward): Edges starting from the current layer sources
                phase_a_edges = []
                for u in current_sources:
                    u_out_edges = [e for e in edges_to_process if e[0] == u]
                    phase_a_edges.extend(u_out_edges)
                
                new_a_edges = sorted(list(set(phase_a_edges))) 
                ordered_edges_to_replay.extend(new_a_edges)
                processed_in_this_run.update(new_a_edges)

                # 2. PHASE B (Feedback to current_source_node)
                next_layer_targets = nodes_by_layer.get(k + 1, [])
                phase_b_edges = []
                for v in next_layer_targets:
                    feedback_edge = (v, current_source_node)
                    if feedback_edge in edges_to_process:
                        phase_b_edges.append(feedback_edge)
                        
                new_b_edges = sorted(list(set(phase_b_edges)))
                ordered_edges_to_replay.extend(new_b_edges)
                processed_in_this_run.update(new_b_edges)

            # e. Update remaining_edges
            if not processed_in_this_run:
                # Should not be reached if current_source_node was correctly determined
                break
                
            edges_to_process.difference_update(processed_in_this_run)
            
            print(f"Diffusion init (Run {run_count}): Processed {len(processed_in_this_run)} edges. {len(edges_to_process)} remaining.")


        print(f"Diffusion init: Total edges ordered for replay: {len(ordered_edges_to_replay)} out of {initial_total_edges}.")
        
        # 3. Execute the vouches in the new sequence
        for from_idx, to_idx in ordered_edges_to_replay:
            from_address = self.idx_to_address[from_idx]
            to_address = self.idx_to_address[to_idx]

            # We must remove the edge from the networkx graph copy before vouching, 
            # as vouch() will re-add it (essential for correct state counting during replay).
            if self.network.has_edge(from_idx, to_idx):
                self.network.remove_edge(from_idx, to_idx)
                
            try:
                self.vouch(from_address, to_address)
            except ValueError as e:
                pass # Exceptions like 'exists' are handled by the replay logic.
    
    # --- (vouch, get_score, get_rank, etc. methods are required but omitted for brevity) ---
    def _get_or_create_idx(self, address: str) -> int:
        if address not in self.address_to_idx:
            idx = self._next_idx
            self._next_idx += 1
            self.address_to_idx[address] = idx
            self.idx_to_address[idx] = address
            self.network.add_node(idx)
        return self.address_to_idx[address]
    
    def _rank_or_default(self, address: str) -> int:
        node = self.nodes.get(address)
        if not node:
            return 0
        r = node.rank
        return DEFAULT_RANK if r == 0 else r
    
    def _w(self, r: int) -> int:
        if r >= DEFAULT_RANK:
            return 0
        if r <= R:
            return 1 << (R - r)
        return 0
    
    def _recompute_rank_only(self, address: str):
        node = self.nodes.get(address)
        if not node or len(node.in_neighbors) == 0:
            if address in self.nodes:
                if self.nodes[address].rank != 0:
                    self.nodes[address].previous_rank = self.nodes[address].rank
                self.nodes[address].rank = DEFAULT_RANK
            return
        
        if node.rank != 0:
            node.previous_rank = node.rank
        
        k = DEFAULT_RANK
        m = 0
        
        for neighbor in node.in_neighbors:
            r_u = self._rank_or_default(neighbor)
            if r_u < k:
                k = r_u
                m = 1
            elif r_u == k:
                m += 1
        
        m_modified = min(m, 3)
        rv = 3 * k + 1 - m_modified
        
        node.rank = rv
    
    def _recompute_score(self, address: str):
        node = self.nodes.get(address)
        if not node:
            return
        
        if node.score != 0:
            node.previous_score = node.score
        
        s = 0
        
        for neighbor in node.in_neighbors:
            r_u = self._rank_or_default(neighbor)
            s += self._w(r_u)
        
        outdeg = node.outdegree
        if outdeg > SCOREBOOST_CAP:
            outdeg = SCOREBOOST_CAP
        s += SCOREBOOST_OUT * outdeg
        
        node.score = s
    
    def vouch(self, from_address: str, to_address: str):
        if not to_address or to_address == "":
            raise ValueError("zero")
        if from_address == to_address:
            raise ValueError("self")
        
        if self.has_vouch(from_address, to_address):
            raise ValueError("exists")
        
        if from_address not in self.has_edge:
            self.has_edge[from_address] = {}
        self.has_edge[from_address][to_address] = True
        
        if from_address not in self.nodes:
            self.nodes[from_address] = Node()
        if to_address not in self.nodes:
            self.nodes[to_address] = Node()
        
        from_node = self.nodes[from_address]
        to_node = self.nodes[to_address]
        
        from_node.outdegree += 1
        
        if from_address not in to_node.in_neighbors:
            to_node.in_neighbors.append(from_address)
        
        from_idx = self._get_or_create_idx(from_address)
        to_idx = self._get_or_create_idx(to_address)
        if not self.network.has_edge(from_idx, to_idx):
            self.network.add_edge(from_idx, to_idx)
        
        if self.seed_vouch_count < SEEDVOUCHCOUNT:
            if from_node.rank != 0:
                from_node.previous_rank = from_node.rank
            if to_node.rank != 0:
                to_node.previous_rank = to_node.rank
            
            from_node.rank = 1
            to_node.rank = 1
            
            self._recompute_score(from_address)
            self._recompute_score(to_address)
            
            self.seed_vouch_count += 1
            return
        
        self._recompute_rank_only(to_address)
        self._recompute_score(from_address)
        self._recompute_score(to_address)
    
    def has_vouch(self, from_address: str, to_address: str) -> bool:
        if from_address not in self.has_edge:
            return False
        return self.has_edge[from_address].get(to_address, False)
    
    def get_rank(self, address: str) -> int:
        node = self.nodes.get(address)
        if not node:
            return 0
        r = node.rank
        return DEFAULT_RANK if r == 0 else r
    
    def get_score(self, address: str) -> int:
        node = self.nodes.get(address)
        return node.score if node else 0

# --- Helper function from utils.py ---

def generate_alphabetical_names(n: int) -> List[str]:
    names = [
        "Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Henry",
        "Ivy", "Jack", "Kate", "Leo", "Mia", "Noah", "Olivia", "Paul",
        "Quinn", "Rachel", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xavier",
        "Yara", "Zoe",
        "Aaron", "Bella", "Clara", "Dylan", "Eliza", "Finn", "Gina", "Hank",
        "Isabel", "Joel", "Kira", "Liam", "Mona", "Nate", "Opal", "Perry",
        "Quincy", "Rosa", "Seth", "Tara", "Uri", "Vera", "Will", "Xena",
        "Yosef", "Zara",
        "Ava", "Ben", "Cara", "Derek", "Elsa", "Felix", "Gia", "Harvey",
        "Ingrid", "Jade", "Kyle", "Leah", "Mason", "Nina", "Omar", "Paula",
        "Queenie", "Rex", "Sara", "Trevor", "Ulric", "Violet", "Wade", "Ximena",
        "Yvonne", "Zander",
    ]
    if n > len(names):
        extended = []
        for i in range(n):
            if i < len(names):
                extended.append(names[i])
            else:
                base_idx = i % len(names)
                suffix = (i // len(names)) + 1
                extended.append(f"{names[base_idx]}{suffix}")
        return extended[:n]
    return names[:n]

# --- Helper for plotting and printing ---

def print_status(vm: VouchMinimal, users: dict, title: str):
    """Prints a formatted table of user ranks and scores."""
    user_data = []
    
    # Get alphabetical display names
    display_names = {addr: data['name'] for addr, data in users.items()}

    # Sort by score descending for readability
    sorted_addresses = sorted(users.keys(), key=lambda addr: vm.get_score(addr), reverse=True)

    for addr in sorted_addresses:
        score = vm.get_score(addr)
        rank = vm.get_rank(addr)
        user_data.append({
            'User': display_names[addr],
            'Rank': 'DEFAULT' if rank >= DEFAULT_RANK else str(rank),
            'Score': f"{score:,}"
        })

    df = pd.DataFrame(user_data)
    
    print(f"\n--- {title} (Sorted by Score) ---")
    print(df[['User', 'Rank', 'Score']].to_string(index=False))

def plot_status(vm: VouchMinimal, users: dict, title: str, filename: str):
    """Generates and saves the graph plot."""
    graph = vm.network
    pos = nx.spring_layout(graph, seed=42)
    
    # Collect data for plotting
    node_scores = [vm.get_score(vm.idx_to_address.get(node_idx)) for node_idx in graph.nodes()]
    node_ranks = [vm.get_rank(vm.idx_to_address.get(node_idx)) for node_idx in graph.nodes()]
    labels = {node_idx: users[vm.idx_to_address.get(node_idx)]['name'] for node_idx in graph.nodes()}

    # Size calculation
    scores_array = np.array(node_scores)
    min_score, max_score = scores_array.min(), scores_array.max()
    min_size, max_size = 500, 2000
    
    if max_score > min_score:
        normalized_scores = (scores_array - min_score) / (max_score - min_score)
        node_sizes = min_size + normalized_scores * (max_size - min_size)
    else:
        node_sizes = [min_size + (max_size - min_size) / 2] * len(scores_array)
        
    # Color calculation
    colors = []
    non_default_ranks = [r for r in node_ranks if r < DEFAULT_RANK]
    cmap = plt.cm.get_cmap('plasma')
    min_r, max_r = (min(non_default_ranks), max(non_default_ranks)) if non_default_ranks else (0, 0)

    for rank in node_ranks:
        if rank >= DEFAULT_RANK:
            colors.append('#cccccc')
        else:
            if max_r > min_r:
                normalized_rank = (rank - min_r) / (max_r - min_r)
                colors.append(cmap(1.0 - normalized_rank))
            else:
                colors.append(cmap(0.5))

    # Plotting
    plt.figure(figsize=(12, 10))
    nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color=colors, alpha=0.9)
    nx.draw_networkx_edges(graph, pos, edge_color='gray', arrowsize=20, width=1.5)
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=10, font_color='black')

    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Graph saved to {filename}")


# ------------------ MAIN DIFFUSION SCRIPT ------------------

# --- Configuration ---
N_NODES = 30
M_EDGES = 5
CLUSTER_SIZE = 5 # Nodes 0 through 4 (Alice, Bob, Charlie, David, Eve)
ATTACK_TARGET_INDEX = 0 # Alice

def run_diffusion_problem():
    
    # 1. Setup the Network
    # We use a randomized base graph to represent a real network
    base_graph = nx.barabasi_albert_graph(N_NODES, M_EDGES) 
    directed_network = base_graph.to_directed()
    
    # Contract initialization uses the structured __init__
    contract = VouchMinimal(directed_network)
    
    users = {}
    names = generate_alphabetical_names(len(contract.nodes))
    for addr, idx in contract.address_to_idx.items():
        users[addr] = {'name': names[idx], 'address': addr}

    # Identify Key Addresses
    target_addr = contract.idx_to_address[ATTACK_TARGET_INDEX]
    target_name = users[target_addr]['name']
    
    # Cluster (Inner) Addresses: Nodes 0-4 (Alice to Eve)
    cluster_addrs = [contract.idx_to_address[i] for i in range(CLUSTER_SIZE)]
    
    print(f"--- Diffusion Problem Setup ---")
    print(f"Target: {target_name} (Index {ATTACK_TARGET_INDEX})")
    print(f"Cluster Size: {CLUSTER_SIZE} users")
    print("---------------------------------")
    
    # --- INITIAL STATE ---
    print_status(contract, users, "Initial State (Base Graph Loaded)")
    plot_status(contract, users, 
                "Diffusion replay", 
                "diffusion_replay.png")

if __name__ == "__main__":
    run_diffusion_problem()