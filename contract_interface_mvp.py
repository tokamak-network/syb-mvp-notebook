"""MVP Contract Interface - Python implementation of VouchMinimal Solidity contract"""

import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import networkx as nx
# Removed: from utils import generate_mul_eth_addresses

# --- Address Generation Functions (Moved from utils.py) ---

def generate_random_eth_address():
    """Generates a fake random ethereum address."""
    return f"0x{random.randint(0, 0xffffffffffffffffffffffffffffffffffffffff):040x}"

def generate_mul_eth_addresses(n):
    """Generates a list of n fake random ethereum addresses."""
    return [generate_random_eth_address() for _ in range(n)]

# --- Constants matching Solidity contract ---
DEFAULT_RANK = 6 #10**24  # rank if no IN neighbors
R = 5 #64  # weight window: c_r = 2^(R - r) for r<=R
SCOREBOOST_OUT = 1 # 2**59  # per-edge outdegree bonus
# SCOREBOOST_CAP = 15  # cap for outdegree bonus
# SCOREBOOST_CAP depends on the rank of each node, so we set it dynamically in the code
SEEDVOUCHCOUNT = 5  # First N vouches seed endpoints to rank=1


@dataclass
class Node:
    """Node state matching the Solidity contract."""
    rank: int = 0  # 0 means not-initialized -> set to DEFAULT_RANK once initialized
    score: int = 0
    outdegree: int = 0
    in_neighbors: List[str] = field(default_factory=list)
    previous_rank: int = 0  # Previous rank value (0 means no previous value)
    previous_score: int = 0  # Previous score value


class VouchMinimal:
    """
    Python implementation of VouchMinimal Solidity contract.
    Implements vouching/unvouching with rank and score computation.
    """
    
    # def __init__(self, network: nx.DiGraph=None):
    #     """Initialize contract with network. Generates addresses for integer nodes."""
    #     self.nodes: Dict[str, Node] = {}
    #     self.has_edge: Dict[str, Dict[str, bool]] = {}  # from -> to -> bool
    #     self.seed_vouch_count = 0  # First 5 vouches seed endpoints to rank=1
    #     self.address_to_idx: Dict[str, int] = {}
    #     self.idx_to_address: Dict[int, str] = {}
    #     self._next_idx = 0

    #     """
    #     reference: https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.erdos_renyi_graph.html
    #     Note: the expected network is generated from a random digraph like
    #     `nx.erdos_renyi_graph(n_nodes, p, directed=True)` where
    #     - n_nodes is the number of nodes in the network
    #     - p is the probability of an edge between two nodes
    #     - directed=True means the graph is directed
    #     """
    #     if network is None:
    #         raise ValueError("Network is required.")
        
    #     self.network = network
        
    #     # Get all nodes from network
    #     network_nodes = list(network.nodes())
        
    #     # If network is empty, nothing to do
    #     if not network_nodes:
    #         return
        
    #     # Generate addresses and map network nodes to addresses
    #     num_nodes = len(network_nodes)
    #     addresses = generate_mul_eth_addresses(num_nodes)
    #     node_to_address = dict(zip(sorted(network_nodes), addresses))
        
    #     # Set up index mappings to preserve network node indices
    #     for node_idx, address in node_to_address.items():
    #         self.address_to_idx[address] = node_idx
    #         self.idx_to_address[node_idx] = address
    #         # Ensure a Node object is created for existing nodes
    #         if address not in self.nodes:
    #             self.nodes[address] = Node()
    #         self._next_idx = max(self._next_idx, node_idx + 1)
        
    #     # Create vouches for each edge - vouch() handles the rest
    #     # Make a copy of edges to iterate over as self.vouch modifies the graph
    #     edges_to_process = list(network.edges())
    #     for from_node, to_node in edges_to_process:
    #         from_address = node_to_address.get(from_node)
    #         to_address = node_to_address.get(to_node)
    #         if from_address and to_address:
    #             # Need to manually remove the edge from the graph first
    #             # because self.vouch() will add it back.
    #             # This ensures correct state initialization.
    #             if self.network.has_edge(from_node, to_node):
    #                 self.network.remove_edge(from_node, to_node)
    #             try:
    #                 self.vouch(from_address, to_address)
    #             except ValueError as e:
    #                 # e.g., "exists" error if logic is not perfect, just log
    #                 print(f"Warning during init: {e}")
        
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

            # b. Create a subgraph containing ALL nodes and only the remaining edges for BFS calculation
            subgraph = nx.DiGraph()
            # Ensure ALL nodes from the VouchMinimal instance are added first.
            subgraph.add_nodes_from(self.idx_to_address.keys())
            subgraph.add_edges_from(edges_to_process)

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
    
    def _get_or_create_idx(self, address: str) -> int:
        """Get or create index for an address."""
        if address not in self.address_to_idx:
            idx = self._next_idx
            self._next_idx += 1
            self.address_to_idx[address] = idx
            self.idx_to_address[idx] = address
            self.network.add_node(idx)
        return self.address_to_idx[address]
    
    def _rank_or_default(self, address: str) -> int:
        """Get rank for address, returning DEFAULT_RANK if rank is 0."""
        node = self.nodes.get(address)
        if not node:
            return 0
        r = node.rank
        return DEFAULT_RANK if r == 0 else r
    
    def _w(self, r: int) -> int:
        """
        Weight function: c_r = 2^(R - r) for r<=R, 0 otherwise or if r==DEFAULT_RANK.
        Matches Solidity contract's _w function.
        """
        if r >= DEFAULT_RANK:
            return 0
        if r <= R:
            return 1 << (R - r)  # 2^(R - r)
        return 0
    
    def _recompute_rank_only(self, address: str):
        """
        Recompute rank for address v: r[v] = 3*k + 1 - m
        where k = min rank over IN(v), m = multiplicity of k.
        Matches Solidity contract's _recomputeRankOnly function.
        """
        node = self.nodes.get(address)
        if not node or len(node.in_neighbors) == 0:
            if address in self.nodes:
                # Only save previous rank if it was actually computed (non-zero)
                if self.nodes[address].rank != 0:
                    self.nodes[address].previous_rank = self.nodes[address].rank
                self.nodes[address].rank = DEFAULT_RANK
            return
        
        # Only save previous rank if it was actually computed (non-zero)
        if node.rank != 0:
            node.previous_rank = node.rank
        
        # Find minimum rank and its multiplicity
        k = DEFAULT_RANK
        m = 0
        
        for neighbor in node.in_neighbors:
            r_u = self._rank_or_default(neighbor)
            if r_u < k:
                k = r_u
                m = 1
            elif r_u == k:
                m += 1
        
        # 3k + 1 - min(m, 3)
        m_modified = min(m, 3)
        rv = 3 * k + 1 - m_modified
        
        node.rank = rv
    
    def _recompute_score(self, address: str):
        """
        Recompute score for address: score[a] = sum_{u in IN(a)} c_{r[u]} + SCOREBOOST_OUT * min(SCOREBOOST_CAP, outdeg(a))
        Matches Solidity contract's _recomputeScore function.
        """
        node = self.nodes.get(address)
        if not node:
            return
        
        # Only save previous score if it was actually computed (non-zero)
        if node.score != 0:
            node.previous_score = node.score
        
        s = 0
        
        ## 1. Sum of weights from in-neighbors
        for neighbor in node.in_neighbors:
            r_u = self._rank_or_default(neighbor)
            s += self._w(r_u)
        
        # 2. Add outdegree bonus with rank-dependent cap
        outdeg = node.outdegree
        
        # Get the node's current effective rank (default to DEFAULT_RANK if 0)
        rank = self._rank_or_default(address)
        
        # Calculate the rank-dependent bonus cap:
        # SCOREBOOST_CAP = (rank > R) ? 0 : (2 ** (R - rank)) where R=5.
        # We use the internal weight function _w() as it calculates this exact value.
        SCOREBOOST_CAP = self._w(rank)
        
        # Apply the cap to the outdegree used for the bonus calculation
        if outdeg > SCOREBOOST_CAP:
            outdeg = SCOREBOOST_CAP
        
        s += SCOREBOOST_OUT * outdeg
        
        node.score = s
    
    def vouch(self, from_address: str, to_address: str):
        """
        Create a vouch (directed edge from -> to).
        Matches Solidity contract's vouch function.
        """
        # Validation
        if not to_address or to_address == "":
            raise ValueError("zero")
        if from_address == to_address:
            raise ValueError("self")
        
        # Check if edge already exists
        if self.has_vouch(from_address, to_address):
            raise ValueError("exists")
        
        # Create edge
        if from_address not in self.has_edge:
            self.has_edge[from_address] = {}
        self.has_edge[from_address][to_address] = True
        
        # Ensure nodes exist
        if from_address not in self.nodes:
            self.nodes[from_address] = Node()
        if to_address not in self.nodes:
            self.nodes[to_address] = Node()
        
        from_node = self.nodes[from_address]
        to_node = self.nodes[to_address]
        
        # Update outdegree
        from_node.outdegree += 1
        
        # Update in-neighbors
        if from_address not in to_node.in_neighbors:
            to_node.in_neighbors.append(from_address)
        
        # Update NetworkX graph
        from_idx = self._get_or_create_idx(from_address)
        to_idx = self._get_or_create_idx(to_address)
        if not self.network.has_edge(from_idx, to_idx):
            self.network.add_edge(from_idx, to_idx)
        
        # Bootstrap logic: first SEEDVOUCHCOUNT vouches (0-(SEEDVOUCHCOUNT-1)) seed endpoints to rank=1
        if self.seed_vouch_count < SEEDVOUCHCOUNT:
            # Only save previous ranks if they were actually computed (non-zero)
            if from_node.rank != 0:
                from_node.previous_rank = from_node.rank
            if to_node.rank != 0:
                to_node.previous_rank = to_node.rank
            
            # Set both endpoints to rank 1
            from_node.rank = 1
            to_node.rank = 1
            
            # Recompute scores
            self._recompute_score(from_address)
            self._recompute_score(to_address)
            
            self.seed_vouch_count += 1
            return
        
        # Normal: update rank(to) from IN(to), then recompute scores of from & to only
        self._recompute_rank_only(to_address)
        self._recompute_score(from_address)
        self._recompute_score(to_address)
    
    def unvouch(self, from_address: str, to_address: str):
        """
        Remove a vouch (directed edge from -> to).
        """
        # Validation
        if not to_address or to_address == "":
            raise ValueError("zero")
        if from_address == to_address:
            raise ValueError("self")
        
        # Check if edge exists
        if not self.has_vouch(from_address, to_address):
            raise ValueError("not found")
        
        # Remove edge
        if from_address in self.has_edge:
            self.has_edge[from_address].pop(to_address, None)
        
        # Update nodes
        if from_address in self.nodes:
            from_node = self.nodes[from_address]
            if from_node.outdegree > 0:
                from_node.outdegree -= 1
        
        if to_address in self.nodes:
            to_node = self.nodes[to_address]
            if from_address in to_node.in_neighbors:
                to_node.in_neighbors.remove(from_address)
        
        # Update NetworkX graph
        if from_address in self.address_to_idx and to_address in self.address_to_idx:
            from_idx = self.address_to_idx[from_address]
            to_idx = self.address_to_idx[to_address]
            if self.network.has_edge(from_idx, to_idx):
                self.network.remove_edge(from_idx, to_idx)
        
        # Recompute affected nodes
        if to_address in self.nodes:
            self._recompute_rank_only(to_address)
            self._recompute_score(to_address)
        
        if from_address in self.nodes:
            self._recompute_score(from_address)
    
    def has_vouch(self, from_address: str, to_address: str) -> bool:
        """Check if a vouch exists from -> to."""
        if from_address not in self.has_edge:
            return False
        return self.has_edge[from_address].get(to_address, False)
    
    def get_rank(self, address: str) -> int:
        """Get rank for address, returning DEFAULT_RANK if rank is 0."""
        node = self.nodes.get(address)
        if not node:
            return 0
        r = node.rank
        return DEFAULT_RANK if r == 0 else r
    
    def get_score(self, address: str) -> int:
        """Get score for address."""
        node = self.nodes.get(address)
        return node.score if node else 0
    
    def get_outdegree(self, address: str) -> int:
        """Get outdegree for address."""
        node = self.nodes.get(address)
        return node.outdegree if node else 0
    
    def get_in_count(self, address: str) -> int:
        """Get in-neighbor count for address."""
        node = self.nodes.get(address)
        return len(node.in_neighbors) if node else 0
    
    def get_in_neighbor_at(self, address: str, index: int) -> Optional[str]:
        """Get in-neighbor at index for address."""
        node = self.nodes.get(address)
        if not node or index < 0 or index >= len(node.in_neighbors):
            return None
        return node.in_neighbors[index]
    
    def get_previous_rank(self, address: str) -> int:
        """Get previous rank for address, returning DEFAULT_RANK if previous_rank is 0."""
        node = self.nodes.get(address)
        if not node:
            return DEFAULT_RANK
        r = node.previous_rank
        return DEFAULT_RANK if r == 0 else r
    
    def get_previous_score(self, address: str) -> int:
        """Get previous score for address."""
        node = self.nodes.get(address)
        return node.previous_score if node else 0
    
    def get_out_neighbors(self, address: str) -> List[str]:
        """Get list of out-neighbors (addresses that this address vouches for)."""
        out_neighbors = []
        if address in self.has_edge:
            for target in self.has_edge[address]:
                if self.has_edge[address][target]:
                    out_neighbors.append(target)
        return out_neighbors
    
    def compute_scores_for_display(self) -> List[float]:
        """
        Compute scores as floats for display purposes.
        Returns scores ordered by node index.
        """
        node_order = sorted(self.network.nodes())
        scores = []
        for idx in node_order:
            address = self.idx_to_address.get(idx)
            if address:
                score = self.get_score(address)
                scores.append(float(score))
            else:
                scores.append(0.0)
        return scores

