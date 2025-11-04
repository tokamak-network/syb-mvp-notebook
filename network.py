"""
Network class for graph operations with balance tracking and VouchMinimal scoring.

This class manages a directed graph with node balances and uses the 
VouchMinimal algorithm to compute node ranks and scores automatically.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Union
from scipy.sparse import csr_matrix
from plot_utils import generate_random_graph, plot_graph_with_scores

# Import the VouchMinimal class and Node dataclass
from contract_interface_mvp import VouchMinimal, Node, generate_mul_eth_addresses


class Network:
    """
    A network class that manages a directed graph with node balances and
    integrates the VouchMinimal scoring algorithm.
    
    The graph represents the network structure, and balance_list tracks the balance
    of each node. Scores are computed by the VouchMinimal instance on
    vouch/unvouch operations (add_edge/remove_edge).
    """
    
    def __init__(self, graph: Optional[nx.DiGraph] = None, balance_list: Optional[List[float]] = None):
        """
        Initialize the network.
        
        Args:
            graph: NetworkX DiGraph (optional). If provided, VouchMinimal will
                   be initialized with this graph, and all existing edges
                   will be processed as vouches.
            balance_list: List of balances for each node (optional)
        """
        self.graph = graph or nx.DiGraph()
        self.balance_list = balance_list or []
        
        # Initialize VouchMinimal with the graph.
        # self.vm.network is an alias for self.graph
        self.vm = VouchMinimal(self.graph) 
        
        self._node_order = []
        self._update_node_order()
        self._ensure_balance_consistency()
    
    @classmethod
    def from_matrix(cls, matrix: Union[np.ndarray, csr_matrix], 
                   balance_list: Optional[List[float]] = None,
                   create_using: type = nx.DiGraph) -> 'Network':
        """
        Create network from adjacency matrix.
        
        Args:
            matrix: Adjacency matrix
            balance_list: List of balances for each node (optional)
            create_using: Graph class to create (should be nx.DiGraph or compatible)
            
        Returns:
            Network instance
        """
        if isinstance(matrix, csr_matrix):
            matrix = matrix.toarray()
        
        graph = nx.from_numpy_array(matrix, create_using=create_using)
        if not isinstance(graph, nx.DiGraph):
            graph = nx.DiGraph(graph) # Ensure graph is directed
            
        return cls(graph, balance_list)
    
    @classmethod
    def init_random(cls, n_nodes: int, random_name: str = '',
                   balance_range: tuple = (0.0, 100.0)) -> 'Network':
        """
        Initialize a random network.
        
        Args:
            n_nodes: Number of nodes
            balance_range: Range for random balances (min, max)
            
        Returns:
            Network instance
        """
        balance_list = np.random.uniform(balance_range[0], balance_range[1], n_nodes).tolist()
        if random_name == 'erdos_renyi':
            # Ensure a directed graph is created
            graph = nx.erdos_renyi_graph(n_nodes, 0.5, directed=True)
        elif random_name == 'barabasi_albert':
            # Use scale_free_graph for a directed equivalent
            graph = nx.scale_free_graph(n_nodes)
        elif random_name == 'watts_strogatz':
            # Create undirected and convert to directed
            graph = nx.watts_strogatz_graph(n_nodes, 4, 0.1)
            graph = nx.DiGraph(graph)
        elif random_name == '':
            m = np.random.randint(0, n_nodes * (n_nodes - 1) // 2)
            graph = generate_random_graph(n_nodes, m) # Assumes plot_utils returns undirected
            graph = nx.DiGraph(graph) # Convert to directed
        else:
            raise ValueError(f"Unknown random graph name: {random_name}")
            
        return cls(graph, balance_list)
    
    def _update_node_order(self) -> None:
        """Update node order for consistent operations."""
        self._node_order = sorted(self.graph.nodes())
    
    def _ensure_balance_consistency(self) -> None:
        """Ensure balance_list matches the number of nodes."""
        n_nodes = self.graph.number_of_nodes()
        n_balances = len(self.balance_list)
        
        if n_balances < n_nodes:
            self.balance_list.extend([0.0] * (n_nodes - n_balances))
        elif n_balances > n_nodes:
            self.balance_list = self.balance_list[:n_nodes]
    
    def to_matrix(self) -> Union[np.ndarray, csr_matrix]:
        """Convert graph to matrix representation."""
        return nx.to_numpy_array(self.graph, nodelist=self._node_order)
    
    def get_graph(self) -> nx.DiGraph:
        """Get the underlying NetworkX DiGraph."""
        return self.graph
    
    def get_info(self) -> Dict[str, Union[List, int, float]]:
        """Get comprehensive information about the network."""
        return {
            'nodes': list(self.graph.nodes()),
            'edges': list(self.graph.edges()),
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'node_order': self._node_order,
            'balance_list': self.balance_list.copy(),
            'total_balance': sum(self.balance_list),
            'avg_balance': np.mean(self.balance_list) if self.balance_list else 0.0,
            'in_degrees': [self.graph.in_degree(node) for node in self._node_order],
            'out_degrees': [self.graph.out_degree(node) for node in self._node_order],
            'is_strongly_connected': nx.is_strongly_connected(self.graph) if self.graph.number_of_nodes() > 0 else False
        }
    
    def add_edge(self, u: int, v: int, **attr) -> None:
        """
        Add a directed edge (u -> v) to the graph and trigger vm.vouch().
        
        Args:
            u: From node (integer ID)
            v: To node (integer ID)
            **attr: Additional edge attributes
        """
        from_address = self.vm.idx_to_address.get(u)
        to_address = self.vm.idx_to_address.get(v)

        # Ensure nodes and their addresses exist before vouching
        if from_address is None:
            self.add_node(u) # This will create the node and its address
            from_address = self.vm.idx_to_address[u]
        if to_address is None:
            self.add_node(v) # This will create the node and its address
            to_address = self.vm.idx_to_address[v]

        try:
            # VouchMinimal handles adding the edge to self.graph
            self.vm.vouch(from_address, to_address)
            
            # Add attributes to the edge if vouch was successful
            if self.graph.has_edge(u, v) and attr:
                 nx.set_edge_attributes(self.graph, {(u, v): attr})
                 
        except ValueError as e:
            print(f"Vouch failed for {u} -> {v}: {e}")
        
        self._update_node_order()
        self._ensure_balance_consistency()
    
    def remove_edge(self, u: int, v: int) -> None:
        """
        Remove a directed edge (u -> v) from the graph and trigger vm.unvouch().
        
        Args:
            u: From node (integer ID)
            v: To node (integer ID)
        """
        if self.graph.has_edge(u, v):
            from_address = self.vm.idx_to_address.get(u)
            to_address = self.vm.idx_to_address.get(v)

            if from_address and to_address:
                try:
                    # VouchMinimal handles removing the edge from self.graph
                    self.vm.unvouch(from_address, to_address)
                except ValueError as e:
                    print(f"Unvouch failed for {u} -> {v}: {e}")
            
            self._update_node_order()
    
    def add_node(self, node: Optional[int] = None, balance: float = 0.0, **attr) -> int:
        """
        Add a node to the graph. If node is None, automatically assign the next available ID.
        This also creates a corresponding mock ETH address for VouchMinimal.
        
        Args:
            node: Integer node ID (optional)
            balance: Initial balance for the node
            **attr: Additional node attributes
            
        Returns:
            The integer node ID
        """
        if node is None:
            node = max(self._node_order) + 1 if self._node_order else 0
        
        if node not in self.graph:
            self.graph.add_node(node, **attr)
            self._update_node_order()
            insertion_idx = self._node_order.index(node)
            self.balance_list.insert(insertion_idx, balance)
            
            # --- VouchMinimal Integration ---
            # Create address and state for the new node
            if node not in self.vm.idx_to_address:
                new_address = generate_mul_eth_addresses(1)[0]
                self.vm.idx_to_address[node] = new_address
                self.vm.address_to_idx[new_address] = node
                self.vm.nodes[new_address] = Node() # Create new Node state
            # --- End Integration ---
        else:
            # Node already exists, just update balance
            idx = self._node_order.index(node)
            self.balance_list[idx] = balance

        self._ensure_balance_consistency()
        return node
    
    def remove_node(self, node: int) -> None:
        """
        Remove a node and all its associated vouches (in and out) from the graph.
        
        Args:
            node: Integer node ID
        """
        if node in self.graph:
            address = self.vm.idx_to_address.get(node)
            
            # Unvouch all incoming edges
            # Must copy list as self.remove_edge modifies the graph
            in_edges = list(self.graph.in_edges(node))
            for u, v in in_edges:
                self.remove_edge(u, v) # This calls vm.unvouch()

            # Unvouch all outgoing edges
            # Must copy list as self.remove_edge modifies the graph
            out_edges = list(self.graph.out_edges(node))
            for u, v in out_edges:
                self.remove_edge(u, v) # This calls vm.unvouch()
            
            # Now remove the node from balance list
            if node in self._node_order:
                idx = self._node_order.index(node)
                self.balance_list.pop(idx)
            
            # Remove node from NetworkX graph
            # This is safe now as all edges are gone
            self.graph.remove_node(node)
            
            # Clean up VouchMinimal state
            if address:
                if address in self.vm.address_to_idx:
                    self.vm.address_to_idx.pop(address)
                if node in self.vm.idx_to_address:
                    self.vm.idx_to_address.pop(node)
                if address in self.vm.nodes:
                    self.vm.nodes.pop(address)
            
            self._update_node_order()
            self._ensure_balance_consistency()
    
    def compute_score(self, **kwargs) -> List[float]:
        """
        Get current node scores from the VouchMinimal algorithm.
        
        The scores are computed automatically on vouch/unvouch (add_edge/remove_edge).
        This method retrieves the current scores.
            
        Returns:
            List of scores for each node, in node order
        """
        scores = []
        for node_idx in self._node_order:
            address = self.vm.idx_to_address.get(node_idx)
            if address:
                score = self.vm.get_score(address)
                scores.append(float(score))
            else:
                scores.append(0.0)
        return scores
    
    def get_ranks(self) -> List[int]:
        """
        Get current node ranks from the VouchMinimal algorithm.
            
        Returns:
            List of ranks for each node, in node order
        """
        ranks = []
        for node_idx in self._node_order:
            address = self.vm.idx_to_address.get(node_idx)
            if address:
                rank = self.vm.get_rank(address)
                ranks.append(rank)
            else:
                ranks.append(self.vm.DEFAULT_RANK) # Use default rank
        return ranks

    def set_balance(self, node: int, balance: float) -> None:
        """Set balance for a specific node."""
        if node in self._node_order:
            idx = self._node_order.index(node)
            self.balance_list[idx] = balance
        else:
            self.add_node(node, balance=balance)
    
    def get_balance_list(self) -> List[float]:
        """Get balance list."""
        return self.balance_list

    def display_graph(self, scores: Optional[List[float]] = None, title: str = "Network Graph") -> None:
        """
        Visualizes the network graph with nodes colored by score and sized by balance.
        """
        if self.graph.number_of_nodes() == 0:
            print("Network is empty.")
            return

        pos = nx.spring_layout(self.graph, seed=42)
        
        if scores is None:
            # If no scores provided, compute them
            scores = self.compute_score()
        
        # Normalize scores for better visualization if they are very large
        scores_array = np.array(scores)
        if np.max(scores_array) > 0:
             vis_scores = scores_array / np.max(scores_array)
        else:
             vis_scores = scores_array
        
        plot_graph_with_scores(self.graph, pos, vis_scores, title)

    def __repr__(self) -> str:
        """String representation of the network."""
        return f"Network(nodes={self.graph.number_of_nodes()}, edges={self.graph.number_of_edges()}, total_balance={sum(self.balance_list):.2f})"
    
    def __len__(self) -> int:
        """Number of nodes in the network."""
        return self.graph.number_of_nodes()
    
    def __contains__(self, node: int) -> bool:
        """Check if node is in the network."""
        return node in self.graph


# Example usage
if __name__ == "__main__":
    # Import utility files needed for the example
    # Ensure contract_interface_mvp.py, utils.py, and plot_utils.py are present
    
    # Create a random network
    net = Network.init_random(10, balance_range=(10, 100), random_name='erdos_renyi')
    print(f"Network: {net}")
    
    # Compute scores (they are already computed by __init__ due to vouches)
    initial_scores = net.compute_score()
    initial_ranks = net.get_ranks()
    print(f"Initial Scores: {initial_scores}")
    print(f"Initial Ranks: {initial_ranks}")
    
    # Display graph with scores
    net.display_graph(scores=initial_scores, title="Scores on Random Network")
    
    # Test modifications
    net.add_node(10, balance=50.0)
    net.add_edge(0, 10) # This will trigger vm.vouch() and recompute scores
    net.set_balance(0, 200.0)
    print(f"\nAfter modifications: {net}")
    
    new_scores = net.compute_score()
    new_ranks = net.get_ranks()
    print(f"New Scores: {new_scores}")
    print(f"New Ranks: {new_ranks}")
    net.display_graph(scores=new_scores, title="After Modifying Network")
    
    # Test node removal
    net.remove_node(0)
    print(f"\nAfter removing node 0: {net}")
    final_scores = net.compute_score()
    print(f"Final Scores: {final_scores}")
    net.display_graph(scores=final_scores, title="After Removing Node 0")

