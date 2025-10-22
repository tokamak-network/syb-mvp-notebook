"""
Network class for graph operations with balance tracking and scoring algorithms.

This class manages a graph with node balances and provides various scoring algorithms.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Union
from scipy.sparse import csr_matrix
from comparison import generate_random_graph, plot_graph_with_scores

# Import the four scoring functions
from initial_scores import compute_initial_scores
from random_walk_scoring_algorithm import (
    compute_next_scores as compute_random_walk,
)
from pagerank_scoring_algorithm import (
    compute_next_scores as compute_pagerank
)
from equal_split_scoring_algorithm import (
    compute_next_scores as compute_equal_split
)
from argmax_scoring_algorithm import (
    compute_next_scores as compute_argmax
)


class Network:
    """
    A network class that manages a graph with node balances and scoring algorithms.
    
    The graph represents the network structure, and balance_list tracks the balance
    of each node. Various scoring algorithms can be applied to compute node scores.
    """
    
    def __init__(self, graph: Optional[nx.Graph] = None, balance_list: Optional[List[float]] = None):
        """
        Initialize the network.
        
        Args:
            graph: NetworkX graph (optional)
            balance_list: List of balances for each node (optional)
        """
        self.graph = graph or nx.Graph()
        self.balance_list = balance_list or []
        self.config = {
            'SIGMA_EQUAL_SPLIT': 2.0,
            'SIGMA_ARGMAX': 2.0,
            'SIGMA_PAGERANK': 2.0,
            'ALPHA_PAGERANK': 0.15,
            'MAX_PR_ITERATIONS': 100,
        }
        self._node_order = []
        self._update_node_order()
        self._ensure_balance_consistency()
    
    @classmethod
    def from_matrix(cls, matrix: Union[np.ndarray, csr_matrix], 
                   balance_list: Optional[List[float]] = None,
                   create_using: type = nx.Graph) -> 'Network':
        """
        Create network from adjacency matrix.
        
        Args:
            matrix: Adjacency matrix
            balance_list: List of balances for each node (optional)
            create_using: Graph class to create
            
        Returns:
            Network instance
        """
        if isinstance(matrix, csr_matrix):
            matrix = matrix.toarray()
        
        graph = nx.from_numpy_array(matrix, create_using=create_using)
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
            graph = nx.erdos_renyi_graph(n_nodes, 0.5)
        elif random_name == 'barabasi_albert':
            graph = nx.barabasi_albert_graph(n_nodes, 1)
        elif random_name == 'watts_strogatz':
            graph = nx.watts_strogatz_graph(n_nodes, 1, 0.5)
        elif random_name == '':
            m = np.random.randint(0, n_nodes * (n_nodes - 1) // 2)
            graph = generate_random_graph(n_nodes, m)
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
    
    def get_graph(self) -> nx.Graph:
        """Get the underlying NetworkX graph."""
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
            'degrees': [self.graph.degree(node) for node in self._node_order],
            'is_connected': nx.is_connected(self.graph) if self.graph.number_of_nodes() > 0 else False
        }
    
    def add_edge(self, u: int, v: int, **attr) -> None:
        """Add an edge to the graph."""
        self.graph.add_edge(u, v, **attr)
        self._update_node_order()
        self._ensure_balance_consistency()
    
    def remove_edge(self, u: int, v: int) -> None:
        """Remove an edge from the graph."""
        if self.graph.has_edge(u, v):
            self.graph.remove_edge(u, v)
            self._update_node_order()
    
    def add_node(self, node: Optional[int] = None, balance: float = 0.0, **attr) -> int:
        """Add a node to the graph. If node is None, automatically assign the next available ID."""
        if node is None:
            node = max(self._node_order) + 1 if self._node_order else 0
        
        if node not in self.graph:
            self.graph.add_node(node, **attr)
            self._update_node_order()
            insertion_idx = self._node_order.index(node)
            self.balance_list.insert(insertion_idx, balance)
        else:
            idx = self._node_order.index(node)
            self.balance_list[idx] = balance

        self._ensure_balance_consistency()
        return node
    
    def remove_node(self, node: int) -> None:
        """Remove a node from the graph."""
        if node in self.graph:
            if node in self._node_order:
                idx = self._node_order.index(node)
                self.balance_list.pop(idx)
            
            self.graph.remove_node(node)
            self._update_node_order()
            self._ensure_balance_consistency()
    
    def compute_score(self, algo_name: str, **kwargs) -> List[float]:
        """
        Compute node scores using specified algorithm.
        
        Args:
            algo_name: Name of the scoring algorithm
            **kwargs: Additional parameters for the algorithm
            
        Returns:
            List of scores for each node in node order
        """
        G = self.graph
        N_VERTICES = G.number_of_nodes()

        fallback_scores = np.ones(N_VERTICES) / N_VERTICES if N_VERTICES > 0 else np.array([])
        previous_scores = kwargs.get('previous_scores', None)
        input_scores = previous_scores if previous_scores is not None else fallback_scores

        if algo_name == 'random_walk':
            return self._compute_random_walk(G, input_scores)
        elif algo_name == 'pagerank':
            return self._compute_pagerank(G, input_scores)
        elif algo_name == 'equal_split':
            return self._compute_equal_split(G, input_scores)
        elif algo_name == 'argmax':
            return self._compute_argmax(G, input_scores)
        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")
    
    def _compute_random_walk(self, G, input_scores) -> List[float]:
        """Compute scores using random walk algorithm."""
        return compute_random_walk(G, input_scores)
    
    def _compute_pagerank(self, G, input_scores) -> List[float]:
        """Compute scores using PageRank algorithm."""
        return compute_pagerank(G, input_scores,
                                alpha=self.config['ALPHA_PAGERANK'],
                                sigma=self.config['SIGMA_PAGERANK'],
                                max_pr_iterations=self.config['MAX_PR_ITERATIONS'])

    def _compute_equal_split(self, G, input_scores) -> List[float]:
        """Compute scores using equal split algorithm."""
        return compute_equal_split(G, input_scores, sigma=self.config['SIGMA_EQUAL_SPLIT'])
    
    def _compute_argmax(self, G, input_scores) -> List[float]:
        """Compute scores using argmax algorithm."""
        return compute_argmax(G, input_scores, sigma=self.config['SIGMA_ARGMAX'])

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
            scores = np.ones(len(self._node_order)) / len(self._node_order)
        
        plot_graph_with_scores(self.graph, pos, scores, title)

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
    # Create a random network
    net = Network.init_random(10, balance_range=(10, 100), random_name='erdos_renyi')
    print(f"Network: {net}")
    
    # Compute scores
    pagerank_scores = net.compute_score('pagerank')
    print(f"PageRank Scores: {pagerank_scores}")
    
    # Display graph with scores
    net.display_graph(scores=pagerank_scores, title="PageRank on Random Network")
    
    # Test modifications
    net.add_node(10, balance=50.0)
    net.add_edge(0, 10)
    net.set_balance(0, 200.0)
    print(f"\nAfter modifications: {net}")
    
    new_scores = net.compute_score('pagerank')
    net.display_graph(scores=new_scores, title="After Modifying Network")
