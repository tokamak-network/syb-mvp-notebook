# pagerank_scoring_algorithm.py

import networkx as nx
import numpy as np
import itertools

def normalize_scores(scores):
    """
    Normalizes a vector of scores so they sum to 1.
    """
    score_sum = np.sum(scores)
    if score_sum > 0:
        return scores / score_sum
    return np.zeros_like(scores)

def compute_pagerank_series(graph, alpha, source_node, max_iterations=100):
    """
    Computes a personalized PageRank vector using the correct
    infinite series expansion formula.
    """
    n = graph.number_of_nodes()
    if n == 0:
        return np.array([])
    
    nodes = sorted(list(graph.nodes()))
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # Create the personalization vector 's'
    s = np.zeros(n)
    if source_node in node_to_idx:
        s[node_to_idx[source_node]] = 1.0

    # Get the adjacency matrix A
    A = nx.to_numpy_array(graph, nodelist=nodes)

    # Get degrees
    degrees = np.array([graph.degree(node) for node in nodes])

    # Create the correct transition matrix W
    W = np.zeros((n, n))
    for i in range(n):
        if degrees[i] > 0:
            W[i, :] = A[i, :] / degrees[i]

    # Compute the infinite series sum
    pr_vector_sum = np.zeros(n)
    s_W_t = s.copy() # Represents the term s * W^t

    for t in range(max_iterations):
        pr_vector_sum += ((1 - alpha) ** t) * s_W_t
        s_W_t = s_W_t @ W # Matrix multiplication for the next term

    pr_vector = alpha * pr_vector_sum

    return pr_vector

def compute_next_scores(graph, current_scores=None, alpha=0.15, sigma=1.0, max_pr_iterations=100):
    """
    Computes new scores based on Personalized PageRank.
    """
    n = graph.number_of_nodes()
    if n == 0:
        return []
        
    nodes = sorted(list(graph.nodes()))
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    if current_scores is None or len(current_scores) != n:
        x = np.ones(n) / n
    else:
        x = np.array(current_scores)
    
    total_score_sum = np.sum(x)
    y = np.zeros(n)

    # Compute a score for each vertex
    for i, v in enumerate(nodes):
        # 1. Compute personalized pagerank vector for vertex v
        # Use the series expansion algorithm
        pr_vector = compute_pagerank_series(graph, alpha, v, max_pr_iterations)
        # #   Or use the power iteration method:
        # pr_vector = compute_pagerank_power_iterations(graph, alpha, v, max_pr_iterations)
        
        # 2. Get the permutation based on descending pagerank scores
        sorted_indices = np.argsort(pr_vector)[::-1]
        sorted_nodes = [nodes[idx] for idx in sorted_indices]
        
        min_ratio_for_v = float('inf')
        
        # 3. Iterate through subsets Sj
        for j in range(1, n + 1):
            Sj = set(sorted_nodes[:j])
            
            # Condition 1: i must be in Sj
            if v not in Sj:
                continue
                
            # Condition 2: Sum of scores in Sj < 1/2 of total score sum
            subset_score_sum = np.sum([x[node_to_idx[node]] for node in Sj])
            if subset_score_sum >= total_score_sum / 2:
                continue
                
            # If conditions are met, calculate the ratio
            num_boundary_edges = len(list(nx.edge_boundary(graph, Sj)))
            size_Sj = len(Sj)
            
            if size_Sj > 0:
                current_ratio = (num_boundary_edges ** sigma) / size_Sj
                min_ratio_for_v = min(min_ratio_for_v, current_ratio)
                
        y[i] = 0 if min_ratio_for_v == float('inf') else min_ratio_for_v

    return normalize_scores(y)


def compute_next_scores_unnorm(graph, current_scores=None, alpha=0.15, sigma=1.0, max_pr_iterations=100):
    """
    Computes new scores based on Personalized PageRank without normalization.
    """
    n = graph.number_of_nodes()
    if n == 0:
        return []

    nodes = sorted(list(graph.nodes()))
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    if current_scores is None or len(current_scores) != n:
        x = np.ones(n) / n
    else:
        x = np.array(current_scores)
    
    total_score_sum = np.sum(x)
    y = np.zeros(n)

    # Compute a score for each vertex
    for i, v in enumerate(nodes):
        # 1. Compute personalized pagerank vector for vertex v
        # Use the corrected series expansion algorithm
        pr_vector = compute_pagerank_series(graph, alpha, v, max_pr_iterations)
        # #   Or use the power iteration method:
        # pr_vector = compute_pagerank_power_iterations(graph, alpha, v, max_pr_iterations)
        
        # 2. Get the permutation based on descending pagerank scores
        sorted_indices = np.argsort(pr_vector)[::-1]
        sorted_nodes = [nodes[idx] for idx in sorted_indices]
        
        min_ratio_for_v = float('inf')
        
        # 3. Iterate through subsets Sj
        for j in range(1, n + 1):
            Sj = set(sorted_nodes[:j])
            
            # Condition 1: i must be in Sj
            if v not in Sj:
                continue
                
            # Condition 2: Sum of scores in Sj < 1/2 of total score sum
            subset_score_sum = np.sum([x[node_to_idx[node]] for node in Sj])
            if subset_score_sum >= total_score_sum / 2:
                continue
                
            # If conditions are met, calculate the ratio
            num_boundary_edges = len(list(nx.edge_boundary(graph, Sj)))
            size_Sj = len(Sj)
            
            if size_Sj > 0:
                current_ratio = (num_boundary_edges ** sigma) / size_Sj
                min_ratio_for_v = min(min_ratio_for_v, current_ratio)
                
        y[i] = 0 if min_ratio_for_v == float('inf') else min_ratio_for_v

    return y