from dataclasses import dataclass
from typing import List, Set, Tuple, Optional
import numpy as np
from bezier_fitting import fit_bezier_curve, BezierCurve
from topological_graph import Edge, Node

@dataclass
class Hyperedge:
    """Represents a hyperedge that groups multiple edges from the original graph"""
    edges: List[Edge]  # Original graph edges that form this hyperedge
    bezier_curve: Optional[BezierCurve] = None  # Fitted Bezier curve
    shared_edges: Set[Edge] = None  # Edges shared with other hyperedges
    
    def __post_init__(self):
        if self.shared_edges is None:
            self.shared_edges = set()
            
    def get_pixel_chain(self) -> List[Tuple[int, int]]:
        """Get ordered list of pixels from all edges in hyperedge"""
        pixels = []
        for edge in self.edges:
            if edge == self.edges[0]:  # First edge
                pixels.extend(edge.pixels)
            else:  # Subsequent edges - avoid duplicating connection point
                pixels.extend(edge.pixels[1:])
        return pixels

class Hypergraph:
    """Implementation of the paper's hypergraph representation and optimization"""
    def __init__(self, nodes: List[Node], edges: List[Edge], lambda_param=0.9, mu=0.7):
        self.nodes = nodes
        self.original_edges = edges
        self.lambda_param = lambda_param  # Controls fidelity vs simplicity
        self.mu = mu  # Controls curve degree penalty
        
        # Initialize each edge as its own hyperedge
        self.hyperedges = [Hyperedge([edge]) for edge in self.original_edges]
        self.fit_bezier_curves()  # Initial curve fitting
        
    def fit_bezier_curves(self):
        """Fit Bezier curves to all hyperedges"""
        for hedge in self.hyperedges:
            pixels = hedge.get_pixel_chain()
            # Try fitting curves of different degrees, choose best
            best_error = float('inf')
            best_curve = None
            best_degree = 3
            
            for degree in [1, 2, 3]:
                curve = fit_bezier_curve(pixels, degree)
                if curve.error < best_error:
                    best_error = curve.error
                    best_curve = curve
                    best_degree = degree
            
            hedge.bezier_curve = best_curve
            
    def calculate_energy(self) -> float:
        """Calculate total energy U(x) = (1-λ)U_fidelity + λU_simplicity"""
        # Fidelity term: sum of fitting errors
        fidelity = sum(h.bezier_curve.error for h in self.hyperedges)
        
        # Simplicity term: number of curves + degree penalty
        simplicity = sum(1 + self.mu * h.bezier_curve.degree for h in self.hyperedges)
        
        return (1 - self.lambda_param) * fidelity + self.lambda_param * simplicity
        
    
    def optimize(self, T_init=1.0, T_end=1e-4, callback=None, max_iter=3000):
        """Optimized Metropolis-Hastings optimization loop"""
        T = T_init
        C = np.power(0.999, 1.0/len(self.nodes))
        
        iteration = 0
        best_energy = self.calculate_energy()
        best_state = self.hyperedges.copy()
        no_improvement_count = 0
        print(f"Starting optimization with {len(self.hyperedges)} hyperedges")
        print(f"Initial energy: {best_energy:.2f}")
        
        operations = ['merge_split', 'degree_switch', 'overlap']
        op_weights = [0.7, 0.2, 0.1]  # Heavy bias towards merging
        
        while T > T_end and iteration < max_iter and no_improvement_count < 100:
            iteration += 1
            
            if iteration % 50 == 0:
                current_energy = self.calculate_energy()
                print(f"Iteration {iteration}, T={T:.6f}, Energy={current_energy:.2f}, Hyperedges={len(self.hyperedges)}")
            
            # Choose operation based on weights
            op = np.random.choice(operations, p=op_weights)
            
            try:
                # Create candidate solution
                if op == 'merge_split':
                    candidate = self.apply_merge_split()
                elif op == 'degree_switch':
                    candidate = self.apply_degree_switch()
                else:
                    candidate = self.apply_overlap()
                
                if candidate is None:
                    continue
                
                # Calculate energy change
                candidate_energy = candidate.calculate_energy()
                delta_E = candidate_energy - best_energy
                
                # Modified acceptance criterion to favor merging
                acceptance_prob = np.exp(-delta_E / T)
                if len(candidate.hyperedges) < len(self.hyperedges):
                    acceptance_prob *= 1.5  # Boost probability for merging
                
                # Accept or reject
                if delta_E < 0 or np.random.random() < acceptance_prob:
                    self = candidate
                    if candidate_energy < best_energy:
                        best_energy = candidate_energy
                        best_state = self.hyperedges.copy()
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                
            except Exception as e:
                print(f"Error in iteration {iteration}: {str(e)}")
                continue
            
            T *= C
            
            if callback and iteration % 10 == 0:
                callback(self, T)
        
        # Restore best state found
        self.hyperedges = best_state
        
        print(f"\nOptimization finished after {iteration} iterations")
        print(f"Final hyperedges: {len(self.hyperedges)}, Energy: {best_energy:.2f}")

    def are_edges_connected(self, edges1: List[Edge], edges2: List[Edge]) -> bool:
        """Check if two groups of edges can be connected"""
        if not edges1 or not edges2:
            return False
            
        # Check if last node of edges1 is connected to first node of edges2
        return (edges1[-1].end_node == edges2[0].start_node or 
                edges1[-1].end_node == edges2[0].end_node or
                edges1[-1].start_node == edges2[0].start_node or
                edges1[-1].start_node == edges2[0].end_node)

    def apply_merge_split(self) -> Optional['Hypergraph']:
        """Merge two consecutive hyperedges or split one hyperedge"""
        if len(self.hyperedges) < 2:
            return None
            
        if np.random.random() < 0.8:  # Bias towards merging
            # Try to find mergeable hyperedges
            for _ in range(10):  # Try multiple times to find mergeable edges
                idx1 = np.random.randint(len(self.hyperedges))
                h1 = self.hyperedges[idx1]
                
                # Find all potential merge candidates
                candidates = []
                for idx2, h2 in enumerate(self.hyperedges):
                    if idx2 != idx1 and self.are_edges_connected(h1.edges, h2.edges):
                        candidates.append((idx2, h2))
                
                if candidates:
                    # Choose the candidate that results in the lowest error
                    best_candidate = None
                    best_error = float('inf')
                    best_idx = None
                    
                    for idx2, h2 in candidates:
                        # Try merging
                        merged_edges = h1.edges + h2.edges
                        merged_hedge = Hyperedge(merged_edges)
                        
                        # Fit curve and check error
                        pixels = merged_hedge.get_pixel_chain()
                        curve = fit_bezier_curve(pixels, 3)
                        if curve.error < best_error:
                            best_error = curve.error
                            best_candidate = merged_hedge
                            best_idx = idx2
                    
                    if best_candidate and best_error < h1.bezier_curve.error + self.hyperedges[best_idx].bezier_curve.error:
                        # Create new hypergraph with merged edge
                        new_hyperedges = [h for i, h in enumerate(self.hyperedges) 
                                        if i != idx1 and i != best_idx]
                        new_hyperedges.append(best_candidate)
                        
                        new_graph = Hypergraph(self.nodes, 
                                             self.original_edges, 
                                             self.lambda_param, 
                                             self.mu)
                        new_graph.hyperedges = new_hyperedges
                        new_graph.fit_bezier_curves()
                        return new_graph
        else:  # Split
            idx = np.random.randint(len(self.hyperedges))
            hedge = self.hyperedges[idx]
            if len(hedge.edges) > 1:
                # Try different split points and choose the best one
                best_split = None
                best_error = float('inf')
                
                for split_idx in range(1, len(hedge.edges)):
                    h1 = Hyperedge(hedge.edges[:split_idx])
                    h2 = Hyperedge(hedge.edges[split_idx:])
                    
                    # Calculate total error after split
                    pixels1 = h1.get_pixel_chain()
                    pixels2 = h2.get_pixel_chain()
                    curve1 = fit_bezier_curve(pixels1, min(3, len(pixels1)-1))
                    curve2 = fit_bezier_curve(pixels2, min(3, len(pixels2)-1))
                    total_error = curve1.error + curve2.error
                    
                    if total_error < best_error:
                        best_error = total_error
                        best_split = (h1, h2)
                
                if best_split and best_error < hedge.bezier_curve.error:
                    h1, h2 = best_split
                    new_hyperedges = [h for i, h in enumerate(self.hyperedges) if i != idx]
                    new_hyperedges.extend([h1, h2])
                    
                    new_graph = Hypergraph(self.nodes,
                                         self.original_edges,
                                         self.lambda_param, 
                                         self.mu)
                    new_graph.hyperedges = new_hyperedges
                    new_graph.fit_bezier_curves()
                    return new_graph
                    
        return None

    def apply_degree_switch(self) -> Optional['Hypergraph']:
        """Change the degree of a random Bezier curve"""
        if not self.hyperedges:
            return None
            
        # Select random hyperedge
        idx = np.random.randint(len(self.hyperedges))
        current_degree = self.hyperedges[idx].bezier_curve.degree
        
        # Select new degree (different from current)
        possible_degrees = [1, 2, 3]
        possible_degrees.remove(current_degree)
        new_degree = np.random.choice(possible_degrees)
        
        # Create new hypergraph with changed degree
        new_graph = Hypergraph(self.nodes,
                              self.original_edges,
                              self.lambda_param, 
                              self.mu)
        new_graph.hyperedges = self.hyperedges.copy()
        pixels = new_graph.hyperedges[idx].get_pixel_chain()
        new_graph.hyperedges[idx].bezier_curve = fit_bezier_curve(pixels, new_degree)
        
        return new_graph

    def apply_overlap(self) -> Optional['Hypergraph']:
        """Create or remove edge sharing between hyperedges"""
        if len(self.hyperedges) < 2:
            return None
            
        # Select two random hyperedges
        idx1, idx2 = np.random.choice(len(self.hyperedges), 2, replace=False)
        h1, h2 = self.hyperedges[idx1], self.hyperedges[idx2]
        
        # Find potential shared edges
        shared = set(h1.edges) & set(h2.edges)
        
        if shared:  # Remove sharing
            new_graph = Hypergraph(self.nodes,
                                 self.original_edges,
                                 self.lambda_param, 
                                 self.mu)
            new_graph.hyperedges = self.hyperedges.copy()
            edge = shared.pop()
            new_graph.hyperedges[idx2].edges.remove(edge)
            new_graph.hyperedges[idx2].shared_edges.remove(edge)
            new_graph.fit_bezier_curves()
            return new_graph
            
        else:  # Try to create sharing
            # Find edges that could potentially be shared
            for e1 in h1.edges:
                for e2 in h2.edges:
                    if (e1.start_node == e2.start_node and 
                        e1.end_node == e2.end_node):
                        new_graph = Hypergraph(self.nodes,
                                             self.original_edges,
                                             self.lambda_param, 
                                             self.mu)
                        new_graph.hyperedges = self.hyperedges.copy()
                        new_graph.hyperedges[idx2].edges.append(e1)
                        new_graph.hyperedges[idx2].shared_edges.add(e1)
                        new_graph.fit_bezier_curves()
                        return new_graph
                        
        return None