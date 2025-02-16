import numpy as np
from dataclasses import dataclass
from typing import List
from topological_graph import Node, Edge

@dataclass
class BezierCurve:
    """Represents a Bézier curve with its control points and associated nodes"""
    control_points: np.ndarray  # Shape: (2, 2) for linear or (4, 2) for cubic Bézier
    start_node: Node
    end_node: Node

def curve_length(points: np.ndarray) -> float:
    """Calculate the approximate length of a curve from its points"""
    return np.sum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))

def fit_cubic_bezier(points: np.ndarray) -> np.ndarray:
    """Fit a cubic Bézier curve to a sequence of points.
    
    This function fits a cubic Bézier curve using chord-length parameterization
    and least squares fitting. The first and last control points are fixed at
    the endpoints of the curve.
    
    Args:
        points: Array of points to fit (shape: [n, 2])
        
    Returns:
        Array of control points (shape: [4, 2])
    """
    # We need at least 2 points to fit a curve
    if len(points) < 2:
        return np.vstack([points[0], points[0], points[-1], points[-1]])
    
    # Calculate chord lengths for parameterization
    chord_lengths = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    t_values = np.zeros(len(points))
    t_values[1:] = np.cumsum(chord_lengths)
    t_values /= t_values[-1]  # Normalize to [0, 1]
    
    # Build the cubic Bézier basis matrix
    basis_matrix = np.zeros((len(points), 4))
    for i, t in enumerate(t_values):
        basis_matrix[i] = [
            (1-t)**3,           # Basis function for P0
            3*t*(1-t)**2,       # Basis function for P1
            3*t**2*(1-t),       # Basis function for P2
            t**3                # Basis function for P3
        ]
    
    # First and last control points are the endpoints
    P0 = points[0]
    P3 = points[-1]
    
    # Solve for the middle control points
    A = basis_matrix[:, 1:3]
    b = points - np.outer(basis_matrix[:, 0], P0) - np.outer(basis_matrix[:, 3], P3)
    
    # Solve least squares problem for middle control points
    P1P2, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    # Combine all control points
    control_points = np.vstack([P0, P1P2[0], P1P2[1], P3])
    
    return control_points

def fit_bezier_curves(edges: List[Edge], min_curve_length: float = 10.0) -> List[BezierCurve]:
    """Fit Bézier curves to the edges of the topological graph.
    
    This function creates a Bézier curve for each edge in the graph,
    preserving the topological connections between nodes. For short edges,
    it creates linear Bézier curves (straight lines) instead of cubic curves.
    
    Args:
        edges: List of Edge objects from the topological graph
        min_curve_length: Minimum length threshold for cubic curves. Shorter curves
                         will be rendered as straight lines.
        
    Returns:
        List of fitted BezierCurve objects
    """
    bezier_curves = []
    
    for edge in edges:
        # Convert pixel coordinates to numpy array
        points = np.array(edge.pixels)
        
        # Calculate curve length
        length = curve_length(points)
        
        if length < min_curve_length:
            # For short curves, just use start and end points (linear Bézier)
            control_points = np.vstack([points[0], points[-1]])
        else:
            # For longer curves, fit a cubic Bézier
            control_points = fit_cubic_bezier(points)
        
        # Create BezierCurve object
        curve = BezierCurve(
            control_points=control_points,
            start_node=edge.start_node,
            end_node=edge.end_node
        )
        bezier_curves.append(curve)
    
    return bezier_curves