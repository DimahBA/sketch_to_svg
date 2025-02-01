import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from topological_graph import Node, Edge

@dataclass
class BezierCurve:
    """Represents a Bézier curve with its control points"""
    degree: int  # Will be 3 for cubic curves
    control_points: np.ndarray  # Shape: (4, 2) for cubic curves
    start_node: Node
    end_node: Node
    error: float

def fit_bezier_curves(edges: List[Edge], max_error: float = 2.0) -> List[BezierCurve]:
    """Fit cubic Bézier curves to the edges of the topological graph"""
    if not edges:
        return []
        
    bezier_curves = []
    for edge in edges:
        curve = fit_cubic_bezier(edge)
        bezier_curves.append(curve)
                
    return bezier_curves

def fit_cubic_bezier(edge: Edge) -> BezierCurve:
    """
    Fit a cubic Bézier curve to an edge.
    Uses edge pixels to determine control points positioning.
    """
    points = np.array(edge.pixels)
    n_points = len(points)
    
    if n_points < 2:
        # If not enough points, create a simple line
        control_points = np.array([
            [edge.start_node.x, edge.start_node.y],
            [edge.start_node.x, edge.start_node.y],
            [edge.end_node.x, edge.end_node.y],
            [edge.end_node.x, edge.end_node.y]
        ])
        return BezierCurve(3, control_points, edge.start_node, edge.end_node, 0.0)
    
    # Start and end points
    p0 = np.array([edge.start_node.x, edge.start_node.y])
    p3 = np.array([edge.end_node.x, edge.end_node.y])
    
    # Use points at 1/3 and 2/3 along the path to guide control points
    one_third_idx = n_points // 3
    two_thirds_idx = (2 * n_points) // 3
    
    # Get direction vectors
    v_start = points[one_third_idx] - points[0]
    v_end = points[-1] - points[two_thirds_idx]
    
    # Scale vectors to 1/3 of the path length
    path_length = np.linalg.norm(p3 - p0) / 3
    if np.linalg.norm(v_start) > 0:
        v_start = v_start * path_length / np.linalg.norm(v_start)
    if np.linalg.norm(v_end) > 0:
        v_end = v_end * path_length / np.linalg.norm(v_end)
    
    # Control points
    p1 = p0 + v_start
    p2 = p3 - v_end
    
    control_points = np.vstack([p0, p1, p2, p3])
    
    # Compute fitting error
    error = compute_fitting_error(points, control_points)
    
    return BezierCurve(3, control_points, edge.start_node, edge.end_node, error)

def compute_fitting_error(points: np.ndarray, control_points: np.ndarray) -> float:
    """Compute the fitting error between points and a cubic Bézier curve"""
    # Sample curve points
    t_values = np.linspace(0, 1, len(points))
    curve_points = np.array([evaluate_bezier(control_points, 3, t) for t in t_values])
    
    # Compute average distance between curve and points
    errors = np.sqrt(np.sum((points - curve_points) ** 2, axis=1))
    return np.mean(errors)

def evaluate_bezier(control_points: np.ndarray, degree: int, t: float) -> np.ndarray:
    """Evaluate a Bézier curve at parameter value t"""
    if degree != 3:
        raise ValueError("Only cubic Bézier curves are supported")
        
    # Cubic Bézier formula
    p0, p1, p2, p3 = control_points
    return (1-t)**3 * p0 + 3*(1-t)**2*t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3