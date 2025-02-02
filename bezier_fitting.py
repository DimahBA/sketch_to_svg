from dataclasses import dataclass
import numpy as np
from typing import List, Tuple, Optional
from scipy.optimize import minimize
from scipy.interpolate import splprep, splev

@dataclass
class BezierCurve:
    control_points: np.ndarray
    degree: int
    error: float

def fit_bezier_curve(pixels: List[Tuple[int, int]], degree: int) -> BezierCurve:
    """Simplified and faster curve fitting using B-spline interpolation"""
    if len(pixels) < 2:
        raise ValueError("Need at least 2 pixels to fit a curve")
    
    if len(pixels) < degree + 1:
        degree = len(pixels) - 1
    
    try:
        # Convert pixels to numpy array
        points = np.array(pixels)
        
        # For straight lines, just use endpoints
        if degree == 1:
            control_points = np.array([points[0], points[-1]])
            error = np.mean(np.sum((points - control_points[0])**2, axis=1))
            return BezierCurve(control_points=control_points, degree=1, error=error)
        
        # For higher degrees, use spline interpolation
        x = points[:, 0]
        y = points[:, 1]
        
        # Fit a B-spline
        t = np.linspace(0, 1, len(x))
        tck, _ = splprep([x, y], k=degree, s=0)
        
        # Sample control points
        num_control_points = degree + 1
        u = np.linspace(0, 1, num_control_points)
        control_points = np.column_stack(splev(u, tck))
        
        # Calculate error
        curve_points = np.column_stack(splev(t, tck))
        error = np.mean(np.sum((points - curve_points)**2, axis=1))
        
        return BezierCurve(
            control_points=control_points,
            degree=degree,
            error=error
        )
        
    except Exception as e:
        # Fallback to linear fit
        control_points = np.array([points[0], points[-1]])
        return BezierCurve(control_points=control_points, degree=1, error=float('inf'))