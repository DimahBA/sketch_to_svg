from typing import List
from pathlib import Path
from bezier_fitting import BezierCurve

def generate_svg(curves: List[BezierCurve], width: int, height: int) -> str:
    """Generate an SVG string from a list of Bézier curves.
    
    Args:
        curves: List of BezierCurve objects with either 2 control points (linear)
               or 4 control points (cubic)
        width: Width of the SVG viewport
        height: Height of the SVG viewport
        
    Returns:
        String containing the SVG XML
    """
    # Start SVG document with viewBox
    svg = f'<?xml version="1.0" encoding="UTF-8"?>\n'
    svg += f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">\n'
    
    # Add style for paths
    svg += ' <style>\n'
    svg += ' path { fill: none; stroke: black; stroke-width: 2; }\n'
    svg += ' </style>\n'
    
    # Create path for each Bézier curve
    for curve in curves:
        points = curve.control_points
        
        # Check if it's a linear or cubic Bézier curve
        if len(points) == 2:
            # Linear Bézier (straight line) using SVG line command: L = lineto
            path = f' <path d="M {points[0][0]:.1f},{points[0][1]:.1f} '
            path += f'L {points[1][0]:.1f},{points[1][1]:.1f}"/>\n'
        else:
            # Cubic Bézier using SVG curve command: C = curveto
            path = f' <path d="M {points[0][0]:.1f},{points[0][1]:.1f} '
            path += f'C {points[1][0]:.1f},{points[1][1]:.1f} '
            path += f'{points[2][0]:.1f},{points[2][1]:.1f} '
            path += f'{points[3][0]:.1f},{points[3][1]:.1f}"/>\n'
        
        svg += path
    
    # Close SVG document
    svg += '</svg>'
    return svg

def save_svg(curves: List[BezierCurve], width: int, height: int, output_path: str):
    """Save Bézier curves as an SVG file.
    
    Args:
        curves: List of BezierCurve objects
        width: Width of the SVG viewport
        height: Height of the SVG viewport
        output_path: Path where the SVG file will be saved
    """
    svg_content = generate_svg(curves, width, height)
    
    # Ensure directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write SVG file
    with open(output_path, 'w') as f:
        f.write(svg_content)