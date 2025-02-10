import cv2
import numpy as np
import argparse
from pathlib import Path
from extract_skeleton import extract_skeleton
from topological_graph import build_topological_graph, Node, Edge
from bezier_fitting import fit_bezier_curves

def generate_distinct_colors(n):
    """Generate n visually distinct colors using HSV color space."""
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7 + np.random.random() * 0.3
        value = 0.7 + np.random.random() * 0.3
        
        h = hue * 6
        c = value * saturation
        x = c * (1 - abs(h % 2 - 1))
        m = value - c
        
        if h < 1:
            r, g, b = c, x, 0
        elif h < 2:
            r, g, b = x, c, 0
        elif h < 3:
            r, g, b = 0, c, x
        elif h < 4:
            r, g, b = 0, x, c
        elif h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
            
        rgb = (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))
        colors.append(rgb)
        
    return colors

def draw_topological_graph(skeleton, nodes, edges, straight_lines=False):
    """Draw the topological graph with nodes and edges
    
    Args:
        skeleton: Input skeleton image
        nodes: List of Node objects
        edges: List of Edge objects
        straight_lines: If True, draw straight lines between connected nodes instead of pixel paths
    """
    viz = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR) if len(skeleton.shape) == 2 else skeleton.copy()
    
    # Draw edges
    for edge in edges:
        if straight_lines:
            # Draw straight green line between start and end nodes
            cv2.line(viz, 
                    (edge.start_node.x, edge.start_node.y),
                    (edge.end_node.x, edge.end_node.y),
                    (0, 255, 0), 1, cv2.LINE_AA)
        else:
            # Draw the pixel path in blue
            for i in range(len(edge.pixels) - 1):
                pt1 = edge.pixels[i]
                pt2 = edge.pixels[i + 1]
                cv2.line(viz, pt1, pt2, (255, 0, 0), 1)
    
    # Draw nodes
    for node in nodes:
        # Different colors for different node types
        color = {
            'junction': (0, 0, 255),    # Red
            'endpoint': (0, 255, 0),    # Green
            'turn': (0, 255, 255)       # Yellow
        }.get(node.type, (255, 255, 255))
        
        # Draw node with larger radius for visibility
        cv2.circle(viz, (node.x, node.y), 4, color, -1)
        cv2.circle(viz, (node.x, node.y), 5, color, 1)
    
    return viz


def evaluate_bezier(control_points: np.ndarray, t: float) -> np.ndarray:
    """Evaluate a cubic Bézier curve at parameter t using de Casteljau's algorithm.
    
    De Casteljau's algorithm provides a numerically stable way to evaluate Bézier curves
    by repeatedly interpolating between control points.
    
    Args:
        control_points: Array of 4 control points defining the cubic Bézier curve
        t: Parameter value between 0 and 1
        
    Returns:
        Point on the curve at parameter t
    """
    # Start with the control points
    points = control_points.copy()
    
    # Since we're working with cubic curves, we always do 3 iterations
    for _ in range(3):
        # In each iteration, create new points by interpolating between pairs
        for i in range(len(points) - 1):
            points[i] = (1 - t) * points[i] + t * points[i + 1]
    
    return points[0]

def draw_bezier_curves(image, curves, show_control_points=False):
    """Draw fitted cubic Bézier curves with each curve in a different color.
    
    This function visualizes cubic Bézier curves by sampling points along each curve
    and drawing them with anti-aliased lines. Each curve gets a unique color for
    clear visualization. Optionally shows control points and their connecting
    polygon to help understand how the curve is shaped.
    
    Args:
        image: Input image to draw on
        curves: List of BezierCurve objects (each having 4 control points)
        show_control_points: If True, draw control points and control polygon
        
    Returns:
        Visualization image with drawn curves
    """
    # Convert grayscale to color if needed
    viz = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    
    # Generate visually distinct colors for each curve
    colors = generate_distinct_colors(len(curves))
    
    # Draw each Bézier curve
    for curve, color in zip(curves, colors):
        # Sample 100 points along the curve for smooth rendering
        t_values = np.linspace(0, 1, 100)
        # Note: We don't need the degree parameter anymore since all curves are cubic
        points = np.array([evaluate_bezier(curve.control_points, t) for t in t_values])
        points = points.astype(np.int32)
        
        # Draw the curve as a series of small line segments
        for i in range(len(points) - 1):
            cv2.line(viz, 
                    tuple(points[i]), 
                    tuple(points[i + 1]), 
                    color,           # Unique color for each curve
                    2,               # Line thickness
                    cv2.LINE_AA)     # Anti-aliasing for smooth appearance
        
        if show_control_points:
            # Draw the four control points that define the curve
            for pt in curve.control_points:
                pt = pt.astype(int)
                cv2.circle(viz, tuple(pt), 3, color, -1)
            
            # Draw the control polygon (lines connecting control points)
            # This helps visualize how the control points influence the curve
            pts = curve.control_points.astype(np.int32)
            for i in range(len(pts) - 1):
                cv2.line(viz, 
                        tuple(pts[i]), 
                        tuple(pts[i + 1]), 
                        color, 
                        1,            # Thinner lines for the control polygon
                        cv2.LINE_AA)
    
    return viz
    
def process_image(image_path: str, output_dir: str = None, debug: bool = False):
    """Process a single image to extract skeleton, build graph and fit curves."""
    # Read and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    print(f"Processing {image_path}...")
    skeleton = extract_skeleton(gray)
    
    # Build topological graph
    nodes, edges = build_topological_graph(skeleton)
    print(f"Found {len(nodes)} nodes and {len(edges)} edges")
    
    # Fit Bézier curves
    print("Fitting Bézier curves...")
    bezier_curves = fit_bezier_curves(edges)
    print(f"Generated {len(bezier_curves)} Bézier curves")
    
    # Draw visualizations
    topo_viz = draw_topological_graph(skeleton, nodes, edges, straight_lines=False)
    straight_viz = draw_topological_graph(skeleton, nodes, edges, straight_lines=True)
    bezier_viz = draw_bezier_curves(skeleton, bezier_curves)

    # Create visualization with five panels
    spacing = 10
    viz_width = image.shape[1] * 5 + spacing * 4
    viz = np.zeros((image.shape[0], viz_width, 3), dtype=np.uint8)

    # Convert grayscale to BGR for visualization
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Show all images side by side
    viz[:, :image.shape[1]] = image
    viz[:, image.shape[1] + spacing:2*image.shape[1] + spacing] = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
    viz[:, 2*image.shape[1] + 2*spacing:3*image.shape[1] + 2*spacing] = topo_viz
    viz[:, 3*image.shape[1] + 3*spacing:4*image.shape[1] + 3*spacing] = straight_viz
    viz[:, 4*image.shape[1] + 4*spacing:] = bezier_viz

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(viz, 'Input', (10, 30), font, 1, (255,255,255), 2)
    cv2.putText(viz, 'Skeleton', (image.shape[1] + spacing + 10, 30), font, 1, (255,255,255), 2)
    cv2.putText(viz, 'Graph', (2*image.shape[1] + 2*spacing + 10, 30), font, 1, (255,255,255), 2)
    cv2.putText(viz, 'Straight', (3*image.shape[1] + 3*spacing + 10, 30), font, 1, (255,255,255), 2)
    cv2.putText(viz, 'Bezier', (4*image.shape[1] + 4*spacing + 10, 30), font, 1, (255,255,255), 2)

    if debug:
        cv2.imshow('Vectorization Pipeline', viz)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if output_dir:
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save results
        base_name = Path(image_path).stem
        cv2.imwrite(str(Path(output_dir) / f"{base_name}_skeleton.png"), skeleton)
        cv2.imwrite(str(Path(output_dir) / f"{base_name}_graph.png"), topo_viz)
        cv2.imwrite(str(Path(output_dir) / f"{base_name}_bezier.png"), bezier_viz)
        cv2.imwrite(str(Path(output_dir) / f"{base_name}_comparison.png"), viz)

        # Save curve data
        with open(str(Path(output_dir) / f"{base_name}_curves.txt"), 'w') as f:
            f.write(f"Number of curves: {len(bezier_curves)}\n\n")
            for i, curve in enumerate(bezier_curves):
                f.write(f"Curve {i + 1}:\n")
                f.write("  Control points:\n")
                for j, pt in enumerate(curve.control_points):
                    f.write(f"    P{j}: ({pt[0]:.2f}, {pt[1]:.2f})\n")
                f.write(f"  Start node: ({curve.start_node.x}, {curve.start_node.y}) - {curve.start_node.type}\n")
                f.write(f"  End node: ({curve.end_node.x}, {curve.end_node.y}) - {curve.end_node.type}\n")
                f.write("\n")

        print(f"Results saved in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Extract and vectorize sketches')
    parser.add_argument('input', help='Input image path or directory')
    parser.add_argument('--output', '-o', help='Output directory', default='output')
    parser.add_argument('--debug', '-d', action='store_true', help='Show debug visualization')
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if input_path.is_file():
        process_image(str(input_path), args.output, args.debug)
    elif input_path.is_dir():
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
        for img_path in input_path.iterdir():
            if img_path.suffix.lower() in image_extensions:
                try:
                    process_image(str(img_path), args.output, args.debug)
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
    else:
        print(f"Error: Input path {args.input} does not exist")

if __name__ == "__main__":
    main()