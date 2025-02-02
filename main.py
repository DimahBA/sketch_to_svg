import cv2
import numpy as np
import math
import argparse
from pathlib import Path
from extract_skeleton import extract_skeleton
from topological_graph import build_topological_graph, Node, Edge
from hyper_graph import Hypergraph

def evaluate_bezier(control_points: np.ndarray, degree: int, t: float) -> np.ndarray:
    """Evaluate a Bézier curve at parameter t using De Casteljau's algorithm."""
    points = control_points.copy()
    for r in range(degree):
        for i in range(degree - r):
            points[i] = (1 - t) * points[i] + t * points[i + 1]
    return points[0]

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

def draw_topological_graph(skeleton, nodes, edges):
    """Draw the topological graph with nodes and edges"""
    viz = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR) if len(skeleton.shape) == 2 else skeleton.copy()
    
    # Draw edges
    for edge in edges:
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

def draw_hypergraph(image, hypergraph, show_control_points=False):
    """Draw hypergraph with each hyperedge in a different color"""
    viz = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    
    # Generate distinct colors for each hyperedge
    colors = generate_distinct_colors(len(hypergraph.hyperedges))
    
    # Draw each hyperedge
    for hedge, color in zip(hypergraph.hyperedges, colors):
        # Draw the Bézier curve
        if hedge.bezier_curve is not None:
            # Sample points along the curve
            t_values = np.linspace(0, 1, 100)
            points = np.array([evaluate_bezier(hedge.bezier_curve.control_points, 
                                            hedge.bezier_curve.degree, t) 
                             for t in t_values])
            points = points.astype(np.int32)
            
            # Draw curve
            for i in range(len(points) - 1):
                cv2.line(viz, 
                        tuple(points[i]), 
                        tuple(points[i + 1]), 
                        color, 2, 
                        cv2.LINE_AA)
            
            if show_control_points:
                # Draw control points
                for pt in hedge.bezier_curve.control_points:
                    pt = pt.astype(int)
                    cv2.circle(viz, tuple(pt), 3, color, -1)
                
                # Draw control polygon
                pts = hedge.bezier_curve.control_points.astype(np.int32)
                for i in range(len(pts) - 1):
                    cv2.line(viz, 
                            tuple(pts[i]), 
                            tuple(pts[i + 1]), 
                            color, 1, 
                            cv2.LINE_AA)
    
    # Draw shared edges with a distinctive style
    for hedge in hypergraph.hyperedges:
        for edge in hedge.shared_edges:
            for i in range(len(edge.pixels) - 1):
                pt1 = edge.pixels[i]
                pt2 = edge.pixels[i + 1]
                # Draw shared edges with dashed white line
                cv2.line(viz, pt1, pt2, (255, 255, 255), 3)
                cv2.line(viz, pt1, pt2, (0, 0, 0), 1)
    
    return viz

def draw_optimization_progress(image, hypergraph, iteration, energy):
    """Create a visualization of the optimization progress"""
    viz = draw_hypergraph(image, hypergraph)
    
    # Add text with iteration and energy information
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(viz, f'Iteration: {iteration}', (10, 30), font, 1, (255,255,255), 2)
    cv2.putText(viz, f'Energy: {energy:.2f}', (10, 70), font, 1, (255,255,255), 2)
    
    return viz

def process_image(image_path: str, output_dir: str = None, debug: bool = False):
    """Process a single image to extract skeleton, build graph and optimize hypergraph."""
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
    
    # Create and optimize hypergraph
    print("Building and optimizing hypergraph...")
    hypergraph = Hypergraph(nodes, edges)
    
    # Setup progress tracking
    if debug:
        frame_count = 0
        def optimization_callback(current_graph, temperature):
            nonlocal frame_count
            if frame_count % 10 == 0:  # Only show every 10th frame
                viz = draw_optimization_progress(skeleton, current_graph, 
                                              frame_count, current_graph.calculate_energy())
                cv2.imshow('Optimization Progress', viz)
                cv2.waitKey(1)
            frame_count += 1
    else:
        optimization_callback = None
    
    # Run optimization
    hypergraph.optimize(callback=optimization_callback)
    print(f"Generated {len(hypergraph.hyperedges)} hyperedges")
    
    if debug and optimization_callback:
        cv2.destroyWindow('Optimization Progress')
    
    # Draw visualizations
    topo_viz = draw_topological_graph(skeleton, nodes, edges)
    hyper_viz = draw_hypergraph(skeleton, hypergraph)

    # Create visualization with four panels
    spacing = 10
    viz_width = image.shape[1] * 4 + spacing * 3
    viz = np.zeros((image.shape[0], viz_width, 3), dtype=np.uint8)

    # Convert grayscale to BGR for visualization
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Show all images side by side
    viz[:, :image.shape[1]] = image
    viz[:, image.shape[1] + spacing:2*image.shape[1] + spacing] = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
    viz[:, 2*image.shape[1] + 2*spacing:3*image.shape[1] + 2*spacing] = topo_viz
    viz[:, 3*image.shape[1] + 3*spacing:] = hyper_viz

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(viz, 'Input', (10, 30), font, 1, (255,255,255), 2)
    cv2.putText(viz, 'Skeleton', (image.shape[1] + spacing + 10, 30), font, 1, (255,255,255), 2)
    cv2.putText(viz, 'Graph', (2*image.shape[1] + 2*spacing + 10, 30), font, 1, (255,255,255), 2)
    cv2.putText(viz, 'Hypergraph', (3*image.shape[1] + 3*spacing + 10, 30), font, 1, (255,255,255), 2)

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
        cv2.imwrite(str(Path(output_dir) / f"{base_name}_hyper.png"), hyper_viz)
        cv2.imwrite(str(Path(output_dir) / f"{base_name}_comparison.png"), viz)

        # Save hypergraph data
        with open(str(Path(output_dir) / f"{base_name}_hypergraph.txt"), 'w') as f:
            f.write(f"Number of hyperedges: {len(hypergraph.hyperedges)}\n\n")
            for i, hedge in enumerate(hypergraph.hyperedges):
                f.write(f"Hyperedge {i + 1}:\n")
                f.write(f"  Number of edges: {len(hedge.edges)}\n")
                f.write(f"  Number of shared edges: {len(hedge.shared_edges)}\n")
                if hedge.bezier_curve:
                    f.write(f"  Curve degree: {hedge.bezier_curve.degree}\n")
                    f.write(f"  Fitting error: {hedge.bezier_curve.error:.2f}\n")
                    f.write("  Control points:\n")
                    for j, pt in enumerate(hedge.bezier_curve.control_points):
                        f.write(f"    P{j}: ({pt[0]:.2f}, {pt[1]:.2f})\n")
                f.write("\n")

        print(f"Results saved in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Extract and vectorize sketches using hypergraph optimization')
    parser.add_argument('input', help='Input image path or directory')
    parser.add_argument('--output', '-o', help='Output directory', default='output')
    parser.add_argument('--debug', '-d', action='store_true', help='Show debug visualization')
    parser.add_argument('--lambda-param', type=float, default=0.6, 
                       help='Balance between fidelity and simplicity (default: 0.6)')
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