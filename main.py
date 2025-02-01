import cv2
import numpy as np
import argparse
from pathlib import Path
from extract_skeleton import extract_skeleton
from topological_graph import build_topological_graph
from bezier_fitting import fit_bezier_curves, evaluate_bezier

def draw_bezier_curves(image, bezier_curves):
    """Draw Bézier curves and their control points"""
    viz = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    
    # Draw each Bézier curve
    for curve in bezier_curves:
        # Sample points along the curve
        t_values = np.linspace(0, 1, 100)
        points = np.array([evaluate_bezier(curve.control_points, curve.degree, t) for t in t_values])
        points = points.astype(np.int32)
        
        # Draw the curve in green
        for i in range(len(points) - 1):
            cv2.line(viz, 
                    tuple(points[i]), 
                    tuple(points[i + 1]), 
                    (0, 255, 0), 2)
        
        # Draw control points and their connections
        for i in range(len(curve.control_points)):
            pt = curve.control_points[i].astype(int)
            # Draw control points in red
            cv2.circle(viz, tuple(pt), 3, (0, 0, 255), -1)
            
        # Draw control polygon in blue
        for i in range(len(curve.control_points) - 1):
            pt1 = curve.control_points[i].astype(int)
            pt2 = curve.control_points[i + 1].astype(int)
            cv2.line(viz, tuple(pt1), tuple(pt2), (255, 0, 0), 1, cv2.LINE_AA)
    
    return viz

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

def process_image(image_path: str, output_dir: str = None, debug: bool = False):
    """Process a single image to extract skeleton, build graph and fit curves."""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Extract skeleton
    print(f"Processing {image_path}...")
    skeleton = extract_skeleton(gray)
    
    # Build topological graph
    nodes, edges = build_topological_graph(skeleton)
    print(f"Found {len(nodes)} nodes and {len(edges)} edges")
    
    # Draw topological graph
    topo_viz = draw_topological_graph(skeleton, nodes, edges)
    
    # Fit Bézier curves
    print("Fitting Bézier curves...")
    bezier_curves = fit_bezier_curves(edges)
    print(f"Generated {len(bezier_curves)} Bézier curves")
    
    # Draw Bézier curves
    bezier_viz = draw_bezier_curves(skeleton, bezier_curves)

    # Prepare visualization with four panels
    spacing = 10
    viz_width = image.shape[1] * 4 + spacing * 3
    viz = np.zeros((image.shape[0], viz_width, 3), dtype=np.uint8)

    # Convert grayscale to BGR for visualization
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Show all four images side by side
    viz[:, :image.shape[1]] = image
    viz[:, image.shape[1] + spacing:2*image.shape[1] + spacing] = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
    viz[:, 2*image.shape[1] + 2*spacing:3*image.shape[1] + 2*spacing] = topo_viz
    viz[:, 3*image.shape[1] + 3*spacing:] = bezier_viz

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(viz, 'Input', (10, 30), font, 1, (255,255,255), 2)
    cv2.putText(viz, 'Skeleton', (image.shape[1] + spacing + 10, 30), font, 1, (255,255,255), 2)
    cv2.putText(viz, 'Graph', (2*image.shape[1] + 2*spacing + 10, 30), font, 1, (255,255,255), 2)
    cv2.putText(viz, 'Bezier', (3*image.shape[1] + 3*spacing + 10, 30), font, 1, (255,255,255), 2)

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
        
        # Save Bézier curve data
        with open(str(Path(output_dir) / f"{base_name}_curves.txt"), 'w') as f:
            for i, curve in enumerate(bezier_curves):
                f.write(f"Curve {i + 1}:\n")
                f.write(f"  Degree: {curve.degree}\n")
                f.write(f"  Start node: ({curve.start_node.x}, {curve.start_node.y}) - {curve.start_node.type}\n")
                f.write(f"  End node: ({curve.end_node.x}, {curve.end_node.y}) - {curve.end_node.type}\n")
                f.write(f"  Control points:\n")
                for j, pt in enumerate(curve.control_points):
                    f.write(f"    P{j}: ({pt[0]:.2f}, {pt[1]:.2f})\n")
                f.write(f"  Fitting error: {curve.error:.2f}\n")
                f.write("\n")
                
        print(f"Results saved in {output_dir}")

        
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract and vectorize sketches')
    parser.add_argument('input', help='Input image path or directory')
    parser.add_argument('--output', '-o', help='Output directory', default='output')
    parser.add_argument('--debug', '-d', action='store_true', help='Show debug visualization')
    args = parser.parse_args()
    
    # Process input path
    input_path = Path(args.input)
    if input_path.is_file():
        # Process single image
        process_image(str(input_path), args.output, args.debug)
    elif input_path.is_dir():
        # Process all images in directory
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
        for img_path in input_path.iterdir():
            if img_path.suffix.lower() in image_extensions:
                try:
                    process_image(str(img_path), args.output, args.debug)
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
    else:
        print(f"Error: Input path {args.input} does not exist")
        return

if __name__ == "__main__":
    main()