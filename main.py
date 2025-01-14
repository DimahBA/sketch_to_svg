import cv2
import numpy as np
import argparse
from pathlib import Path
from extract_skeleton import extract_skeleton

def process_image(image_path: str, output_dir: str = None, debug: bool = False):
    """
    Process a single image to extract its skeleton.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save results (optional)
        debug: If True, shows intermediate results and waits for key press
    """
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
    
    # Prepare visualization
    viz = np.zeros((image.shape[0], image.shape[1] * 2 + 10, 3), dtype=np.uint8)
    
    # Convert grayscale to BGR for visualization
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Show original and result side by side
    viz[:, :image.shape[1]] = image
    viz[:, -image.shape[1]:] = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(viz, 'shoes', (10, 30), font, 1, (255,255,255), 2)
    cv2.putText(viz, 'Skeleton', (image.shape[1] + 20, 30), font, 1, (255,255,255), 2)
    
    if debug:
        # Show result and wait for key press
        cv2.imshow('Skeleton Extraction', viz)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    if output_dir:
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save results
        base_name = Path(image_path).stem
        cv2.imwrite(str(Path(output_dir) / f"{base_name}_skeleton.png"), skeleton)
        cv2.imwrite(str(Path(output_dir) / f"{base_name}_comparison.png"), viz)
        print(f"Results saved in {output_dir}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract skeleton from sketch images')
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
