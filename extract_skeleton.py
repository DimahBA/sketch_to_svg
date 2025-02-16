import numpy as np
import cv2

def get_ball_structuring_element(radius: int):
    """Get a ball shape structuring element with specific radius for morphology operation."""
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))

def clean_small_components(image, min_size=10):
    """Remove small connected components."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=8)
    cleaned = np.zeros_like(image)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_size:
            cleaned[labels == label] = 255
    return cleaned

def preprocess_image(image):
    """Apply preprocessing to reduce noise and enhance lines using Sobel edge detection."""
    # Normalize the image
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    
    # Apply bilateral filter to reduce noise while preserving edges
    smoothed = cv2.bilateralFilter(normalized, d=5, sigmaColor=50, sigmaSpace=50)
    
    # Apply Sobel operators in both x and y directions
    sobelx = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute the magnitude of gradients
    magnitude = cv2.magnitude(sobelx, sobely)
    
    # Normalize magnitude to 0-255 range
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(magnitude)
    
    # Apply threshold to get binary image
    # Using a lower threshold value since Sobel already highlights edges
    _, binary = cv2.threshold(enhanced, 30, 255, cv2.THRESH_BINARY)
    
    # Clean up noise and connect nearby lines
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Invert the image to get black lines on white background
    inverted = cv2.bitwise_not(cleaned)

    return inverted

def zhang_suen_thinning(image):
    """Zhang-Suen thinning algorithm implementation."""
    image = image.copy()
    changing = True
    
    while changing:
        changing = False
        deletion_markers = []
        
        # Phase 1
        rows, columns = image.shape
        for i in range(1, rows - 1):
            for j in range(1, columns - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = n = [
                    image[i-1, j  ] == 255,
                    image[i-1, j+1] == 255,
                    image[i  , j+1] == 255,
                    image[i+1, j+1] == 255,
                    image[i+1, j  ] == 255,
                    image[i+1, j-1] == 255,
                    image[i  , j-1] == 255,
                    image[i-1, j-1] == 255,
                ]
                
                if (image[i, j] == 255 and
                    2 <= sum(n) <= 6 and
                    sum((n[k] and not n[k-1]) for k in range(8)) == 1 and
                    not (P2 and P4 and P6) and
                    not (P4 and P6 and P8)):
                    deletion_markers.append((i, j))
                    changing = True
        
        for i, j in deletion_markers:
            image[i, j] = 0

        # Phase 2
        deletion_markers = []
        for i in range(1, rows - 1):
            for j in range(1, columns - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = n = [
                    image[i-1, j  ] == 255,
                    image[i-1, j+1] == 255,
                    image[i  , j+1] == 255,
                    image[i+1, j+1] == 255,
                    image[i+1, j  ] == 255,
                    image[i+1, j-1] == 255,
                    image[i  , j-1] == 255,
                    image[i-1, j-1] == 255,
                ]
                
                if (image[i, j] == 255 and
                    2 <= sum(n) <= 6 and
                    sum((n[k] and not n[k-1]) for k in range(8)) == 1 and
                    not (P2 and P4 and P8) and
                    not (P2 and P6 and P8)):
                    deletion_markers.append((i, j))
                    changing = True
        
        for i, j in deletion_markers:
            image[i, j] = 0
                    
    return image

def merge_and_thin_lines(skeleton, dilation_size=2):
    """Merge nearby parallel lines by dilation and then thin back to single pixel width."""
    # Create a circular structuring element for dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size*2+1, dilation_size*2+1))
    
    # Dilate to merge nearby lines
    dilated = cv2.dilate(skeleton, kernel, iterations=1)
    
    # Close small gaps
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    
    # Thin back to single pixel width
    thinned = zhang_suen_thinning(closed)
    
    return thinned

def extract_skeleton(image: np.ndarray):
    """Extract skeleton from a sketch image using region-based approach with improved line merging."""
    # Convert to grayscale if needed
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    print("ES: preprocess_image")
    # Preprocess image
    preprocessed = preprocess_image(gray)
    
    print("ES: trapped_ball_segmentation")
    # Extract regions
    regions = trapped_ball_segmentation(preprocessed, ball_radius=2)
    
    # Initialize skeleton
    skeleton = np.zeros_like(preprocessed)
    
    print("ES: region boundaries")
    # Extract boundaries between regions
    for label in range(1, regions.max() + 1):
        region = (regions == label).astype(np.uint8) * 255
        dilated = cv2.dilate(region, get_ball_structuring_element(1))
        boundary = cv2.subtract(dilated, region)
        skeleton = cv2.bitwise_or(skeleton, boundary)
    
    print("ES: open curves handling")
    # Handle open curves
    binary_inv = cv2.bitwise_not(preprocessed)
    # Only process areas far from existing boundaries
    distance = cv2.distanceTransform(cv2.bitwise_not(skeleton), 
                                   cv2.DIST_L2, 5)
    far_pixels = (distance > 2).astype(np.uint8) * 255
    far_pixels = cv2.bitwise_and(far_pixels, binary_inv)
    
    print("ES: zhang_suen_thinning")
    # Thin the remaining pixels
    thinned = zhang_suen_thinning(far_pixels)
    
    # Combine boundaries with thinned open curves
    combined = cv2.bitwise_or(skeleton, thinned)
    
    print("ES: clean_small_components")
    # Clean small components
    cleaned = clean_small_components(combined, min_size=5)
    
    print("ES: merge_and_thin_lines")
    # Merge nearby parallel lines and thin back to single pixel width
    merged = merge_and_thin_lines(cleaned, dilation_size=3)
    
    # Final cleaning of any remaining small artifacts
    #final_cleaned = clean_small_components(merged, min_size=10)

    print("ES: dedupe nearby pixels")
    h, w = merged.shape
    for y in range(1, h-1):
        for x in range(1, w-1):
            if merged[y, x] == 0:
                continue

            


            north = merged[y+1, x  ]
            south = merged[y-1, x  ]
            east  = merged[y  , x-1]
            west  = merged[y  , x+1]

            diag_ne = merged[y+1, x-1]
            diag_nw = merged[y+1, x+1]
            diag_se = merged[y-1, x-1]
            diag_sw = merged[y-1, x+1]

            # check for L's, like
            #
            #  .X.
            #  .XX
            #  ...
            #
            # (northwest L)

            north_east = (north and east) and (not ( south or west ))
            north_west = (north and west) and (not ( south or east ))
            south_east = (south and east) and (not ( north or west ))
            south_west = (south and west) and (not ( north or east ))
            
            if north_east and diag_sw:
                continue
            if north_west and diag_se:
                continue
            if south_east and diag_nw:
                continue
            if south_west and diag_ne:
                continue


            if north_east or north_west or south_east or south_west:
                merged[y, x] = 0

    #cv2.imshow('merged', merged)
    #cv2.waitKey(0)

    # Invert the final output - black lines on white background
    result = cv2.bitwise_not(merged)
    
    return result

def trapped_ball_segmentation(image: np.ndarray, ball_radius: int = 3):
    """Segment image using trapped-ball approach to handle small gaps between regions."""
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    binary_inv = cv2.bitwise_not(binary)
    result = binary_inv.copy()
    ball = get_ball_structuring_element(ball_radius)
    
    h, w = image.shape[:2]
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    filled_map = np.zeros_like(image)
    
    next_label = 1
    step = max(1, ball_radius // 2)
    for y in range(0, h, step):
        for x in range(0, w, step):
            if filled_map[y,x] != 0 or binary_inv[y,x] == 0:
                continue
                
            cv2.floodFill(result, mask, (x,y), next_label)
            _, filled, _, _ = cv2.floodFill(filled_map, mask, (x,y), next_label)
            next_label += 1
            
    return filled_map

def test_skeleton_extraction():
    """Test function to demonstrate usage"""
    image = cv2.imread('sketch.png', cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Could not load image")
        return
        
    skeleton = extract_skeleton(image)
    cv2.imshow('Original', image)
    cv2.imshow('Skeleton', skeleton)
    cv2.waitKey(0)
    cv2.destroyAllWindows()