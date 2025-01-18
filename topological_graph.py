import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Node:
    x: int
    y: int
    type: str  # 'junction', 'endpoint', or 'turn'
    
@dataclass
class Edge:
    start_node: Node
    end_node: Node
    pixels: List[Tuple[int, int]]  # List of pixel coordinates along the edge

def build_topological_graph(skeleton: np.ndarray):
    """Convert skeleton to topological graph"""
    # Invert skeleton if necessary (ensure black background, white lines)
    if np.mean(skeleton) > 127:
        skeleton = cv2.bitwise_not(skeleton)
    
    # 1. Find junction points and endpoints
    nodes = find_critical_points(skeleton)
    
    # 2. Extract edges by tracing paths between nodes
    edges = trace_skeleton_paths(skeleton, nodes)
    
    return nodes, edges

def find_critical_points(skeleton: np.ndarray) -> List[Node]:
    """Find junctions and endpoints in the skeleton"""
    h, w = skeleton.shape
    nodes = []
    
    # Collect all potential points first
    potential_points = []
    for y in range(1, h-1):
        for x in range(1, w-1):
            if skeleton[y,x] > 0:  # Check for skeleton pixels
                # Get 8-connected neighbors
                patch = skeleton[y-1:y+2, x-1:x+2].copy()
                patch[1,1] = 0  # Don't count center pixel
                neighbor_count = np.count_nonzero(patch)
                
                if neighbor_count == 1:
                    potential_points.append((x, y, 'endpoint'))
                elif neighbor_count >= 3:
                    potential_points.append((x, y, 'junction'))
    
    # Group close points
    min_distance = 15  # Minimum distance between points
    grouped_points = []
    used = set()
    
    for i, (x, y, point_type) in enumerate(potential_points):
        if i in used:
            continue
            
        # Find close points
        current_group = []
        for j, (x2, y2, type2) in enumerate(potential_points):
            if j not in used and type2 == point_type:  # Only group same type points
                dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                if dist < min_distance:
                    current_group.append((x2, y2))
                    used.add(j)
        
        if current_group:
            # Get median point from group
            median_x = int(np.median([p[0] for p in current_group]))
            median_y = int(np.median([p[1] for p in current_group]))
            nodes.append(Node(median_x, median_y, point_type))
        else:
            # Single point
            nodes.append(Node(x, y, point_type))
    
    return nodes

def get_next_pixel(skeleton: np.ndarray, current: Tuple[int, int], visited: np.ndarray) -> Tuple[int, int]:
    """Get next unvisited neighbor pixel in skeleton"""
    y, x = current
    # Check 8-connected neighbors
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            ny, nx = y + dy, x + dx
            if (0 <= ny < skeleton.shape[0] and 
                0 <= nx < skeleton.shape[1] and 
                skeleton[ny, nx] > 0 and 
                not visited[ny, nx]):
                return (ny, nx)
    return None

def trace_skeleton_paths(skeleton: np.ndarray, nodes: List[Node]) -> List[Edge]:
    """Trace paths between nodes along skeleton"""
    edges = []
    visited = np.zeros_like(skeleton, dtype=bool)
    node_positions = {(node.y, node.x): node for node in nodes}
    
    # Mark nodes as visited
    for node in nodes:
        visited[node.y, node.x] = True
    
    # Start from each node
    for start_node in nodes:
        start_pos = (start_node.y, start_node.x)
        
        # Find unvisited neighbors of the start node
        next_pixel = get_next_pixel(skeleton, start_pos, visited)
        while next_pixel is not None:
            path = [start_pos]
            current = next_pixel
            
            # Trace path until we hit another node or dead end
            while current is not None:
                path.append(current)
                visited[current] = True
                
                if current in node_positions:
                    # Found path to another node
                    end_node = node_positions[current]
                    # Convert (y,x) coordinates to (x,y) for Edge pixels
                    pixel_path = [(x, y) for y, x in path]
                    edges.append(Edge(start_node, end_node, pixel_path))
                    break
                
                current = get_next_pixel(skeleton, current, visited)
            
            # Look for other paths from the start node
            next_pixel = get_next_pixel(skeleton, start_pos, visited)
    
    return edges