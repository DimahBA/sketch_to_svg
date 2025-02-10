import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set

@dataclass(frozen=True)
class Node:
    x: int
    y: int
    type: str  # 'junction', 'endpoint', or 'turn'
    
    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.x == other.x and self.y == other.y and self.type == other.type
        
    def __hash__(self):
        return hash((self.x, self.y, self.type))

@dataclass(frozen=True)
class Edge:
    start_node: Node
    end_node: Node
    pixels: Tuple[Tuple[int, int], ...]  # Tuple of pixel coordinates for hashability
    
    def __post_init__(self):
        # Convert pixels list to tuple for hashability
        object.__setattr__(self, 'pixels', tuple(self.pixels))
    
    def __hash__(self):
        return hash((self.start_node, self.end_node, self.pixels))
        
    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return (self.start_node == other.start_node and 
                self.end_node == other.end_node and 
                self.pixels == other.pixels)

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
    h, w = skeleton.shape
    nodes = []
    
    def count_branch_transitions(patch: np.ndarray) -> int:
        """Count the number of 0->1 transitions when going around the 8 neighbors.
        This helps identify true branch points by counting how many separate
        branches connect to this point.
        """
        # Extract the 8 neighbors in clockwise order
        neighbors = [
            patch[0, 0], patch[0, 1], patch[0, 2],  # Top row
            patch[1, 2],                            # Right
            patch[2, 2], patch[2, 1], patch[2, 0],  # Bottom row
            patch[1, 0],                            # Left
            patch[0, 0]                             # Back to start to close the circle
        ]
        # Count 0->1 transitions
        transitions = sum((neighbors[i] == 0 and neighbors[i+1] > 0) 
                        for i in range(8))
        return transitions

    def analyze_local_structure(y: int, x: int) -> Optional[str]:
        """Analyze the local 3x3 region around a point to determine its type.
        Returns 'endpoint', 'junction', or None.
        """
        patch = skeleton[y-1:y+2, x-1:x+2].copy()
        center_val = patch[1, 1]
        patch[1, 1] = 0  # Don't count center pixel
        
        # Count total neighbors and transitions
        neighbor_count = np.count_nonzero(patch)
        transitions = count_branch_transitions(patch)
        
        if neighbor_count == 1:
            # One neighbor = endpoint
            return 'endpoint'
        elif neighbor_count >= 3 and transitions >= 3:
            # 3+ neighbors with 3+ distinct branches = junction
            return 'junction'
        return None

    # Scan the image for critical points
    for y in range(1, h-1):
        for x in range(1, w-1):
            if skeleton[y, x] > 0:  # Only check skeleton pixels
                point_type = analyze_local_structure(y, x)
                if point_type:
                    # We found a critical point - create a node
                    nodes.append(Node(x, y, point_type))
    
    return nodes
    

def trace_skeleton_paths(skeleton: np.ndarray, nodes: List[Node]) -> List[Edge]:
    """Trace paths between nodes along skeleton"""
    edges = []
    h, w = skeleton.shape
    node_positions = {(node.y, node.x): node for node in nodes}
    visited = np.zeros_like(skeleton, dtype=bool)
    
    def get_neighbors(y: int, x: int, local_visited: np.ndarray) -> List[Tuple[int, int]]:
        """Get unvisited neighboring skeleton pixels"""
        neighbors = []
        # Try orthogonal directions first, then diagonals for smoother paths
        for dy, dx in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,1), (-1,-1), (1,-1)]:
            ny, nx = y + dy, x + dx
            if (0 <= ny < h and 0 <= nx < w and 
                skeleton[ny, nx] > 0 and 
                not local_visited[ny, nx]):
                neighbors.append((ny, nx))
        return neighbors

    def find_closest_node(y: int, x: int, exclude_node: Node = None) -> Optional[Node]:
        """Find the closest node to the given point using BFS"""
        if (y, x) in node_positions:
            return node_positions[(y, x)]
            
        queue = [(y, x, [])]  # Include path in queue
        bfs_visited = set([(y, x)])
        
        while queue:
            cy, cx, path = queue.pop(0)
            current_path = path + [(cx, cy)]
            
            # Check neighbors
            for dy, dx in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,1), (-1,-1), (1,-1)]:
                ny, nx = cy + dy, cx + dx
                if (0 <= ny < h and 0 <= nx < w and 
                    skeleton[ny, nx] > 0 and 
                    (ny, nx) not in bfs_visited):
                    
                    if (ny, nx) in node_positions:
                        node = node_positions[(ny, nx)]
                        if node != exclude_node:
                            return node, current_path
                            
                    queue.append((ny, nx, current_path))
                    bfs_visited.add((ny, nx))
        return None, []

    # First pass: standard path tracing between nodes
    for start_node in nodes:
        local_visited = visited.copy()
        # Don't mark nodes as visited initially
        for node in nodes:
            local_visited[node.y, node.x] = False
            
        neighbors = get_neighbors(start_node.y, start_node.x, local_visited)
        
        for ny, nx in neighbors:
            if local_visited[ny, nx]:
                continue
                
            path = [(start_node.x, start_node.y)]
            curr_y, curr_x = ny, nx
            
            while True:
                path.append((curr_x, curr_y))
                local_visited[curr_y, curr_x] = True
                
                if (curr_y, curr_x) in node_positions:
                    end_node = node_positions[(curr_y, curr_x)]
                    if end_node != start_node:
                        edges.append(Edge(start_node, end_node, path))
                        local_visited[curr_y, curr_x] = False
                    break
                
                next_pixels = get_neighbors(curr_y, curr_x, local_visited)
                if not next_pixels:
                    break
                curr_y, curr_x = next_pixels[0]

    # Create a map of visited pixels
    edge_pixels = np.zeros_like(skeleton, dtype=bool)
    for edge in edges:
        for x, y in edge.pixels:
            edge_pixels[y, x] = True
            
    # Find remaining unvisited skeleton pixels
    remaining_pixels = set()
    for y in range(h):
        for x in range(w):
            if skeleton[y, x] > 0 and not edge_pixels[y, x]:
                remaining_pixels.add((y, x))

    # Process remaining pixels
    while remaining_pixels:
        # Take a pixel from remaining set
        y, x = remaining_pixels.pop()
        
        # Find closest node
        start_result = find_closest_node(y, x)
        if not start_result[0]:
            continue
        start_node, path_to_start = start_result
        
        # Find second closest node
        end_result = find_closest_node(y, x, start_node)
        if not end_result[0]:
            continue
        end_node, path_to_end = end_result
        
        # Create complete path
        complete_path = list(reversed(path_to_start[:-1])) + [(x, y)] + path_to_end[1:]
        edges.append(Edge(start_node, end_node, complete_path))
        
        # Remove traced pixels from remaining set
        for px, py in complete_path:
            remaining_pixels.discard((py, px))

    # Remove duplicate edges
    unique_edges = []
    seen_pairs = set()
    for edge in edges:
        pair = tuple(sorted([(edge.start_node.x, edge.start_node.y), 
                           (edge.end_node.x, edge.end_node.y)]))
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            unique_edges.append(edge)
    
    return unique_edges
