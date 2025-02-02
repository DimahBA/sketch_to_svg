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
    min_distance = 30  # Minimum distance between points
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
    h, w = skeleton.shape
    node_positions = {(node.y, node.x): node for node in nodes}
    visited = np.zeros_like(skeleton, dtype=bool)
    
    def get_neighbors(y: int, x: int, local_visited: np.ndarray) -> List[Tuple[int, int]]:
        """Get unvisited neighboring skeleton pixels"""
        neighbors = []
        # Try orthogonal directions first, then diagonals
        for dy, dx in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,1), (-1,-1), (1,-1)]:
            ny, nx = y + dy, x + dx
            if (0 <= ny < h and 0 <= nx < w and 
                skeleton[ny, nx] > 0 and 
                not local_visited[ny, nx]):
                neighbors.append((ny, nx))
        return neighbors

    # First pass: standard path tracing
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

    # Create a map of pixels that haven't been included in any edge
    edge_pixels = np.zeros_like(skeleton, dtype=bool)
    for edge in edges:
        for x, y in edge.pixels:
            edge_pixels[y, x] = True
            
    # Find remaining skeleton pixels that haven't been used
    unvisited_skeleton = np.logical_and(skeleton > 0, np.logical_not(edge_pixels))
    
    # Second pass: try all nodes again with focus on unvisited skeleton pixels
    for start_node in nodes:
        local_visited = np.zeros_like(skeleton, dtype=bool)
        for dy, dx in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,1), (-1,-1), (1,-1)]:
            ny, nx = start_node.y + dy, start_node.x + dx
            
            # Only start paths in directions of unvisited skeleton pixels
            if (0 <= ny < h and 0 <= nx < w and 
                skeleton[ny, nx] > 0 and 
                unvisited_skeleton[ny, nx]):
                
                path = [(start_node.x, start_node.y)]
                curr_y, curr_x = ny, nx
                path_visited = local_visited.copy()
                
                while True:
                    path.append((curr_x, curr_y))
                    path_visited[curr_y, curr_x] = True
                    
                    if (curr_y, curr_x) in node_positions:
                        end_node = node_positions[(curr_y, curr_x)]
                        if end_node != start_node:
                            edges.append(Edge(start_node, end_node, path))
                            # Mark these pixels as visited in the main map
                            for x, y in path:
                                edge_pixels[y, x] = True
                        break
                    
                    next_pixels = get_neighbors(curr_y, curr_x, path_visited)
                    if not next_pixels:
                        break
                    curr_y, curr_x = next_pixels[0]
                    
                local_visited |= path_visited

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

#trace a line to link nodes that are supposed to be linked but are not
""" def trace_skeleton_paths(skeleton: np.ndarray, nodes: List[Node]) -> List[Edge]:
    #Trace paths between nodes along skeleton
    edges = []
    h, w = skeleton.shape
    node_positions = {(node.y, node.x): node for node in nodes}
    visited = np.zeros_like(skeleton, dtype=bool)
    
    def get_neighbors(y: int, x: int, local_visited: np.ndarray) -> List[Tuple[int, int]]:
        #Get unvisited neighboring skeleton pixels
        neighbors = []
        # Check in all 8 directions, prioritizing direct neighbors
        for dy, dx in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,1), (-1,-1), (1,-1)]:
            ny, nx = y + dy, x + dx
            if (0 <= ny < h and 0 <= nx < w and 
                skeleton[ny, nx] > 0 and 
                not local_visited[ny, nx]):
                neighbors.append((ny, nx))
        return neighbors

    def find_closest_node(y: int, x: int, exclude_node: Node = None) -> Optional[Node]:
        #Find the closest node to the given point using BFS
        if (y, x) in node_positions:
            return node_positions[(y, x)]
            
        queue = [(y, x)]
        bfs_visited = set([(y, x)])
        
        while queue:
            cy, cx = queue.pop(0)
            
            # Check neighbors
            for dy, dx in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,1), (-1,-1), (1,-1)]:
                ny, nx = cy + dy, cx + dx
                if (0 <= ny < h and 0 <= nx < w and 
                    skeleton[ny, nx] > 0 and 
                    (ny, nx) not in bfs_visited):
                    
                    if (ny, nx) in node_positions:
                        node = node_positions[(ny, nx)]
                        if node != exclude_node:
                            return node
                            
                    queue.append((ny, nx))
                    bfs_visited.add((ny, nx))
        return None

    # First pass: standard path tracing
    for start_node in nodes:
        local_visited = visited.copy()
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

    # Create a map of visited pixels from existing edges
    edge_pixels = np.zeros_like(skeleton, dtype=bool)
    for edge in edges:
        for x, y in edge.pixels:
            edge_pixels[y, x] = True

    # Final pass: look for unvisited skeleton pixels
    for y in range(h):
        for x in range(w):
            if skeleton[y, x] > 0 and not edge_pixels[y, x]:
                # Found an unvisited skeleton pixel
                start_node = find_closest_node(y, x)
                if start_node:
                    end_node = find_closest_node(y, x, start_node)
                    if end_node:
                        # Trace path between the two closest nodes
                        local_visited = np.zeros_like(skeleton, dtype=bool)
                        path = [(start_node.x, start_node.y)]
                        curr_y, curr_x = y, x
                        
                        while True:
                            path.append((curr_x, curr_y))
                            local_visited[curr_y, curr_x] = True
                            
                            if (curr_y, curr_x) in node_positions:
                                if node_positions[(curr_y, curr_x)] == end_node:
                                    edges.append(Edge(start_node, end_node, path))
                                break
                            
                            next_pixels = get_neighbors(curr_y, curr_x, local_visited)
                            if not next_pixels:
                                break
                            curr_y, curr_x = next_pixels[0]

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
"""
    