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
    """Find junctions and endpoints in the skeleton using a balanced approach.
    Combines neighbor analysis with transition counting and moderate grouping."""
    h, w = skeleton.shape
    nodes = []
    
    def count_branch_transitions(patch: np.ndarray) -> int:
        """Count the number of 0->1 transitions in 8-neighborhood.
        This helps identify true branches vs noise."""
        # Extract the 8 neighbors in clockwise order
        neighbors = [
            patch[0, 0], patch[0, 1], patch[0, 2],  # Top row
            patch[1, 2],                            # Right
            patch[2, 2], patch[2, 1], patch[2, 0],  # Bottom row
            patch[1, 0],                            # Left
            patch[0, 0]                             # Back to start
        ]
        # Count 0->1 transitions
        transitions = sum((neighbors[i] == 0 and neighbors[i+1] > 0) 
                        for i in range(8))
        return transitions

    def analyze_local_structure(y: int, x: int) -> Optional[str]:
        """Analyze the local 3x3 region to determine point type.
        Uses both neighbor count and transition analysis."""
        patch = skeleton[y-1:y+2, x-1:x+2].copy()
        center_val = patch[1, 1]
        patch[1, 1] = 0  # Don't count center pixel
        
        # Count neighbors and transitions
        neighbor_count = np.count_nonzero(patch)
        transitions = count_branch_transitions(patch)
        
        if neighbor_count == 1:
            return 'endpoint'
        elif (neighbor_count >= 3 and transitions >= 2):  # More lenient than before
            # Additional check for branch strength
            branch_strength = np.sum(patch) / 255.0  # Normalize
            if branch_strength >= 2.5:  # Require significant branch presence
                return 'junction'
        return None

    # First pass: collect potential critical points
    potential_points = []
    for y in range(1, h-1):
        for x in range(1, w-1):
            if skeleton[y, x] > 0:
                point_type = analyze_local_structure(y, x)
                if point_type:
                    potential_points.append((x, y, point_type))
    
    # Second pass: group nearby points
    min_distance = 15  # Moderate grouping distance
    used = set()
    
    for i, (x, y, point_type) in enumerate(potential_points):
        if i in used:
            continue
            
        # Find close points of same type
        current_group = []
        for j, (x2, y2, type2) in enumerate(potential_points):
            if j not in used and type2 == point_type:
                dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                if dist < min_distance:
                    current_group.append((x2, y2))
                    used.add(j)
        
        if current_group:
            # Use weighted average position based on neighbor count
            weights = []
            for px, py in current_group:
                patch = skeleton[py-1:py+2, px-1:px+2].copy()
                patch[1, 1] = 0
                weights.append(np.count_nonzero(patch))
            
            weights = np.array(weights) / np.sum(weights)
            avg_x = int(np.sum([p[0] * w for p, w in zip(current_group, weights)]))
            avg_y = int(np.sum([p[1] * w for p, w in zip(current_group, weights)]))
            
            nodes.append(Node(avg_x, avg_y, point_type))
        else:
            nodes.append(Node(x, y, point_type))
    
    return nodes

def trace_skeleton_paths(skeleton: np.ndarray, nodes: List[Node]) -> List[Edge]:
    h, w = skeleton.shape
    edges = []
    visited = np.zeros_like(skeleton, dtype=bool)

    def get_neighbors(x: int, y: int) -> List[Tuple[int, int]]:
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if (0 <= ny < h and 0 <= nx < w and skeleton[ny, nx] > 0):
                    neighbors.append((nx, ny))
        return neighbors

    def find_nearest_node(x: int, y: int, start_node: Node) -> Optional[Tuple[Node, List[Tuple[int, int]]]]:
        queue = [(x, y, [(x, y)])]  # (x, y, path to get from start point to end) 
        local_visited = visited.copy() 
        local_visited[y, x] = True #start point is marked as visited
        
        while queue:
            cx, cy, path = queue.pop(0)
            # Check for nearby nodes
            for node in nodes:
                if node != start_node and abs(node.x - cx) <= 3 and abs(node.y - cy) <= 3:
                    return node, path + [(node.x, node.y)] #return that node and the path to it
                    
            # Add unvisited neighbors to queue
            for nx, ny in get_neighbors(cx, cy):
                if not local_visited[ny, nx]:
                    local_visited[ny, nx] = True
                    queue.append((nx, ny, path + [(nx, ny)]))
        return None

    # Process each node
    for start_node in nodes:
        # Find paths from each neighbor of the start node
        for next_x, next_y in get_neighbors(start_node.x, start_node.y):
            if not visited[next_y, next_x]:
                result = find_nearest_node(next_x, next_y, start_node)
                if result:
                    end_node, path = result
                    edges.append(Edge(
                        start_node=start_node,
                        end_node=end_node,
                        pixels=[(start_node.x, start_node.y)] + path
                    ))
                    # Mark path as visited
                    for px, py in path:
                        visited[py, px] = True

    # Remove duplicate edges
    unique_edges = []
    seen = set()
    for edge in edges:
        pair = tuple(sorted([(edge.start_node.x, edge.start_node.y), 
                           (edge.end_node.x, edge.end_node.y)]))
        if pair not in seen:
            unique_edges.append(edge)
            seen.add(pair)

    return unique_edges
    