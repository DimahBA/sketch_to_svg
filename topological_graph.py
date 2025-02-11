import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set
import math

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


def euclidean_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """Calculate Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

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

def analyze_local_structure(skeleton: np.ndarray, y: int, x: int) -> Optional[str]:
    """Analyze the local 3x3 region around a point to determine its type.
    Returns 'endpoint', 'junction', 'turn', or None.
    """
    patch = skeleton[y-1:y+2, x-1:x+2].copy()
    center_val = patch[1, 1]
    patch[1, 1] = 0  # Don't count center pixel
    
    # Count total neighbors and transitions
    neighbor_count = np.count_nonzero(patch)
    transitions = count_branch_transitions(patch)
    
    if neighbor_count == 1:
        return 'endpoint'
    elif neighbor_count >= 3 and transitions >= 3:
        return 'junction'
        
    # New: Detect sharp turns by analyzing local curvature
    if neighbor_count == 2:  # Potential turn point
        # Get indices of the two neighbors
        ys, xs = np.where(patch > 0)
        if len(ys) == 2:  # Verify we have exactly 2 neighbors
            # Convert to relative coordinates
            v1 = np.array([ys[0] - 1, xs[0] - 1])  # Vector to first neighbor
            v2 = np.array([ys[1] - 1, xs[1] - 1])  # Vector to second neighbor
            
            # Calculate angle between vectors
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            angle_degrees = np.degrees(angle)
            
            # If angle is less than 135 degrees, consider it a sharp turn
            if angle_degrees < 135:
                return 'turn'
    
    return None

def merge_nearby_nodes(candidates: List[Tuple[int, int, str]], min_distance: float) -> List[Node]:
    """Merge nodes that are closer than min_distance to each other.
    Priority order: junctions > turns > endpoints
    """
    # Sort candidates by priority
    nodes = []
    priority = {'junction': 0, 'turn': 1, 'endpoint': 2, 'DEBUG': -1}
    candidates.sort(key=lambda x: priority[x[2]])


    while candidates:
        x, y, node_type = candidates.pop(0)
        nearby = []

        if node_type == 'DEBUG':
            nodes.append(Node(x, y, 'DEBUG'))
            continue

        
        # Find all candidates within min_distance
        i = 0
        while i < len(candidates):
            cx, cy, ctype = candidates[i]
            if euclidean_distance((x, y), (cx, cy)) < min_distance:
                nearby.append(candidates.pop(i))
            else:
                i += 1
        
        if not nearby:
            # No nearby nodes, keep this one
            nodes.append(Node(x, y, node_type))
            continue
            
        # Handle merging based on node types
        if node_type == 'junction':
            # Average position of all nearby junctions
            all_points = [(x, y)] + [(nx, ny) for nx, ny, nt in nearby if nt == 'junction']
            avg_x = int(round(sum(p[0] for p in all_points) / len(all_points)))
            avg_y = int(round(sum(p[1] for p in all_points) / len(all_points)))
            nodes.append(Node(avg_x, avg_y, 'junction'))
        else:  # endpoint
            # If there's a junction nearby, skip this endpoint
            if any(nt == 'junction' for _, _, nt in nearby):
                continue
            # Otherwise, keep the most isolated endpoint
            all_points = [(x, y)] + [(nx, ny) for nx, ny, _ in nearby]
            # Choose point furthest from all other nodes
            best_point = max(all_points, key=lambda p: 
                sum(euclidean_distance(p, (n.x, n.y)) for n in nodes))
            nodes.append(Node(best_point[0], best_point[1], 'endpoint'))
    
    return nodes

def find_critical_points(skeleton: np.ndarray, min_distance: float = 6.0) -> List[Node]:
    """Find critical points (endpoints and junctions) in the skeleton,
    maintaining a minimum distance between nodes.
    
    Args:
        skeleton: Binary skeleton image
        min_distance: Minimum Euclidean distance between nodes
        
    Returns:
        List of Node objects representing critical points
    """
    h, w = skeleton.shape
    candidates = []
    

    visited = np.zeros_like(skeleton, dtype=bool)

    min_node_distance = 15 #int(round(min_distance)) # pixels
    allowed_angle_diff = 25 # degrees

    #max_rotation_angle = 10  # degrees


    def get_angle(p1, p2):
        # Calculate the difference vector from p1 to p2
        delta = np.array(p2) - np.array(p1)
        
        # Use numpy's arctan2 function to get the angle in radians
        # arctan2(y, x) handles all quadrants correctly
        angle = np.arctan2(delta[1], delta[0])
        
        # Convert to degrees if desired
        angle_degrees = np.degrees(angle)
        
        return angle_degrees


    def get_neighbors( x: int, y: int ):
        neighbors = []

        neighbor_angles_yx = {
            -1: { -1: 225, 0: 270, 1: 315 },
             0: { -1: 180,         1:   0 }, # 0,0 is not possible
             1: { -1: 135, 0:  90, 1:  45 },
        }

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue

                nx, ny = x + dx, y + dy

                is_in_bounds = (0 <= ny < h and 0 <= nx < w)
                if is_in_bounds and (skeleton[ny, nx] > 0):
                    neighbors.append((nx, ny, neighbor_angles_yx[dy][dx]))
        
        return neighbors

    def angle_difference(angle1, angle2):
        # Calculate the absolute difference and normalize to [0, 360)
        diff = abs(angle1 - angle2) % 360
        # Return the smaller angle between the difference and its complement
        angle_diff = min(diff, abs(360 - diff))

        #print(angle_diff)

        return angle_diff

    def get_closest_candidate_distance( from_x: int, from_y: int ):
        return min([
            math.sqrt( (x - from_x)**2 + (y - from_y)**2 ) for x, y, node_type in candidates
        ])

    """     
    def explore_skeleton( start_x: int, start_y: int ):
        INITIAL_DEPTH = 1
        #angle = None
        #visited[start_x, start_y] = True
        queue = [( start_y, start_x, start_x, start_y, INITIAL_DEPTH, None )]
                
        while queue:
            y, x, origin_y, origin_x, depth, edge_angle = queue.pop(0)

            if visited[y, x]:
                continue

            visited[y, x] = True
            neighbors = get_neighbors(x, y)

            current_angle = get_angle( (x, y), (start_x, start_y) )
            #current_angle = get_angle( (y, x), (start_y, start_x) )

            reached_min_dist = depth > min_node_distance



            if len(neighbors) == 1:
                candidates.append((x, y, 'endpoint'))
                # iterate on neighbors, if e.g. starting point is an endpoint.
                for nx, ny in neighbors:
                    queue.append( (ny, nx, origin_y, origin_x, depth+1, None) )
                continue

            if not reached_min_dist:
                for neighbor in neighbors:
                    nx, ny = neighbor
                    queue.append( (ny, nx, origin_y, origin_x, depth+1, None) )
                continue
            

            next_angle = edge_angle
            if next_angle is None:
                next_angle = current_angle
            

            # if no edge angle, any angle is valid
            angle_is_valid = ((edge_angle is None) or ( angle_difference(current_angle, edge_angle) < allowed_angle_diff ) )
            


            if len(neighbors) > 2:
                candidates.append((x, y, 'junction'))
                for neighbor in neighbors:
                    nx, ny = neighbor
                    queue.append( (ny, nx, y, x, INITIAL_DEPTH, None) )
                continue

            if not angle_is_valid:
                candidates.append((x, y, 'turn'))
                for neighbor in neighbors:
                    nx, ny = neighbor
                    # WARN: using current_angle might be causing issues with correctly detecting turns
                    queue.append( (ny, nx, y, x, INITIAL_DEPTH, next_angle) )
                continue


            # angle is valid
            for neighbor in neighbors:
                nx, ny = neighbor
                queue.append( (ny, nx, origin_y, origin_x, depth+1, next_angle) )
    """

    def explore_skeleton( start_x: int, start_y: int ):
        INITIAL_DEPTH = 1
        #angle = None
        #visited[start_x, start_y] = True
        queue = [( 
            start_x, start_y, # next position
            None, # reset base angle
            None, # reset current angle
            None,   # reset divergence angle
            None, None, # reset divergence start point
        )]

        while queue:
            x, y, base_angle, curr_angle, diverge_start_angle, diverge_start_x, diverge_start_y = queue.pop(0)

            if visited[y, x]:
                continue
            visited[y, x] = True


            neighbors = get_neighbors(x, y)

            # edge case: lone pixel ?
            # should not be possible
            if len(neighbors) == 0:
                print(f"Found a lone pixel at ({x}, {y}), but it shouldn't be possible!")
                candidates.append((x, y, 'endpoint'))
                continue

            if len(neighbors) > 2:
                candidates.append((x, y, 'junction'))
    
                if diverge_start_x is not None:
                    candidates.append((diverge_start_x, diverge_start_y, 'turn'))

                for nx, ny, nangle in neighbors:
                    queue.append((
                        nx, ny, # next position
                        nangle, # reset base angle to this neighbor's angle from the junction
                        nangle, # reset current angle
                        None,   # reset divergence angle
                        None, None, # reset divergence start point
                    ))
                continue

            if len(neighbors) == 1:
                candidates.append((x, y, 'endpoint'))
                    
                if diverge_start_x is not None:
                    candidates.append((diverge_start_x, diverge_start_y, 'turn'))
                    
                # iterate on neighbors, if e.g. starting point is an endpoint.
                for nx, ny, nangle in neighbors:
                    queue.append((
                        nx, ny, # next position
                        nangle, # reset base angle to this neighbor's angle from the endpoint
                        nangle, # reset current angle
                        None,   # reset divergence angle
                        None, None, # reset divergence start point
                    ))
                continue

            # -----------------------------------------------------------------
            # Now: current pixel is part of a line, with two neighbors exactly.
            # -----------------------------------------------------------------

            # first run!
            if base_angle is None:
                for nx, ny, nangle in neighbors:
                    queue.append((
                        nx, ny,      # next position
                        nangle,  # base angle is the same
                        nangle,  # current angle is the same
                        None,        # the starting divergence angle is None since we have no divergence yet
                        None, None,  # no divergence start point
                    ))
                continue

            angle_diff = angle_difference(curr_angle, base_angle)

            #max_angle_diff_reached = angle_diff > allowed_angle_diff
            
            # If we have diverged, i.e. the current angle is too far from our original angle
            if angle_diff > allowed_angle_diff:
                # failsafe: if no divergence start point was detected, use current point as
                # divergence start point. This probably implies a very sharp turn ?
                if (diverge_start_x is None) or (diverge_start_y is None):
                    diverge_start_x, diverge_start_y = x, y

                if get_closest_candidate_distance( x, y ) > min_node_distance:
                    candidates.append((diverge_start_x, diverge_start_y, 'turn'))

                for nx, ny, nangle in neighbors:
                    queue.append((
                        nx, ny,     # next position
                        nangle,     # reset the base angle to the angle of the divergence start point
                        nangle,     # the next angle is the current base angle, since we're starting anew
                        None,       # the starting divergence angle is None since we have no divergence yet
                        None, None, # no divergence start point
                    ))
                continue

            # -----------------------------------------------------------------
            # we did not diverge (yet ?)
            # -----------------------------------------------------------------

            # Converging !
            # FIXME: doesn't actually detect convergence
            if angle_diff < allowed_angle_diff:
                diverge_start_angle = None
                diverge_start_x, diverge_start_y = None, None                



            for nx, ny, nangle in neighbors:
                rotation_from_base_to_target = angle_difference(nangle, base_angle) / 3

                if rotation_from_base_to_target == 0:
                    rotation_from_base_to_target = angle_difference(curr_angle, base_angle) / 3


                n_diverge_start_angle = diverge_start_angle
                n_diverge_start_x, n_diverge_start_y = diverge_start_x, diverge_start_y
                if (nangle != base_angle) and (n_diverge_start_angle is None):
                    n_diverge_start_angle = nangle
                    n_diverge_start_x, n_diverge_start_y = nx, ny

                if angle_difference(curr_angle, nangle) < rotation_from_base_to_target:
                    next_angle = nangle
                else:
                    next_angle = curr_angle + rotation_from_base_to_target

                queue.append((
                    nx, ny,      # next position
                    base_angle,  # base angle is the same
                    next_angle,  # current angle is the same
                    n_diverge_start_angle,        # the starting divergence angle is None since we have no divergence yet
                    n_diverge_start_x, n_diverge_start_y,  # no divergence start point
                ))
            continue



    for y in range(0, h):
        for x in range(0, w):
            if (skeleton[y, x] == 0) or visited[y, x]:
                continue

            explore_skeleton(x, y)


    #return [ Node(x, y, node_type) for x, y, node_type in candidates ]
    nodes = merge_nearby_nodes(candidates, min_distance)
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

    def find_nearby_nodes(x: int, y: int, start_node: Node) -> Optional[Tuple[Node, List[Tuple[int, int]]]]:
        queue = [(x, y, [(x, y)])]  # (x, y, path to get from start point to end) 
        local_visited = visited.copy() 
        local_visited[y, x] = True #start point is marked as visited

        nearby = []

        while queue:
            cx, cy, path = queue.pop(0)
            # Check for nearby nodes
            skip_queue_item = False
            for node in nodes:
                if node != start_node and abs(node.x - cx) <= 3 and abs(node.y - cy) <= 3:
                    #return node, path + [(node.x, node.y)] #return that node and the path to it
                    nearby.append((node, path + [(node.x, node.y)]))
                    skip_queue_item = True
                    break
            if skip_queue_item:
                continue

            # Add unvisited neighbors to queue
            for nx, ny in get_neighbors(cx, cy):
                if not local_visited[ny, nx]:
                    local_visited[ny, nx] = True
                    queue.append((nx, ny, path + [(nx, ny)]))

        return nearby

    # Process each node
    for start_node in nodes:
        # Find paths from each neighbor of the start node
        for next_x, next_y in get_neighbors(start_node.x, start_node.y):
            if not visited[next_y, next_x]:
                nearby_nodes = find_nearby_nodes(next_x, next_y, start_node)
                for node in nearby_nodes:
                    end_node, path = node
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
    