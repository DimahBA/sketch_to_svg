import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set

import math
import time
import uuid
import random

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
    nodes = find_critical_points(skeleton, 6)
    
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

def merge_nearby_nodes(candidates: List[Tuple[int, int, str]], min_distance: float) -> List[Node]:
    """Merge nodes that are closer than min_distance to each other.
    Priority order: junctions > turns > endpoints
    
    Args:
        candidates: List of tuples (x, y, node_type) representing node candidates
        min_distance: Minimum Euclidean distance between nodes to trigger merging
        
    Returns:
        List of merged Node objects
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
            
        elif node_type == 'turn':
            # If there's a junction nearby, skip this turn point
            if any(nt == 'junction' for _, _, nt in nearby):
                continue
                
            # Filter nearby points to only include turns
            nearby_turns = [(nx, ny, nt) for nx, ny, nt in nearby if nt == 'turn']
            
            if not nearby_turns:
                # If no other turn points nearby, keep this one
                nodes.append(Node(x, y, 'turn'))
                continue
                
            # Get all turn points including current one
            all_turn_points = [(x, y)] + [(nx, ny) for nx, ny, _ in nearby_turns]
            
            # For each turn point, compute its angle using analyze_local_structure
            # We need to look at the surrounding pixels to get the angle
            # This requires access to the skeleton image, so we'll rely on the angles 
            # being similar if the turns are close and pointing in similar directions
            
            # Choose the turn point that maximizes distance from existing nodes
            # while being central to other nearby turns
            best_point = max(all_turn_points, key=lambda p:
                sum(euclidean_distance(p, (n.x, n.y)) for n in nodes) +
                -sum(euclidean_distance(p, (op[0], op[1])) for op in all_turn_points) / len(all_turn_points)
            )
            nodes.append(Node(best_point[0], best_point[1], 'turn'))
            
        else:  # endpoint
            # If there's a junction or turn nearby, skip this endpoint
            if any(nt in ['junction', 'turn'] for _, _, nt in nearby):
                continue
            # Otherwise, keep the most isolated endpoint
            all_points = [(x, y)] + [(nx, ny) for nx, ny, _ in nearby]
            # Choose point furthest from all other nodes
            best_point = max(all_points, key=lambda p: 
                sum(euclidean_distance(p, (n.x, n.y)) for n in nodes))
            nodes.append(Node(best_point[0], best_point[1], 'endpoint'))
    
    return nodes

def find_critical_points(skeleton: np.ndarray, min_distance: float = 12.0) -> List[Node]:
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

    # tested couples that seem to work (cannot explain why though):
    # 120° and 2
    # 25° and 12

    allowed_angle_diff = 45 # degrees
    rotation_inertia = 12


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

        #neighbor_angles_yx = {
        #    -1: { -1: 225, 0: 270, 1: 315 },
        #     0: { -1: 180,         1:   0 }, # 0,0 is not possible
        #     1: { -1: 135, 0:  90, 1:  45 },
        #}

        neighbor_angles_yx = {
            -1: { -1: 225, 0: 270, 1: 315 },
             0: { -1: 180,         1: 0 },   # 0,0 is not possible
             1: { -1: 135, 0: 90,  1: 45 },
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

    def signed_angle_difference(angle1, angle2):
        """
        Calculate the signed angle difference between two angles in degrees.
        Positive means angle2 is counter-clockwise from angle1.
        Negative means angle2 is clockwise from angle1.
        Returns value in range [-180, 180]
        """
        # Normalize difference to [-180, 180]
        #diff = (angle2 - angle1) % 360
        #if diff > 180:
        #    diff -= 360
        #return diff
        diff = angle2 - angle1
        if diff > 180:
            diff -= 360
        if diff < -180:
            diff += 360
        return diff


    def get_closest_candidate_distance( from_x: int, from_y: int ):
        return min([
            math.sqrt( (x - from_x)**2 + (y - from_y)**2 ) for x, y, node_type in candidates
        ])

    def get_pixel_at_angle(x, y, angle_degrees, distance):
        # Convert the angle from degrees to radians
        angle_radians = math.radians(angle_degrees)
        
        # Calculate new coordinates
        new_x = x + distance * math.cos(angle_radians)
        new_y = y + distance * math.sin(angle_radians)
        
        # Return the new pixel coordinates
        return round(new_x), round(new_y)

    def explore_skeleton( start_x: int, start_y: int ):
        arrows = {}
        ARROW_LENGTH = 20

        def random_color():
            return (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )

        def get_arrow_target(x, y, angle):
            return get_pixel_at_angle(x, y, angle, ARROW_LENGTH)


        def draw_img():
            img = skeleton.copy()
            img = cv2.bitwise_not(img)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


            # Draw nodes
            for x, y, node_type in candidates:
                # Different colors for different node types
                color = {
                    'junction': (0, 0, 255),    # Red
                    'endpoint': (0, 255, 0),    # Green
                    'turn': (0, 255, 255),      # Yellow
                    'DEBUG': (255, 0, 0)         # special debug node
                }.get(node_type, (255, 255, 255))
                
                # Draw node with larger radius for visibility
                cv2.circle(img, (x, y), 4, color, -1)
                cv2.circle(img, (x, y), 5, color, 1)

            for arrow_id, arrow_data in arrows.items():
                base = arrow_data['base']
                angle = arrow_data['angle']
                color = arrow_data['color']
                head = get_arrow_target(base[0], base[1], angle)

                cv2.arrowedLine(img, base, head, color, 4)
                #cv2.circle(img, base, min_distance, (255, 0, 255), 2)


            cv2.imshow("Cool animation demo window", img)
            key = cv2.waitKey(1)

            if key != -1:
                print("AHH I'M DYING :sob:")
                cv2.destroyAllWindows()
                exit()


        INITIAL_DEPTH = 1


        narrow_id = uuid.uuid4()
        narrow_color = random_color()
        narrow_base = (start_x, start_y)
        narrow_angle = 90

        arrows[narrow_id] = {'base': narrow_base, 'angle': narrow_angle, 'color': narrow_color}


        #angle = None
        #visited[start_x, start_y] = True
        queue = [( 
            start_x, start_y, # next position
            None, # reset base angle
            None, # reset current angle
            None,   # reset divergence angle
            None, None, # reset divergence start point
            narrow_id # no arrow to start with
        )]

        last_time = time.time_ns() // 1_000_000
        
        FPS = 30
        MS_PER_FRAME = 1000 / FPS

        draw_img()
        cv2.waitKey(5000)

        while queue:

            curr_time = time.time_ns() // 1_000_000
            #if curr_time - last_time <= MS_PER_FRAME:
            #    time.sleep( curr_time - last_time )


            x, y, base_angle, curr_angle, diverge_start_angle, diverge_start_x, diverge_start_y, arrow_id = queue.pop(0)

            if visited[y, x]:
                if arrow_id:
                    arrows.pop(arrow_id, None)
                continue
            visited[y, x] = True


            neighbors = get_neighbors(x, y)

            # edge case: lone pixel ?
            # should not be possible
            if len(neighbors) == 0:
                print(f"Found a lone pixel at ({x}, {y}), but it shouldn't be possible!")
                arrows.pop(arrow_id, None)
                candidates.append((x, y, 'endpoint'))
                draw_img()
                continue

            if len(neighbors) > 2:
                candidates.append((x, y, 'junction'))
                draw_img()
    
                if diverge_start_x is not None:
                    candidates.append((diverge_start_x, diverge_start_y, 'turn'))

                arrows.pop(arrow_id, None)

                for nx, ny, nangle in neighbors:
                    if visited[ny, nx]:
                        continue

                    narrow_id = uuid.uuid4()
                    narrow_color = random_color()
                    narrow_base = (nx, ny)
                    narrow_angle = nangle % 360

                    arrows[narrow_id] = {'base': narrow_base, 'angle': narrow_angle, 'color': narrow_color}

                    queue.append((
                        nx, ny, # next position
                        nangle % 360, # reset base angle to this neighbor's angle from the junction
                        nangle % 360, # reset current angle
                        None,   # reset divergence angle
                        None, None, # reset divergence start point
                        narrow_id # split a new arrow along each line
                    ))
                continue

            if len(neighbors) == 1:
                candidates.append((x, y, 'endpoint'))
                draw_img()
                    
                if diverge_start_x is not None:
                    candidates.append((diverge_start_x, diverge_start_y, 'turn'))
                    
                arrows_made = 0
                # iterate on neighbors, if e.g. starting point is an endpoint.
                for nx, ny, nangle in neighbors:
                    if visited[ny, nx]:
                        continue

                    if arrow_id:
                        arrow = arrows[arrow_id]
                        arrow['base'] = (nx, ny)
                        arrow['angle'] = nangle % 360
                        arrows_made += 1

                    queue.append((
                        nx, ny, # next position
                        nangle % 360, # reset base angle to this neighbor's angle from the endpoint
                        nangle % 360, # reset current angle
                        None,   # reset divergence angle
                        None, None, # reset divergence start point
                        arrow_id # keep same arrow_id
                    ))
                if arrows_made == 0:
                    arrows.pop(arrow_id, None)
                continue

            # -----------------------------------------------------------------
            # Now: current pixel is part of a line, with two neighbors exactly.
            # -----------------------------------------------------------------

            # first run!
            if base_angle is None:
                # FIXME: try to put a turn here to fix that we might be starting on a turn and not detect it
                candidates.append((x, y, 'turn'))

                #arrows.pop(arrow_id, None)


                for nx, ny, nangle in neighbors:
                    if visited[ny, nx]:
                        continue

                    narrow_id = uuid.uuid4()
                    narrow_color = random_color()
                    narrow_base = (nx, ny)
                    narrow_angle = nangle % 360

                    arrows[narrow_id] = {'base': narrow_base, 'angle': narrow_angle, 'color': narrow_color}


                    queue.append((
                        nx, ny,      # next position
                        nangle % 360,  # base angle is the same
                        nangle % 360,  # current angle is the same
                        None,        # the starting divergence angle is None since we have no divergence yet
                        None, None,  # no divergence start point
                        narrow_id
                    ))
                continue

            draw_img( )
            

            angle_diff = signed_angle_difference(base_angle, curr_angle)
            #print(angle_diff)

            DAMPENING_FACTOR = 0.1


            next_base_angle = (base_angle + (angle_diff * DAMPENING_FACTOR)) % 360

            #if abs(angle_diff) > allowed_angle_diff:
            #    print(f"NYOOM at {x}, {y}")

            #base_arrow = ( (x, y), get_pixel_at_angle(x, y, base_angle, 20) )
            #angle_arrow = ( (x, y), get_pixel_at_angle(x, y, curr_angle, 20) )



            #max_angle_diff_reached = angle_diff > allowed_angle_diff
            
            # If we have diverged, i.e. the current angle is too far from our original angle
            if abs(angle_diff) > allowed_angle_diff:
                # failsafe: if no divergence start point was detected, use current point as
                # divergence start point. This probably implies a very sharp turn ?
                if (diverge_start_x is None) or (diverge_start_y is None):
                    diverge_start_x, diverge_start_y = x, y

                #if get_closest_candidate_distance( x, y ) > min_distance:
                    #candidates.append((diverge_start_x, diverge_start_y, 'turn'))
                    pass

                distance_from_nearest_node = math.sqrt( (x - diverge_start_x)**2 + (y - diverge_start_y)**2 )

                if distance_from_nearest_node > min_distance:
                    #candidates.append((diverge_start_x, diverge_start_y, 'turn'))
                    candidates.append((x, y, 'turn'))
                #elif distance_from_nearest_node > 4:
                #    candidates.append((x, y, 'DEBUG'))
                else:
                    pass #candidates.append((diverge_start_x, diverge_start_y, 'turn'))
                

                diverge_start_angle = diverge_start_angle or curr_angle

                #arrows.pop(arrow_id, None)


                nb_arrows_made = 0
                for nx, ny, nangle in neighbors:
                    if visited[ny, nx]:
                        continue


                    if nb_arrows_made == 0:
                        narrow_id = arrow_id
                        arrow = arrows[arrow_id]
                        arrow['base'] = (nx, ny)
                        arrow['angle'] = next_base_angle % 360
                    else:
                        narrow_id = uuid.uuid4()
                        narrow_color = random_color()
                        narrow_base = (nx, ny)
                        narrow_angle = next_base_angle % 360

                        arrows[narrow_id] = {'base': narrow_base, 'angle': narrow_angle, 'color': narrow_color}
                    nb_arrows_made += 1


                    queue.append((
                        nx, ny,     # next position
                        next_base_angle,     # reset the base angle to the angle of the divergence start point
                        nangle,     # the next angle is the current base angle, since we're starting anew
                        None,       # the starting divergence angle is None since we have no divergence yet
                        None, None, # no divergence start point
                        narrow_id
                    ))
                continue

            # -----------------------------------------------------------------
            # we did not diverge (yet ?)
            # -----------------------------------------------------------------

            # Check for convergence - if we're getting closer to the base angle
            if abs(signed_angle_difference(curr_angle, base_angle)) < 3:  # Small threshold for convergence
                # Reset current angle to base angle since we're converging
                curr_angle = base_angle


            #arrows.pop(arrow_id, None)

            nb_arrows_made = 0
            for nx, ny, nangle in neighbors:
                if visited[ny, nx]:
                    continue

                if nb_arrows_made == 0:
                    narrow_id = arrow_id
                    arrow = arrows[arrow_id]
                    arrow['base'] = (nx, ny)
                    arrow['angle'] = next_base_angle
                else:
                    narrow_id = uuid.uuid4()
                    narrow_color = random_color()
                    narrow_base = (nx, ny)
                    narrow_angle = next_base_angle

                    arrows[narrow_id] = {'base': narrow_base, 'angle': narrow_angle, 'color': narrow_color}
                nb_arrows_made += 1

                n_diverge_start_angle = diverge_start_angle
                n_diverge_start_x, n_diverge_start_y = diverge_start_x, diverge_start_y

                if (abs(signed_angle_difference(nangle, base_angle)) != 0) and (n_diverge_start_angle is None):
                    n_diverge_start_angle = nangle
                    n_diverge_start_x, n_diverge_start_y = nx, ny

                if n_diverge_start_angle is not None:
                    n_diverge_start_angle %= 360

                rotation_from_base_to_target = signed_angle_difference(nangle, base_angle) / rotation_inertia

                if abs(signed_angle_difference(curr_angle, nangle)) < abs(rotation_from_base_to_target):
                    next_angle = nangle
                else:
                    next_angle = curr_angle + rotation_from_base_to_target

                queue.append((
                    nx, ny,      # next position
                    next_base_angle,  
                    nangle, 
                    n_diverge_start_angle,
                    n_diverge_start_x, n_diverge_start_y,
                    narrow_id
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
    