import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math

def is_free(bitmap, point):
    """
    Determines if a given (x,y) point is in free space.
    The bitmap is a 2D numpy array where 1 = free (white) and 0 = obstacle (black).
    
    As the bitmap is indexed as [row, col] and we treat point as (x, y),
    we use:
       col = int(round(x)) and row = int(round(y)).
    """
    x, y = int(round(point[0])), int(round(point[1]))
    if y < 0 or y >= bitmap.shape[0] or x < 0 or x >= bitmap.shape[1]:
        return False
    return bitmap[y, x] == 1

def is_edge(bitmap, point):
    """
    Determines if a point is on the edge of an obstacle.
    A point is on an edge if it's free and has at least one neighboring point that's an obstacle.
    """
    if not is_free(bitmap, point):
        return False
    
    x, y = int(round(point[0])), int(round(point[1]))
    # Check 8-connected neighbors
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if (ny >= 0 and ny < bitmap.shape[0] and 
                nx >= 0 and nx < bitmap.shape[1] and 
                bitmap[ny, nx] == 0):
                return True
    return False

def tangent_bug_algorithm(bitmap, start, goal, step=1.0, tol=2.0, max_iters=100000):
    """
    Implements the Tangent Bug algorithm using the following logic:
    
    1. Go-to-Goal (GTG): Move straight toward the goal until an obstacle is encountered.
    2. Hit Point (H): Record the hit point when an obstacle is encountered.
    3. Wall Following: Follow the boundary (clockwise) until returning to the hit point.
       While following, record the boundary point that is closest to the goal (Leave Point, L).
    4. Resume GTG: Move from the leave point toward the goal.
    
    Parameters:
      bitmap  : 2D binary numpy array (1 = free, 0 = obstacle)
      start   : Starting point as (x, y)
      goal    : Goal point as (x, y)
      step    : Movement increment (in pixels)
      tol     : Tolerance distance (in pixels) to detect reaching goal or hit point
      max_iters: Maximum iterations to avoid infinite loops
    
    Returns:
      path    : List of (x,y) coordinates representing the robot's path.
    """
    current = np.array(start, dtype=float)
    goal_arr = np.array(goal, dtype=float)
    path = [tuple(current)]
    iter_count = 0
    
    # Main loop
    while np.linalg.norm(goal_arr - current) > tol and iter_count < max_iters:
        iter_count += 1
        
        # ---- 1. Go-to-Goal (GTG) ----
        direction = goal_arr - current
        dist_to_goal = np.linalg.norm(direction)
        if dist_to_goal == 0:
            break  # Reached goal
            
        direction = direction / dist_to_goal  # Normalize
        next_point = current + step * direction
        
        # Check if we can move directly toward the goal
        if is_free(bitmap, next_point):
            current = next_point
            path.append(tuple(current))
            continue  # Continue moving toward goal
        
        # ---- 2. Hit Point (H) ----
        # We've encountered an obstacle
        hit_point = current.copy()
        print(f"Hit obstacle at {hit_point}. Starting wall following...")
        
        # ---- 3. Wall Following ----
        # Initialize variables for wall following
        leave_point = hit_point.copy()
        min_dist_to_goal = np.linalg.norm(goal_arr - hit_point)
        
        # Find the initial direction to follow the wall (90° clockwise from goal direction)
        wall_dir = np.array([direction[1], -direction[0]])
        
        # Find the first edge point to start wall following
        edge_found = False
        for search_angle in np.linspace(0, 2*math.pi, 36):
            rot = np.array([
                [math.cos(search_angle), -math.sin(search_angle)],
                [math.sin(search_angle), math.cos(search_angle)]
            ])
            test_dir = rot @ direction
            test_point = current + step * test_dir
            
            if not is_free(bitmap, test_point):
                # Found an obstacle, now look for an adjacent edge point
                for edge_angle in np.linspace(0, 2*math.pi, 36):
                    edge_rot = np.array([
                        [math.cos(edge_angle), -math.sin(edge_angle)],
                        [math.sin(edge_angle), math.cos(edge_angle)]
                    ])
                    edge_dir = edge_rot @ test_dir
                    edge_point = current + step * edge_dir
                    
                    if is_free(bitmap, edge_point) and is_edge(bitmap, edge_point):
                        current = edge_point
                        path.append(tuple(current))
                        wall_dir = edge_dir
                        edge_found = True
                        break
                
                if edge_found:
                    break
        
        if not edge_found:
            print("Could not find edge to follow. Goal may be unreachable.")
            return path
        
        # Start wall following until we return to the hit point
        first_step = True  # Flag to prevent immediate termination
        while iter_count < max_iters:
            iter_count += 1
            
            # Check if we've returned to the hit point (completed circumnavigation)
            # Only check after we've moved away from the hit point
            if not first_step and np.linalg.norm(current - hit_point) < tol:
                print(f"Completed obstacle circumnavigation. Moving to leave point at {leave_point}")
                break
            
            first_step = False
            
            # Try to follow the edge by checking directions in a specific order
            # Start by trying to turn right (keeping obstacle on left), then straight, then left
            moved = False
            
            # Search angles from -90° (right) to +135° (left) relative to current direction
            for angle in np.linspace(-math.pi/2, 3*math.pi/4, 24):
                rot = np.array([
                    [math.cos(angle), -math.sin(angle)],
                    [math.sin(angle), math.cos(angle)]
                ])
                test_dir = rot @ wall_dir
                test_point = current + step * test_dir
                
                # Check if this point is on the edge
                if is_free(bitmap, test_point) and is_edge(bitmap, test_point):
                    # Move to this edge point
                    current = test_point
                    path.append(tuple(current))
                    wall_dir = test_dir  # Update direction for next iteration
                    
                    # Check if this point is closer to the goal
                    dist_to_goal = np.linalg.norm(goal_arr - current)
                    if dist_to_goal < min_dist_to_goal:
                        min_dist_to_goal = dist_to_goal
                        leave_point = current.copy()
                        print(f"New leave point found at {leave_point}, distance to goal: {min_dist_to_goal}")
                    
                    moved = True
                    break
            
            if not moved:
                print("Wall following stuck; cannot find next edge point.")
                return path
        
        # ---- 4. Resume GTG from leave point ----
        # Move to the leave point (if we're not already there)
        if np.linalg.norm(current - leave_point) > tol:
            current = leave_point.copy()
            path.append(tuple(current))
            print(f"Moving to leave point at {leave_point}")
    
    if iter_count >= max_iters:
        print("Maximum iterations reached. Goal may be unreachable.")
    else:
        print(f"Goal reached at {current}")
    
    return path

def main():
    # --- 1. Load the saved image ---
    image_path = "lab1/map1.png"  # Change to your saved image name/path
    img = Image.open(image_path).convert("L")  # convert to grayscale

    # --- 2. Convert image to binary bitmap ---
    # Here, pixels with value > 240 are considered free space.
    arr = np.array(img)
    bitmap = (arr > 240).astype(int)
    height, width = bitmap.shape

    # --- 3. Define start and goal ---
    # Default start: top-left; goal: bottom-right.
    start = (0, 0)  # (x, y)
    goal = (width - 1, height - 1)

    print(f"Running Tangent Bug algorithm from {start} to {goal} ...")
    path = tangent_bug_algorithm(bitmap, start, goal, step=1.0, tol=2.0)
    path = np.array(path)

    # --- 4. Visualize the result ---
    plt.figure(figsize=(8, 8))
    plt.imshow(bitmap, cmap='gray', origin='upper')
    if path.shape[0] > 0:
        plt.plot(path[:, 0], path[:, 1], 'r.-', label="Tangent Bug Path")
    plt.plot(start[0], start[1], "go", label="Start")
    plt.plot(goal[0], goal[1], "bo", label="Goal")
    plt.legend()
    plt.title("Tangent Bug Algorithm: Path from Top-Left to Bottom-Right")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

if __name__ == "__main__":
    main()
