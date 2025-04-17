import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math

def is_free(bitmap, point):
    x, y = int(round(point[0])), int(round(point[1]))
    if y < 0 or y >= bitmap.shape[0] or x < 0 or x >= bitmap.shape[1]:
        return False
    return bitmap[y, x] == 1

def is_edge(bitmap, point):
    if not is_free(bitmap, point):
        return False
    x, y = int(round(point[0])), int(round(point[1]))
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

def on_m_line(start, goal, point, epsilon=1.0):
    start = np.array(start)
    goal = np.array(goal)
    point = np.array(point)
    line_vec = goal - start
    point_vec = point - start
    if np.linalg.norm(line_vec) == 0:
        return False
    proj_length = np.dot(point_vec, line_vec) / np.linalg.norm(line_vec)
    proj_point = start + proj_length * (line_vec / np.linalg.norm(line_vec))
    return np.linalg.norm(proj_point - point) < epsilon

def bug2_algorithm(bitmap, start, goal, step=1.0, tol=2.0, max_iters=100000):
    current = np.array(start, dtype=float)
    goal_arr = np.array(goal, dtype=float)
    path = [tuple(current)]
    intersect_points = []
    iter_count = 0

    while np.linalg.norm(goal_arr - current) > tol and iter_count < max_iters:
        iter_count += 1

        # Move directly along m-line
        direction = goal_arr - current
        dist_to_goal = np.linalg.norm(direction)
        if dist_to_goal == 0:
            break
        direction = direction / dist_to_goal
        next_point = current + step * direction

        if is_free(bitmap, next_point):
            current = next_point
            path.append(tuple(current))
            continue

        # Hit point
        hit_point = current.copy()
        print(f"Hit obstacle at {hit_point}. Starting wall following...")

        # Wall following
        wall_dir = np.array([direction[1], -direction[0]])
        leave_found = False
        leave_point = None
        hit_dist_to_goal = np.linalg.norm(goal_arr - hit_point)

        # Track loop
        loop_start = tuple(np.round(hit_point, 1))
        loop_closed = False

        while iter_count < max_iters:
            iter_count += 1
            moved = False

            for angle in np.linspace(-math.pi/2, 3*math.pi/4, 24):
                rot = np.array([
                    [math.cos(angle), -math.sin(angle)],
                    [math.sin(angle), math.cos(angle)]
                ])
                test_dir = rot @ wall_dir
                test_point = current + step * test_dir

                if is_free(bitmap, test_point) and is_edge(bitmap, test_point):
                    current = test_point
                    wall_dir = test_dir
                    path.append(tuple(current))
                    moved = True

                    # Check for re-intersection with m-line
                    if on_m_line(start, goal, current):
                        dist_to_goal = np.linalg.norm(goal_arr - current)
                        if dist_to_goal < hit_dist_to_goal:
                            leave_found = True
                            leave_point = current.copy()
                            intersect_points.append(tuple(current))  # Save for purple dot
                            print(f"Found leave point on m-line at {leave_point}")
                            break
                    break

            if leave_found:
                current = leave_point.copy()
                path.append(tuple(current))
                break

            if not moved:
                print("Wall following stuck. No movement found.")
                return path, intersect_points

            # Check loop closure
            if np.linalg.norm(current - hit_point) < step and iter_count > 10:
                print("Completed loop around obstacle. Goal unreachable.")
                loop_closed = True
                break

        if loop_closed:
            break

    if np.linalg.norm(goal_arr - current) <= tol:
        print(f"Goal reached at {current}")
    elif iter_count >= max_iters:
        print("Max iterations reached.")

    return path, intersect_points

def main():
    image_path = "map.png"
    img = Image.open(image_path).convert("L")
    arr = np.array(img)
    bitmap = (arr > 240).astype(int)
    height, width = bitmap.shape

    start = (0, 0)
    goal = (width - 1, height - 1)

    print(f"Running Bug2 algorithm from {start} to {goal} ...")
    path, intersect_points = bug2_algorithm(bitmap, start, goal, step=1.0, tol=2.0)
    path = np.array(path)

    plt.figure(figsize=(8, 8))
    plt.imshow(bitmap, cmap='gray', origin='upper')

    # Plot m-line (green)
    plt.plot([start[0], goal[0]], [start[1], goal[1]], 'g--', linewidth=1.5, label="M-Line")

    # Plot path (red)
    if path.shape[0] > 0:
        plt.plot(path[:, 0], path[:, 1], 'r.-', label="Bug2 Path")

    # Plot m-line-edge intersections (purple)
    if len(intersect_points) > 0:
        ip_arr = np.array(intersect_points)
        plt.plot(ip_arr[:, 0], ip_arr[:, 1], 'mo', markersize=6, label="M-Line Intersections")

    plt.plot(start[0], start[1], "go", label="Start")
    plt.plot(goal[0], goal[1], "bo", label="Goal")
    plt.legend()
    plt.title("Bug2 Algorithm: Path with M-Line and Intersections")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
