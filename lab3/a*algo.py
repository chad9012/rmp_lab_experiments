import math
import heapq
import json
import matplotlib.pyplot as plt

# Heuristic function for A* (Euclidean distance)
def heuristic(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

# A* algorithm for path planning
def astar(visibility_graph, start, goal):
    open_list = []
    heapq.heappush(open_list, (0, start))  # Start with a priority queue

    came_from = {}  # For path reconstruction
    g_score = {start: 0}  # Cost from start to node
    f_score = {start: heuristic(start, goal)}  # Estimated cost from start to goal through node

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal:
            # Reconstruct the path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]  # Reverse the path to get it from start to goal

        for neighbor in visibility_graph.get(current, []):
            tentative_g_score = g_score[current] + math.dist(current, neighbor)
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return None  # No path found

# Load the RVG from a JSON file
def load_visibility_graph(file_path):
    with open(file_path, "r") as f:
        visibility_graph_str_keys = json.load(f)

    # Convert the string keys back to tuples
    visibility_graph = {tuple(map(float, k[1:-1].split(','))): v for k, v in visibility_graph_str_keys.items()}
    return visibility_graph

# Visualize the reduced visibility graph and the planned path
def plot_visibility_graph(obstacles, visibility_graph, path=None):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot obstacles
    for poly in obstacles:
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=0.5, fc='gray', ec='black')

    # Plot visibility graph edges
    for vertex, neighbors in visibility_graph.items():
        for neighbor in neighbors:
            xs, ys = zip(vertex, neighbor)
            ax.plot(xs, ys, 'r-', alpha=0.5)

    # Plot the path
    if path:
        path_x, path_y = zip(*path)
        ax.plot(path_x, path_y, marker='o', color='b', label="Path")

    ax.set_aspect('equal')
    plt.title("Path Planning on Reduced Visibility Graph")
    plt.legend()
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Load the RVG (reduced visibility graph) from the file
    visibility_graph = load_visibility_graph("reduced_visibility_graph.json")  

    x_start=1
    y_start=1
    x_goal=250
    y_goal=250
    # Example start and goal vertices (replace with actual points)
    start = (x_start, y_start)  # Example start point
    goal = (x_goal, y_goal)    # Example goal point

    # Plan the path using A* algorithm
    path = astar(visibility_graph, start, goal)

    if path:
        print("Path found:", path)
        # Visualize the RVG and the path
        plot_visibility_graph(obstacles, visibility_graph, path)
    else:
        print("No path found.")
