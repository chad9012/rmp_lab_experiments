import json
import matplotlib.pyplot as plt

# Load the RVG from a JSON file
def load_visibility_graph(file_path):
    with open(file_path, "r") as f:
        visibility_graph_str_keys = json.load(f)

    # Convert the string keys back to tuples
    visibility_graph = {tuple(map(float, k[1:-1].split(','))): v for k, v in visibility_graph_str_keys.items()}
    return visibility_graph

# Visualize the reduced visibility graph
def plot_visibility_graph(obstacles, visibility_graph):
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

    ax.set_aspect('equal')
    plt.title("Visibility Graph")
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Load the RVG (reduced visibility graph) from the file
    visibility_graph = load_visibility_graph("reduced_visibility_graph.json")

    # Placeholder for obstacles (replace with actual obstacle data)
    obstacles = []  # List of shapely Polygon objects or any obstacles you'd like to visualize

    # Visualize the visibility graph
    plot_visibility_graph(obstacles, visibility_graph)
