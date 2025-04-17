import cv2
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString
import math
import json

def angle_between(p1, p2):
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])

def ccw(a, b, c):
    """Check if the points a, b, c make a counter-clockwise turn."""
    return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])

def is_separating(p1, p2, obstacles, min_split_distance=20, extension=1000):
    # Extend line between p1 and p2 to a long line
    p1_np = np.array(p1)
    p2_np = np.array(p2)
    direction = p2_np - p1_np
    length = np.linalg.norm(direction)
    if length == 0:
        return False
    unit_vec = direction / length
    extended_p1 = tuple((p1_np - unit_vec * extension).tolist())
    extended_p2 = tuple((p2_np + unit_vec * extension).tolist())
    extended_line = LineString([extended_p1, extended_p2])

    for obs in obstacles:
        intersection = extended_line.intersection(obs)
        
        if intersection.is_empty:
            continue

        if intersection.geom_type == 'LineString':
            coords = list(intersection.coords)
            if len(coords) >= 2:
                entry, exit = coords[0], coords[-1]
                dist = math.dist(entry, exit)
                if dist > min_split_distance:
                    # Line splits the obstacle
                    return False
        elif intersection.geom_type == 'MultiPoint' or intersection.geom_type == 'GeometryCollection':
            points = [pt for pt in intersection.geoms if pt.geom_type == 'Point']
            if len(points) >= 2:
                dist = points[0].distance(points[-1])
                if dist > min_split_distance:
                    return False

    return True


def construct_reduced_visibility_graph(vertex_dict, obstacles):
    edges = []
    vertex_map = {}  # Will store adjacency list
    print("[INFO] Constructing reduced visibility graph...")
    for i, (obs_id1, p1) in enumerate(vertex_dict):
        if i % 10 == 0:
            print(f"[INFO] Processing vertex {i+1}/{len(vertex_dict)}")
        for j, (obs_id2, p2) in enumerate(vertex_dict):
            if i >= j:
                continue
            same_polygon = (obs_id1 == obs_id2)
            if same_polygon:
                # Only connect consecutive vertices (i.e., polygon edges)
                poly = obstacles[obs_id1]
                coords = list(poly.exterior.coords)
                for k in range(len(coords) - 1):
                    a, b = coords[k], coords[k + 1]
                    if (tuple(a), tuple(b)) in [(p1, p2), (p2, p1)]:
                        edges.append((p1, p2))  # Supporting edge
                        if p1 not in vertex_map:
                            vertex_map[p1] = []
                        vertex_map[p1].append(p2)
                        break
            else:
                # Connect if separating line (clear and doesn't cross obstacles)
                if is_separating(p1, p2, obstacles, min_split_distance=20):
                    edges.append((p1, p2))
                    if p1 not in vertex_map:
                        vertex_map[p1] = []
                    vertex_map[p1].append(p2)
                    if p2 not in vertex_map:
                        vertex_map[p2] = []
                    vertex_map[p2].append(p1)

    print("[INFO] Reduced visibility graph completed.")
    return vertex_map


def plot_visibility_graph(obstacles, visibility_edges, title="Reduced Visibility Graph"):
    print("[INFO] Plotting reduced visibility graph...")
    fig, ax = plt.subplots(figsize=(10, 10))
    for poly in obstacles:
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=0.5, fc='gray', ec='black')
    for p1, p2 in visibility_edges:
        xs, ys = zip(p1, p2)
        ax.plot(xs, ys, 'r-', alpha=0.7)
    for poly in obstacles:
        for x, y in poly.exterior.coords:
            ax.plot(x, y, 'bo')
    ax.set_aspect('equal')
    plt.title(title)
    plt.show()
    print("[INFO] Plotting completed.")

# === Load and Process Image ===
image_path = "lab3/map3.png"
print(f"[INFO] Loading image from {image_path}...")
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}")

print("[INFO] Converting image to binary...")
_, binary = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)

print("[INFO] Finding contours...")
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

obstacles = []
vertex_dict = []  # Stores (obstacle_index, vertex) for each vertex
simplify_tolerance = 10000000

for i, contour in enumerate(contours):
    pts = contour.squeeze().tolist()
    if isinstance(pts[0], int):  # Fix for single-point contour
        continue
    if len(pts) >= 3:
        poly = Polygon(pts).simplify(simplify_tolerance, preserve_topology=True)
        if poly.is_valid and poly.area > 5:
            print(f"[DEBUG] Obstacle {i+1}: {len(list(poly.exterior.coords)) - 1} vertices")
            obstacles.append(poly)
            for v in list(poly.exterior.coords)[:-1]:
                vertex_dict.append((len(obstacles)-1, tuple(v)))

print(f"[INFO] Total obstacles: {len(obstacles)}")
print(f"[INFO] Total simplified vertices: {len(vertex_dict)}")

# === Build Reduced Visibility Graph ===
visibility_graph = construct_reduced_visibility_graph(vertex_dict, obstacles)

# === Save the Reduced Visibility Graph ===
output_file = "reduced_visibility_graph.json"

# Convert tuple keys to strings before saving
visibility_graph_str_keys = {str(k): v for k, v in visibility_graph.items()}

with open(output_file, "w") as f:
    json.dump(visibility_graph_str_keys, f)

print(f"[INFO] Reduced visibility graph saved to {output_file}")
