import cv2
import numpy as np

# Parameters
START = (10, 10)  # (row, col)
GOAL = (699, 699)
MAP_FILE = 'map.png'

# Load and process the map
map_img = cv2.imread(MAP_FILE, cv2.IMREAD_GRAYSCALE)
height, width = map_img.shape
binary_map = (map_img > 127).astype(np.uint8)  # 1 for free, 0 for obstacle

def in_bounds(p):
    x, y = p
    return 0 <= x < height and 0 <= y < width

def is_free(p):
    return in_bounds(p) and binary_map[p] == 1

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def bresenham_line(p0, p1):
    x0, y0 = p0
    x1, y1 = p1
    line = []
    steep = abs(y1 - y0) > abs(x1 - x0)
    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1
    swapped = False
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
        swapped = True
    dx = x1 - x0
    dy = abs(y1 - y0)
    error = dx // 2
    ystep = 1 if y0 < y1 else -1
    y = y0
    for x in range(x0, x1 + 1):
        pt = (y, x) if steep else (x, y)
        line.append(pt)
        error -= dy
        if error < 0:
            y += ystep
            error += dx
    if swapped:
        line.reverse()
    return line

def go_to_goal(curr, goal):
    line = bresenham_line(curr, goal)
    for p in line:
        if not is_free(p):
            return False, p  # Hit point
    return True, goal

def get_contour_containing_point(contours, point):
    for cnt in contours:
        if cv2.pointPolygonTest(cnt, (point[1], point[0]), False) >= 0:
            return cnt
    return None

def wall_follow_using_edges(hit_point, goal, edge_img):
    contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = get_contour_containing_point(contours, hit_point)
    if contour is None:
        return None, []

    contour_points = [tuple(pt[0][::-1]) for pt in contour]  # (row, col)

    # Find the index closest to hit point
    if hit_point in contour_points:
        start_idx = contour_points.index(hit_point)
    else:
        dists = [distance(p, hit_point) for p in contour_points]
        start_idx = int(np.argmin(dists))

    visited = []
    min_dist = float('inf')
    leave_point = hit_point

    n = len(contour_points)
    idx = start_idx
    first_loop = True

    while True:
        pt = contour_points[idx]
        visited.append(pt)

        d = distance(pt, goal)
        if d < min_dist:
            min_dist = d
            leave_point = pt

        idx = (idx + 1) % n  # Clockwise
        if idx == start_idx and not first_loop:
            break
        first_loop = False

    return leave_point, visited

def bug1_with_edge_detection(start, goal, binary_map):
    curr = start
    path = [curr]
    visited_hit_points = set()

    while True:
        success, result = go_to_goal(curr, goal)
        if success:
            path.extend(bresenham_line(curr, goal)[1:])
            break

        hit_point = result
        if hit_point in visited_hit_points:
            print("Goal unreachable. Obstacle loop detected.")
            break
        visited_hit_points.add(hit_point)
        path.append(hit_point)

        obstacle_map = (1 - binary_map) * 255
        edges = cv2.Canny(obstacle_map.astype(np.uint8), 100, 200)

        leave_point, contour_path = wall_follow_using_edges(hit_point, goal, edges)
        if leave_point is None:
            print("No leave point found. Goal unreachable.")
            break

        path.extend(contour_path)
        path.append(leave_point)
        curr = leave_point

    return path, hit_point

def draw_path(image, path, start, hit_point):
    color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw the path
    for p in path:
        cv2.circle(color_img, (p[1], p[0]), 1, (0, 0, 255), -1)

    # Highlight start and goal
    cv2.circle(color_img, (start[1], start[0]), 3, (0, 255, 0), -1)
    cv2.circle(color_img, (GOAL[1], GOAL[0]), 3, (255, 0, 0), -1)

    # Draw the line from start to hit point (collision point)
    cv2.line(color_img, (start[1], start[0]), (hit_point[1], hit_point[0]), (0,0, 255), 2)

    return color_img

if __name__ == "__main__":
    print("Running Bug 1 with edge detection...")
    path, hit_point = bug1_with_edge_detection(START, GOAL, binary_map)
    result_img = draw_path(map_img, path, START, hit_point)
    cv2.imwrite("bug1_edge_path.png", result_img)
    cv2.imshow("Bug 1 with Edge Detection", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
