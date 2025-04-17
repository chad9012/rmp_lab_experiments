import numpy as np
import matplotlib.pyplot as plt
import heapq
import math
import time

# Map dimensions
MAP_WIDTH = 30
MAP_HEIGHT = 20
GRID_RESOLUTION = 1.0  # meters per grid

# Define start and goal positions
START = (2, 2)
GOAL = (25, 15)

def create_map_1():
    """Simple map with a few obstacles"""
    grid = np.zeros((MAP_HEIGHT, MAP_WIDTH))
    grid[5:15, 10] = 1
    grid[5, 10:20] = 1
    grid[10, 15:25] = 1
    grid[14:18, 22] = 1
    return grid

def create_map_2():
    """Maze-like map"""
    grid = np.zeros((MAP_HEIGHT, MAP_WIDTH))
    grid[3:17, 5] = 1
    grid[3:17, 15] = 1
    grid[3:17, 25] = 1
    grid[3, 5:25] = 1
    grid[17, 5:25] = 1
    grid[10, 5:15] = 1
    return grid

def create_map_3():
    """Random obstacles"""
    grid = np.zeros((MAP_HEIGHT, MAP_WIDTH))
    for _ in range(50):
        x, y = np.random.randint(0, MAP_WIDTH), np.random.randint(0, MAP_HEIGHT)
        w, h = np.random.randint(1, 5), np.random.randint(1, 5)
        grid[y:min(y+h, MAP_HEIGHT), x:min(x+w, MAP_WIDTH)] = 1
    return grid

def create_map_4():
    """Open space with a central wall"""
    grid = np.zeros((MAP_HEIGHT, MAP_WIDTH))
    grid[8:12, 5:25] = 1
    return grid

def create_map_5():
    """Complex environment"""
    grid = np.zeros((MAP_HEIGHT, MAP_WIDTH))
    # Vertical walls
    grid[0:20, 5] = 1
    grid[5:20, 15] = 1
    grid[0:15, 25] = 1
    # Horizontal walls
    grid[5, 5:15] = 1
    grid[10, 15:25] = 1
    grid[15, 5:25] = 1
    # Some random blocks
    grid[12:14, 8:10] = 1
    grid[7:9, 18:20] = 1
    grid[17:19, 12:14] = 1
    return grid

def select_map():
    """Let user select which map to use"""
    print("Available maps:")
    print("1. Simple map with a few obstacles")
    print("2. Maze-like map")
    print("3. Random obstacles")
    print("4. Open space with central wall")
    print("5. Complex environment")
    
    while True:
        try:
            choice = int(input("Select map (1-5): "))
            if 1 <= choice <= 5:
                return choice
            print("Please enter a number between 1 and 5")
        except ValueError:
            print("Please enter a valid number")

class Node:
    def __init__(self, x, y, cost, priority, parent=None):
        self.x = x
        self.y = y
        self.cost = cost  # g(n)
        self.priority = priority  # f(n) = g(n) + h(n)
        self.parent = parent

    def __lt__(self, other):
        return self.priority < other.priority

def heuristic(a, b):
    # Euclidean distance
    return math.hypot(b[0] - a[0], b[1] - a[1])

def is_valid(x, y, grid):
    return 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0] and grid[y][x] == 0

def get_neighbors(x, y):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]  # 8 directions
    return [(x + dx, y + dy) for dx, dy in directions]

def a_star(grid, start, goal):
    open_list = []
    closed_set = set()
    start_node = Node(start[0], start[1], 0.0, heuristic(start, goal))
    heapq.heappush(open_list, start_node)

    while open_list:
        current = heapq.heappop(open_list)
        if (current.x, current.y) in closed_set:
            continue
        if (current.x, current.y) == goal:
            return reconstruct_path(current)

        closed_set.add((current.x, current.y))

        for nx, ny in get_neighbors(current.x, current.y):
            if not is_valid(nx, ny, grid) or (nx, ny) in closed_set:
                continue
            cost = current.cost + heuristic((current.x, current.y), (nx, ny))
            priority = cost + heuristic((nx, ny), goal)
            neighbor = Node(nx, ny, cost, priority, current)
            heapq.heappush(open_list, neighbor)
    return None

def reconstruct_path(node):
    path = []
    while node:
        path.append((node.x, node.y))
        node = node.parent
    return path[::-1]  # reverse

class Robot:
    def __init__(self, x, y, theta=0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = 0.0  # linear velocity
        self.w = 0.0  # angular velocity

    def move(self, v, w, dt=0.1):
        self.theta += w * dt
        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt
        self.v = v
        self.w = w

def calc_dynamic_window(robot, config):
    vs = [config["min_speed"], config["max_speed"]]
    ws = [-config["max_yaw_rate"], config["max_yaw_rate"]]
    return vs, ws

def generate_trajectories(robot, dw, config):
    dt = config["dt"]
    predict_time = config["predict_time"]
    trajs = []
    for v in np.linspace(dw[0][0], dw[0][1], 5):
        for w in np.linspace(dw[1][0], dw[1][1], 5):
            traj = []
            temp_robot = Robot(robot.x, robot.y, robot.theta)
            for _ in np.arange(0, predict_time, dt):
                temp_robot.move(v, w, dt)
                traj.append((temp_robot.x, temp_robot.y))
            trajs.append((traj, v, w))
    return trajs

def trajectory_score(traj, goal, grid):
    last = traj[-1]
    gx, gy = goal
    if not is_valid(int(last[0]), int(last[1]), grid):
        return float('inf')  # collision
    return heuristic(last, (gx, gy))

def dwa_control(robot, goal, config, grid):
    dw = calc_dynamic_window(robot, config)
    trajs = generate_trajectories(robot, dw, config)

    min_score = float('inf')
    best_v, best_w = 0.0, 0.0

    for traj, v, w in trajs:
        score = trajectory_score(traj, goal, grid)
        if score < min_score:
            min_score = score
            best_v, best_w = v, w
    return best_v, best_w

def simulate(robot, path, grid):
    config = {
        "max_speed": 1.0,
        "min_speed": -0.5,
        "max_yaw_rate": 1.0,
        "dt": 0.1,
        "predict_time": 1.0
    }

    trajectory = []
    goal_idx = 0

    plt.ion()
    fig, ax = plt.subplots()

    while goal_idx < len(path):
        current_goal = path[goal_idx]
        v, w = dwa_control(robot, current_goal, config, grid)
        robot.move(v, w)
        trajectory.append((robot.x, robot.y))

        if heuristic((robot.x, robot.y), current_goal) < 1.0:
            goal_idx += 1

        # Draw
        ax.clear()
        ax.set_xlim(0, MAP_WIDTH)
        ax.set_ylim(0, MAP_HEIGHT)
        ax.set_title("DWA Path Following")
        ax.imshow(grid, cmap='Greys', origin='lower')

        px, py = zip(*path)
        ax.plot(px, py, '--g', label='Global Path')
        if trajectory:
            tx, ty = zip(*trajectory)
            ax.plot(tx, ty, '-b', label='Robot Trajectory')
        ax.plot(robot.x, robot.y, 'ro', label='Robot')
        ax.legend()
        plt.pause(0.001)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    # Let user select which map to use
    map_choice = select_map()
    
    # Create the selected map
    if map_choice == 1:
        grid_map = create_map_1()
    elif map_choice == 2:
        grid_map = create_map_2()
    elif map_choice == 3:
        grid_map = create_map_3()
    elif map_choice == 4:
        grid_map = create_map_4()
    else:
        grid_map = create_map_5()

    # Find global path
    global_path = a_star(grid_map, START, GOAL)

    if global_path is None:
        print("No path found!")
    else:
        print("Path found, simulating...")
        robot = Robot(START[0], START[1])
        simulate(robot, global_path, grid_map)