import numpy as np
import matplotlib.pyplot as plt
import json

# ----- Load Terrain from File -----
with open("terrain.json", "r") as f:
    terrain_data = json.load(f)

map_size = terrain_data["map_size"]
true_ground = np.array(terrain_data["terrain"])
ground_resolution = 1.0
x_ground = np.linspace(0, map_size * ground_resolution, map_size)

# Configuration
n_steps = 150

# Initial robot state
robot_true_pos = 10.0
robot_estimated_pos = 10.0
direction = 1  # +1 for right, -1 for left
round_trips = 0
last_direction = 1  # to detect turn-around

# Kalman Filter State
x = np.zeros(map_size + 1)
x[0] = robot_estimated_pos
P = np.eye(map_size + 1) * 1.0

Q = np.eye(map_size + 1) * 0.01  # Process noise
R = 0.1  # Measurement noise variance

# History
history_true_pos = []
history_estimated_pos = []
history_measurements = []

for step in range(n_steps):
    # Change direction if hitting boundary
    if robot_true_pos >= (map_size - 1):
        direction = -1
    elif robot_true_pos <= 0:
        direction = 1

    # Count round trips
    if last_direction != direction and direction == 1:
        round_trips += 0.5  # 0.5 for each turn-around
    last_direction = direction

    # ----- Motion -----
    control = direction * 1.0
    robot_true_pos += control + np.random.normal(0, 0.1)

    # ----- Measurement -----
    ground_index = int(robot_true_pos // ground_resolution)
    if 0 <= ground_index < map_size:
        measurement = true_ground[ground_index] + np.random.normal(0, np.sqrt(R))
    else:
        measurement = 0  # Out of bounds

    # ----- EKF Prediction -----
    x[0] += control
    P += Q

    # ----- EKF Update -----
    robot_est_idx = int(x[0] // ground_resolution)
    if 0 <= robot_est_idx < map_size:
        H = np.zeros(map_size + 1)
        H[1 + robot_est_idx] = 1.0  # only affects corresponding ground point

        z_pred = x[1 + robot_est_idx]
        y = measurement - z_pred  # innovation

        S = H @ P @ H.T + R
        K = P @ H.T / S  # Kalman gain

        x += K * y
        P = (np.eye(map_size + 1) - np.outer(K, H)) @ P

    # Store history
    history_true_pos.append(robot_true_pos)
    history_estimated_pos.append(x[0])
    history_measurements.append(measurement)

# Round trips as int
round_trips = int(round_trips)

# -------- Plotting -----------
plt.figure(figsize=(12, 6))

# Plot true and estimated positions
plt.subplot(2, 1, 1)
plt.plot(history_true_pos, label="True Position", color='g')
plt.plot(history_estimated_pos, label="Estimated Position", color='b')
plt.ylabel("X Position")
plt.title(f"Robot Position Over Time | Round Trips: {round_trips}")
plt.legend()
plt.grid(True)

# Plot estimated vs true map
plt.subplot(2, 1, 2)
plt.plot(x_ground, true_ground, label="True Ground Map", color='g')
plt.plot(x_ground, x[1:], label="Estimated Map", color='b', linestyle='--')
plt.xlabel("X")
plt.ylabel("Ground Height")
plt.title("Estimated vs True Ground Map")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
