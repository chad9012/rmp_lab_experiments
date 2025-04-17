import numpy as np
import matplotlib.pyplot as plt

# Number of steps
n = 50

# State: [x, y]
true_pos = np.zeros((n, 2))
measured_pos = np.zeros((n, 2))
estimated_pos = np.zeros((n, 2))

# Initial estimate
x = np.array([0, 0])       # Estimated position
P = np.eye(2)              # Uncertainty covariance matrix

# Control input (velocity in x and y)
u = np.array([1, 0.5])

# Noise parameters
Q = 0.01 * np.eye(2)       # Process noise
R = 1.0 * np.eye(2)        # Measurement noise

# Store initial estimate
estimated_pos[0] = x

# Simulate and estimate
for k in range(1, n):
    # True motion
    true_pos[k] = true_pos[k-1] + u + np.random.multivariate_normal([0, 0], Q)

    # Noisy sensor measurement
    z = true_pos[k] + np.random.multivariate_normal([0, 0], R)
    measured_pos[k] = z

    # ------ Kalman Filter -------
    # Predict
    x = x + u
    P = P + Q

    # Update
    K = P @ np.linalg.inv(P + R)
    x = x + K @ (z - x)
    P = (np.eye(2) - K) @ P

    estimated_pos[k] = x

# --------- Plotting -------------
plt.figure(figsize=(8, 8))
plt.plot(true_pos[:, 0], true_pos[:, 1], 'g-', label="True Position")
plt.plot(measured_pos[:, 0], measured_pos[:, 1], 'rx', label="Measured Position")
plt.plot(estimated_pos[:, 0], estimated_pos[:, 1], 'b-', label="Kalman Estimate")
plt.scatter(true_pos[0, 0], true_pos[0, 1], c='k', label='Start', marker='o')
plt.legend()
plt.grid(True)
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("2D Kalman Filter Robot Localization")
plt.axis("equal")
plt.tight_layout()
plt.show()
