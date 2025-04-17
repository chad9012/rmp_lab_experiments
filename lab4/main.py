import numpy as np
import matplotlib.pyplot as plt

# Number of time steps
n = 50

# Initialize arrays
true_position = np.zeros(n)
measured_position = np.zeros(n)
estimated_position = np.zeros(n)
estimate_uncertainty = np.zeros(n)

# Initial true position
true_position[0] = 0

# Kalman Filter parameters
x = 0                      # Initial estimate
P = 1                      # Initial uncertainty
u = 1                      # Control input (velocity per step)
Q = 0.01                   # Process noise covariance
R = 1                      # Measurement noise covariance

# Store initial estimates
estimated_position[0] = x
estimate_uncertainty[0] = P

# Simulate and estimate
for k in range(1, n):
    # Simulate robot motion
    true_position[k] = true_position[k-1] + u + np.random.normal(0, np.sqrt(Q))

    # Simulate measurement
    measured_position[k] = true_position[k] + np.random.normal(0, np.sqrt(R))

    # -------- Kalman Filter --------
    # Predict
    x = x + u
    P = P + Q

    # Update
    K = P / (P + R)
    x = x + K * (measured_position[k] - x)
    P = (1 - K) * P

    # Store
    estimated_position[k] = x
    estimate_uncertainty[k] = P

# -------- Plotting --------
plt.figure(figsize=(10, 6))
plt.plot(true_position, label="True Position", color='g')
plt.plot(measured_position, label="Measured Position (Noisy)", linestyle='dotted', color='r')
plt.plot(estimated_position, label="Kalman Estimate", color='b')
plt.fill_between(range(n),
                 estimated_position - np.sqrt(estimate_uncertainty),
                 estimated_position + np.sqrt(estimate_uncertainty),
                 color='blue', alpha=0.2, label='Estimate ±1σ')
plt.legend()
plt.xlabel("Time step")
plt.ylabel("Position")
plt.title("1D Kalman Filter Robot Localization")
plt.grid(True)
plt.tight_layout()
plt.show()
