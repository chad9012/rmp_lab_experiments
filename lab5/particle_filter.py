import numpy as np
import matplotlib.pyplot as plt
import json

# ----- Load Terrain from File -----
with open("terrain.json", "r") as f:
    terrain_data = json.load(f)

map_size = terrain_data["map_size"]
true_ground = np.array(terrain_data["terrain"])
x_ground = np.linspace(0, map_size, map_size)

# ----- Simulation Setup -----
robot_true_pos = 10.0
direction = 1

# Particle filter parameters
num_particles = 500
particles = np.random.uniform(0, map_size, num_particles)
weights = np.ones(num_particles) / num_particles

process_noise = 0.5
measurement_noise = 0.3

# Tracking
robot_history = []
est_history = []
round_trips = 0
last_direction = direction
estimates_per_round = []

# ----- Particle Filter Functions -----
def move_particles(particles, control, noise_std):
    return particles + control + np.random.normal(0, noise_std, size=particles.shape)

def measurement_prob(measured, particle_positions):
    particle_indices = np.clip(particle_positions.astype(int), 0, map_size - 1)
    expected = true_ground[particle_indices]
    prob = np.exp(-0.5 * ((measured - expected) ** 2) / (measurement_noise ** 2))
    return prob

def resample(particles, weights):
    indices = np.random.choice(len(particles), size=len(particles), p=weights)
    return particles[indices]

# ----- Run Simulation -----
while round_trips < 4:
    # Move robot
    if robot_true_pos >= map_size - 1:
        direction = -1
    elif robot_true_pos <= 0:
        direction = 1

    if direction == 1 and last_direction == -1:
        round_trips += 1
        mean_estimate = np.mean(particles)
        estimates_per_round.append(particles.copy())
        print(f"Round Trip {round_trips} Completed")
    last_direction = direction

    robot_true_pos += direction * 1.0 + np.random.normal(0, process_noise)

    # Measurement from terrain
    ground_idx = int(np.clip(robot_true_pos, 0, map_size - 1))
    measurement = true_ground[ground_idx] + np.random.normal(0, measurement_noise)

    # Particle filter steps
    particles = move_particles(particles, direction * 1.0, process_noise)
    weights = measurement_prob(measurement, particles)
    weights += 1e-10
    weights /= np.sum(weights)
    particles = resample(particles, weights)

    # Log
    robot_history.append(robot_true_pos)
    est_history.append(np.mean(particles))

# ----- Plotting -----
plt.figure(figsize=(15, 10))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.plot(x_ground, true_ground, label="True Terrain", color='green')
    plt.hist(estimates_per_round[i], bins=50, density=True, alpha=0.6, label=f"Particles After Round {i+1}", color='blue')
    plt.xlim(0, map_size)
    plt.ylim(-0.1, 1)
    plt.title(f"Particle Distribution After Round Trip {i+1}")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# Plot estimated vs true position over time
plt.figure(figsize=(12, 4))
plt.plot(robot_history, label="True Position", color='green')
plt.plot(est_history, label="Estimated Position (Mean of Particles)", color='blue')
plt.title("Robot Position vs Estimated Position Over Time")
plt.xlabel("Time Steps")
plt.ylabel("X Position")
plt.legend()
plt.grid(True)
plt.show()