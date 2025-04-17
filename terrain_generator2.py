import numpy as np
import json
import matplotlib.pyplot as plt

# Terrain configuration
map_size = 100
x_ground = np.linspace(0, map_size, map_size)

# Generate wavy terrain
terrain = np.sin(0.2 * x_ground) + 0.1 * np.random.randn(map_size)

# Function to introduce flat region based on the given percentage
def add_flat_region(terrain, flat_percent):
    flat_size = int(flat_percent * map_size)
    flat_start = (map_size - flat_size) // 2
    flat_end = flat_start + flat_size
    flat_value = np.mean(terrain[flat_start:flat_end])
    terrain[flat_start:flat_end] = flat_value
    return terrain, flat_start, flat_end

# Set the flat region percentage (e.g., 40%)
flat_percent = 0.8

# Add the flat region
terrain, flat_start, flat_end = add_flat_region(terrain, flat_percent)

# Save terrain to JSON
terrain_data = {
    "map_size": map_size,
    "terrain": terrain.tolist()
}

with open("terrain.json", "w") as f:
    json.dump(terrain_data, f)

# Optional: Plot the terrain
plt.figure(figsize=(10, 4))
plt.plot(x_ground, terrain, label="Terrain with Flat Region")
plt.axvspan(flat_start, flat_end, color='red', alpha=0.2, label="Flat Region")
plt.xlabel("X")
plt.ylabel("Height")
plt.title(f"Generated Terrain with {int(flat_percent*100)}% Flat Region")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
