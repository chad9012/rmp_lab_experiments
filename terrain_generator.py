import numpy as np
import matplotlib.pyplot as plt
import json

# ----- Create Wavy Terrain and Save to File -----
map_size = 100
x_ground = np.linspace(0, map_size, map_size)
true_ground = np.sin(0.2 * x_ground) + 0.2 * np.random.randn(map_size)

# Save to JSON file
terrain_data = {
    "map_size": map_size,
    "terrain": true_ground.tolist()
}

with open("terrain.json", "w") as f:
    json.dump(terrain_data, f)

# Plot the terrain
plt.figure(figsize=(10, 4))
plt.plot(x_ground, true_ground, label="Generated Terrain")
plt.title("Generated Terrain (Saved to 'terrain.json')")
plt.xlabel("X Position")
plt.ylabel("Height")
plt.grid(True)
plt.legend()
plt.show()