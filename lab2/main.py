import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy.ndimage import distance_transform_edt

def compute_potential_fields(bitmap, goal, k_att=1.0, inflation_radius=40):
    """
    Compute attractive and repulsive potential fields for path planning.
    
    Parameters:
        bitmap: Binary map (1=free, 0=obstacle)
        goal: Goal position (x, y)
        k_att: Attractive potential gain
        inflation_radius: Radius around obstacles for repulsive potential
        
    Returns:
        Normalized attractive, repulsive, and total potential fields
    """
    height, width = bitmap.shape
    
    # Compute attractive potential (quadratic distance to goal)
    Y, X = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    dist_sq = (X - goal[0])**2 + (Y - goal[1])**2
    att_potential = 0.5 * k_att * dist_sq
    
    # Compute distance from obstacles
    dist_from_obstacle = distance_transform_edt(bitmap)
    
    # Create inflation mask (1 where repulsive field should be applied)
    inflation_mask = (dist_from_obstacle <= inflation_radius).astype(np.uint8)
    
    # Compute repulsive potential (higher when closer to obstacles)
    # Invert the distance field and apply the mask
    rep_potential = (inflation_radius - dist_from_obstacle) * inflation_mask
    rep_potential[rep_potential < 0] = 0
    
    # Normalize fields to [0, 1] range
    att_norm = normalize_field(att_potential)
    rep_norm = normalize_field(rep_potential)
    
    # Compute total potential (sum of attractive and repulsive)
    total_potential = att_norm + rep_norm
    total_norm = normalize_field(total_potential)
    
    return att_norm, rep_norm, total_norm, inflation_mask

def normalize_field(field):
    """Normalize a field to range [0, 1]"""
    min_val = np.min(field)
    max_val = np.max(field)
    if max_val == min_val:
        return np.zeros_like(field)
    return (field - min_val) / (max_val - min_val)

def visualize_potential(bitmap, potential, goal, colormap=cv2.COLORMAP_INFERNO):
    """Create a colored visualization of a potential field"""
    inverted = 1.0 - potential  # Invert for better visualization
    norm = (255 * inverted).astype(np.uint8)
    colored = cv2.applyColorMap(norm, colormap)
    colored[bitmap == 0] = [0, 0, 0]  # Black for obstacles
    
    # Mark goal position
    gx, gy = goal
    radius = 3
    y_min, y_max = max(0, gy-radius), min(bitmap.shape[0], gy+radius+1)
    x_min, x_max = max(0, gx-radius), min(bitmap.shape[1], gx+radius+1)
    colored[y_min:y_max, x_min:x_max] = [0, 0, 255]  # Red goal marker
    
    return colored

def main():
    # Load and process map
    image_path = "lab2/map1.png"
    img = Image.open(image_path).convert("L")
    bitmap = (np.array(img) > 240).astype(np.uint8)
    
    # Set parameters
    height, width = bitmap.shape
    goal = (width // 2, height // 2)
    inflation_radius = 40
    
    # Compute potential fields
    att_norm, rep_norm, total_norm, inflation_mask = compute_potential_fields(
        bitmap, goal, k_att=1.0, inflation_radius=inflation_radius
    )
    
    # Create inflation mask visualization
    inflation_vis = np.ones((height, width, 3), dtype=np.uint8) * 255
    inflation_vis[bitmap == 0] = [0, 0, 0]  # Real obstacles
    inflation_vis[(inflation_mask == 1) & (bitmap != 0)] = [200, 200, 200]  # Inflated region in gray
    
    # Visualize results
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(visualize_potential(bitmap, att_norm, goal))
    axes[0].set_title("Attractive Potential")
    
    axes[1].imshow(visualize_potential(bitmap, rep_norm, goal))
    axes[1].set_title("Repulsive Potential")
    
    axes[2].imshow(visualize_potential(bitmap, total_norm, goal))
    axes[2].set_title("Total Potential")
    
    axes[3].imshow(inflation_vis)
    axes[3].set_title("Inflation Zone")
    
    for ax in axes:
        ax.set_axis_off()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()