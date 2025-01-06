import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional

def check_label_collision(pos1: Tuple[float, float], pos2: Tuple[float, float], 
                         min_distance: float = 10.0) -> bool:
    """Check if two label positions are too close to each other."""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2) < min_distance

def find_label_position(point: Tuple[float, float], existing_labels: List[Tuple[float, float]], 
                       angles: List[float], radius: float, 
                       horizontal_offset: float) -> Tuple[float, float, bool]:
    """
    Find the best position for a label that avoids collisions.
    Returns (x, y, use_point_horizontal) where use_point_horizontal indicates if the 
    horizontal segment should be near the point (True) or near the label (False).
    """
    # Try positions with horizontal segment near point first
    for angle in angles:
        # Position with horizontal line from point
        x = point[0] + horizontal_offset
        y = point[1] + radius * np.sin(angle)
        if not any(check_label_collision((x, y), pos) for pos in existing_labels):
            return x, y, True

    # Try positions with horizontal segment near label
    for angle in angles:
        x = point[0] + radius * np.cos(angle)
        y = point[1] + radius * np.sin(angle)
        if not any(check_label_collision((x, y), pos) for pos in existing_labels):
            return x, y, False
            
    # If no position found, increase radius and try again
    return find_label_position(point, existing_labels, angles, radius * 1.2, horizontal_offset)

def generate_scatter_with_leaders(
    n: int,
    distance_from_point: float = 15,
    distance_from_label: float = 10,
    point_size: float = 50,
    label_font_size: int = 9,
    leader_length_mean: float = 15,
    leader_length_std: float = 5,
    min_label_distance: float = 10,
    title: str = "Scatter Plot",
    xlabel: str = "X-axis",
    ylabel: str = "Y-axis",
    use_label_box: bool = True,
    horizontal_offset: float = 20
) -> pd.DataFrame:
    """
    Generates an improved scatter plot with collision-avoiding labels and leaders.
    
    New features:
    - Label collision detection and avoidance
    - Adaptive leader line lengths based on point density
    - Mixed horizontal and angled leader lines
    - Optional label boxes
    - Improved label placement strategy
    - Input validation
    - Edge case handling
    
    Parameters:
        n (int): Number of points to generate
        distance_from_point (float): Minimum distance from point to leader line
        distance_from_label (float): Minimum distance from label to leader line
        point_size (float): Size of scatter points
        label_font_size (int): Font size for labels
        leader_length_mean (float): Mean length of leader lines
        leader_length_std (float): Standard deviation of leader lengths
        min_label_distance (float): Minimum distance between labels
        title (str): Plot title
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
        use_label_box (bool): Whether to draw boxes around labels
        horizontal_offset (float): Length of horizontal leader line segments
    
    Returns:
        pd.DataFrame: DataFrame with point data and positions
    """
    # Input validation
    if n <= 0:
        raise ValueError("Number of points must be positive")
    if any(param < 0 for param in [distance_from_point, distance_from_label, 
                                 point_size, label_font_size, leader_length_mean]):
        raise ValueError("Distance and size parameters must be positive")

    # Generate random points
    x = np.random.uniform(0, 100, n)
    y = np.random.uniform(0, 100, n)

    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.scatter(x, y, color='blue', s=point_size, label="Data Points")

    # Calculate point density for adaptive leader lengths
    point_density = n / (100 * 100)  # points per unit area
    adaptive_length = leader_length_mean * (1 + np.log1p(point_density))

    # Initialize storage for label positions and data
    existing_labels = []
    data = []
    
    # Generate angles for label placement (8 possible positions around each point)
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)

    # Add labels with leaders
    for i in range(n):
        label = str(i + 1)
        point = (x[i], y[i])
        
        # Find suitable label position and determine leader line style
        label_x, label_y, use_point_horizontal = find_label_position(
            point, 
            existing_labels, 
            angles, 
            adaptive_length,
            horizontal_offset
        )
        existing_labels.append((label_x, label_y))

        # Draw leader lines
        if use_point_horizontal:
            # Horizontal segment near point
            horizontal_x = x[i] + horizontal_offset
            horizontal_y = y[i]
            plt.plot([x[i], horizontal_x], [y[i], horizontal_y], 
                    color='green', linestyle='-', alpha=0.6)
            plt.plot([horizontal_x, label_x], [horizontal_y, label_y], 
                    color='green', linestyle='-', alpha=0.6)
        else:
            # Horizontal segment near label
            horizontal_x = label_x - horizontal_offset
            horizontal_y = label_y
            plt.plot([x[i], horizontal_x], [y[i], horizontal_y], 
                    color='green', linestyle='-', alpha=0.6)
            plt.plot([horizontal_x, label_x], [horizontal_y, label_y], 
                    color='green', linestyle='-', alpha=0.6)

        # Add label with optional background box
        if use_label_box:
            bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            plt.text(label_x, label_y, label, fontsize=label_font_size, 
                    color='darkred', bbox=bbox_props)
        else:
            plt.text(label_x, label_y, label, fontsize=label_font_size, 
                    color='darkred')

        # Calculate image percentages
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        width, height = bbox.width * fig.dpi, bbox.height * fig.dpi

        image_pos_x = 100 * (ax.transData.transform((x[i], y[i]))[0]) / width
        image_pos_y = 100 * (ax.transData.transform((x[i], y[i]))[1]) / height

        data.append({
            "label": label,
            "valueX": x[i],
            "valueY": y[i],
            "posX_image_percent": image_pos_x,
            "posY_image_percent": image_pos_y,
            "label_x": label_x,
            "label_y": label_y,
            "horizontal_leader": use_point_horizontal
        })

    # Adjust plot aesthetics
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    # Add grid with lower opacity
    plt.grid(True, alpha=0.3)
    
    # Ensure all labels are visible
    plt.margins(0.15)
    plt.tight_layout()
    plt.savefig('scatter2.jpg')
    return pd.DataFrame(data)

# Example usage:
if __name__ == "__main__":
    # Test with different configurations
    df1 = generate_scatter_with_leaders(30, use_label_box=True)
    plt.figure()
    df2 = generate_scatter_with_leaders(30, use_label_box=True)
