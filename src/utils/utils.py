import cv2
import numpy as np

def draw_overlay(image, mask, points=None):
    """
    Draw the mask boundary and optionally overlay points on the image.

    Args:
        image (np.ndarray): Original RGB image.
        mask (np.ndarray): Binary mask.
        points (list or None): List of points with classification or None.

    Returns:
        np.ndarray: Overlay image.
    """
    overlay = image.copy()
    mask = (mask * 255).astype(np.uint8) if mask.dtype != np.uint8 else mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), thickness=2)  # Draw mask boundary in blue

    if points is not None:
        for x, y, classification in points:
            if classification == "positive":
                color = (0, 255, 0)  # Green for positive
            elif classification == "negative":
                color = (255, 0, 0)  # Blue for negative
            elif classification == "neutral":
                color = (255, 255, 0)  # Yellow for neutral
            else:
                continue
            cv2.circle(overlay, (x, y), radius=2, color=color, thickness=-1)
    return overlay

def draw_points_on_cropped_region(cropped_image, points, output_path):
    """
    Draw points on the cropped region for debugging purposes.

    Args:
        cropped_image (np.ndarray): Cropped RGB image.
        points (list): List of points with classification.
        output_path (str): Path to save the visualization.
    """
    debug_image = cropped_image.copy()
    for x, y, cls in points:
        color = (0, 255, 0) if cls == "positive" else (255, 0, 0) if cls == "negative" else (255, 255, 0)
        cv2.circle(debug_image, (x, y), radius=5, color=color, thickness=-1)
    cv2.imwrite(output_path, cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))