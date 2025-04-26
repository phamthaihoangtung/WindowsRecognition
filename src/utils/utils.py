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
