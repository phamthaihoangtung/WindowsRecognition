import cv2
import numpy as np


def post_process_refined_mask(refined_mask, area_threshold_ratio=0.001, epsilon=0.01, convex_hull_iou_threshold=0.95):
    """
    Post-process the refined mask by filtering contours based on area and simplifying them.

    Args:
        refined_mask (np.ndarray): Refined binary mask.
        area_threshold_ratio (float): Minimum area ratio for contours to be retained.o be retained.
        epsilon (float): Approximation accuracy for contour simplification.
        prob_thresh (float): Probability threshold for binary conversion.
    Returns:
        np.ndarray: Post-processed binary mask.
    """ 

    # Smooth the mask using Gaussian blur
    # smoothed_mask = cv2.GaussianBlur(refined_mask, (5, 5), 0)
    image_area = refined_mask.shape[0] * refined_mask.shape[1]
    # Convert mask to binaryrea * area_threshold_ratio
    # binary_mask = (refined_mask > prob_thresh).astype(np.uint8)
    # Convert mask to binary

    gray_mask = (refined_mask * 255).astype(np.uint8)
    gray_mask = cv2.GaussianBlur(gray_mask,(7,7),0)
    _, binary_mask = cv2.threshold(gray_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank mask for the post-processed result
    post_processed_mask = np.zeros_like(binary_mask)

    for contour in contours:
        # Filter contours by area
        if cv2.contourArea(contour) >= area_threshold_ratio * image_area:
            # Find convex hull
            convex_hull = cv2.convexHull(contour)

            # Calculate IoU between the contour and its convex hull
            contour_mask = np.zeros_like(binary_mask)
            hull_mask = np.zeros_like(binary_mask)
            cv2.drawContours(contour_mask, [contour], -1, 1, thickness=cv2.FILLED)
            cv2.drawContours(hull_mask, [convex_hull], -1, 1, thickness=cv2.FILLED)
            intersection = np.sum((contour_mask & hull_mask) > 0)
            union = np.sum((contour_mask | hull_mask) > 0)
            iou = intersection / union if union > 0 else 0

            # Replace contour with convex hull if IoU exceeds threshold
            if iou > convex_hull_iou_threshold:
                cv2.drawContours(post_processed_mask, [convex_hull], -1, 1, thickness=cv2.FILLED)
            else:
                # Simplify the contour
                approx_contour = cv2.approxPolyDP(contour, epsilon * cv2.arcLength(contour, True), True)
                cv2.drawContours(post_processed_mask, [approx_contour], -1, 1, thickness=cv2.FILLED)

    return post_processed_mask