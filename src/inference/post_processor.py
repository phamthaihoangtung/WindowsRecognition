import cv2
import numpy as np


def post_process_refined_mask(refined_mask, area_threshold_ratio=0.001, epsilon=0.01, convex_hull_iou_threshold=0.95, prob_thresh=0.5, use_otsu=False, dilation_kernel_size=0):
    """
    Post-process the refined mask by filtering contours based on area and simplifying them.

    Args:
        refined_mask (np.ndarray): Refined binary mask (float32, values in [0, 1]).
        area_threshold_ratio (float): Minimum area ratio for contours to be retained.
        epsilon (float): Approximation accuracy for Douglas-Peucker contour simplification,
            as a fraction of the contour arc length. Higher → fewer vertices, coarser polygon;
            lower → more vertices, closer to original boundary.
        prob_thresh (float): Fixed threshold for binarising the refined mask. Default 0.5.
            Ignored when use_otsu=True.
        use_otsu (bool): If True, use Otsu's method instead of prob_thresh. Suitable for
            bimodal probability masks.
        dilation_kernel_size (int): If > 0, apply per-contour morphological dilation with an
            ellipse kernel of this size before Douglas-Peucker simplification only. Compensates
            for DPP's inward-cutting bias. Convex hull path is unaffected.
    Returns:
        np.ndarray: Post-processed binary mask.
    """

    image_area = refined_mask.shape[0] * refined_mask.shape[1]
    if use_otsu:
        gray_mask = (refined_mask * 255).astype(np.uint8)
        gray_mask = cv2.GaussianBlur(gray_mask, (7, 7), 0)
        _, binary_mask = cv2.threshold(gray_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        binary_mask = (refined_mask > prob_thresh).astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank mask for the post-processed result
    post_processed_mask = np.zeros_like(binary_mask)

    dilation_kernel = None
    if dilation_kernel_size > 0:
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size))

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
                # Optionally dilate this contour before DPP to compensate for inward-cutting bias
                if dilation_kernel is not None:
                    local_mask = np.zeros_like(binary_mask)
                    cv2.drawContours(local_mask, [contour], -1, 255, thickness=cv2.FILLED)
                    local_mask = cv2.dilate(local_mask, dilation_kernel)
                    dilated_contours, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contour_for_dpd = dilated_contours[0] if dilated_contours else contour
                else:
                    contour_for_dpd = contour
                approx_contour = cv2.approxPolyDP(contour_for_dpd, epsilon * cv2.arcLength(contour_for_dpd, True), True)
                cv2.drawContours(post_processed_mask, [approx_contour], -1, 1, thickness=cv2.FILLED)

    return post_processed_mask
