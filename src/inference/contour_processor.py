import numpy as np
import cv2
import torch
from .point_generator import PointGenerator

def clean_mask(probs_mask):
    """
    Clean the probability mask using distance transform and watershed segmentation.

    Args:
        probs_mask (np.ndarray): Probability mask.

    Returns:
        np.ndarray: Cleaned mask.
    """
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(probs_mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)

    # Threshold the probability mask to binary
    _, binary_mask = cv2.threshold((cleaned_mask * 255).astype(np.uint8), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Fill holes inside contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(binary_mask)
    cv2.drawContours(filled_mask, contours, -1, 1, thickness=cv2.FILLED)

    # Noise removal
    opening = cv2.morphologyEx(filled_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 
                                #    0.5 * dist_transform.max()
                                np.percentile(dist_transform[dist_transform>0], 75)
                                , 1, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labeling
    _, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Mark the region of unknown with zero
    markers[unknown == 1] = 0

    # # DEBUG: Save markers for debugging
    # debug_markers_path = os.path.join('data/outputs/debug', f"{os.path.splitext(image_name)[0]}_markers_debug.png")
    # os.makedirs(os.path.dirname(debug_markers_path), exist_ok=True)
    # cv2.imwrite(debug_markers_path, (markers * 255 / markers.max()).astype(np.uint8))

    # Apply watershed segmentation
    color_image = cv2.cvtColor(filled_mask, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color_image, markers)

    # Black out the watershed lines while keeping original probability values
    filled_mask[markers == -1] = 0  # Set boundary regions to 0

    # Erode the binary mask by one pixel
    kernel = np.ones((3, 3), np.uint8)
    filled_mask = cv2.erode(filled_mask, kernel, iterations=1) 

    filled_mask_2 = cv2.cvtColor(filled_mask.copy()*255, cv2.COLOR_GRAY2BGR)
    filled_mask_2[markers == -1] = (255, 0, 0)  # Set boundary regions to 0

    # # DEBUG: Save cleaned mask for debugging
    # debug_cleaned_mask_path = os.path.join('data/outputs/debug', f"{os.path.splitext(image_name)[0]}_split_objects_debug.png")
    # os.makedirs(os.path.dirname(debug_cleaned_mask_path), exist_ok=True)
    # cv2.imwrite(debug_cleaned_mask_path, (filled_mask_2).astype(np.uint8))

    return filled_mask

def find_and_filter_contours(cleaned_mask, config):
    """
    Find and filter contours based on area threshold.

    Args:
        cleaned_mask (np.ndarray): Cleaned binary mask.
        config (dict): Configuration dictionary.

    Returns:
        list: Filtered contours.
    """
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate area threshold as a ratio of the image size
    image_area = cleaned_mask.shape[0] * cleaned_mask.shape[1]
    area_ratio = config.get("min_contour_area_ratio", 0.001)  # Default to 1% of the image area
    min_contour_area = image_area * area_ratio

    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= min_contour_area]
    return filtered_contours

def crop_image_and_mask(image, probs_mask, contour, config):
    """
    Crop the image and mask based on the bounding rectangle of a contour.

    Args:
        image (np.ndarray): Original RGB image.
        probs_mask (np.ndarray): Probability mask.
        contour (np.ndarray): Contour to crop around.
        config (dict): Configuration dictionary.

    Returns:
        tuple: Cropped image and cropped mask.
    """
    x, y, w, h = cv2.boundingRect(contour)
    extend_ratio = config.get("extend_ratio", 0.2)
    x_start = max(0, int(x - w * extend_ratio))
    y_start = max(0, int(y - h * extend_ratio))
    x_end = min(image.shape[1], int(x + w * (1 + extend_ratio)))
    y_end = min(image.shape[0], int(y + h * (1 + extend_ratio)))

    cropped_image = image[y_start:y_end, x_start:x_end]
    cropped_mask = probs_mask[y_start:y_end, x_start:x_end]

    # Create a blank mask for the cropped region
    contour_mask = np.zeros_like(cropped_mask, dtype=np.uint8)

    # Adjust the contour coordinates to the cropped region
    adjusted_contour = contour - [x_start, y_start]

    # Draw the contour on the blank mask
    cv2.drawContours(contour_mask, [adjusted_contour], -1, 1, thickness=cv2.FILLED)

    # Mask the cropped region to isolate the specific contour
    cropped_mask = cropped_mask * contour_mask

    return cropped_image, cropped_mask, (x_start, y_start, x_end, y_end)



def process_contours(image, probs_mask, sam_model, config):
    """
    Process contours in the mask, refine regions using SAM, and create a refined mask.

    Args:
        image (np.ndarray): Original RGB image.
        probs_mask (np.ndarray): Probability mask.
        sam_model (SAM2ImagePredictor): SAM model for segmentation.
        config (dict): Configuration dictionary.

    Returns:
        np.ndarray: Refined binary mask.
    """

    probs_mask = clean_mask(probs_mask)  # Clean the mask before processing

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    kernel = np.ones((15, 15), np.uint8)  # Kernel for morphological operations
    contours = find_and_filter_contours(probs_mask, config)

    # # DEBUG: Draw all contours and save the result
    # debug_contour_image = image.copy()
    # cv2.drawContours(debug_contour_image, contours, -1, (0, 255, 0), 2)  # Green contours
    # debug_contour_path = os.path.join('data/outputs/debug_contours', f"{os.path.splitext(image_name)[0]}_contours_debug.png")
    # print(f"Debugging contours saved at: {debug_contour_path}")
    # cv2.imwrite(debug_contour_path, cv2.cvtColor(debug_contour_image, cv2.COLOR_RGB2BGR))

    refined_mask = np.zeros_like(probs_mask, dtype=np.float32)
    sam_consecutive_iterations = config.get("sam_consecutive_iterations", 20)
    num_sampling_runs = config.get("num_sample_runs", 10)  # Number of times to run the process

    for contour in contours:
        cropped_image, cropped_mask, (x_start, y_start, x_end, y_end) = crop_image_and_mask(image, probs_mask, contour, config)

        sam_model.set_image(cropped_image)  # Set the cropped image for SAM
        # Initialize PointGenerator
        point_generator = PointGenerator(config.get("num_points", 1000), 
                                         cropped_mask=cropped_mask, kernel=kernel, prob_thresh=0.5)

        # Iteratively refine the mask
        refined_masks = []  # Store masks for averaging

        for sampling_index in range(num_sampling_runs):  # Run the process k times
            predicted_mask = None
            all_points = {"input_point": [], "input_label": []}

            prev_logits = None  # Initialize previous logits

            for sam_iter_index in range(sam_consecutive_iterations):
                if predicted_mask is None:
                    # Use PointGenerator to retrieve random points
                    new_points = point_generator.retrieve_random_points(num_points=5)
                else:
                    # Retrieve key points based on predicted_mask
                    new_points = point_generator.retrieve_key_points(predicted_mask, num_points=5)

                if not new_points["input_point"][0]:  # Stop if no new points
                    print(f"num iteration: {sam_iter_index}")
                    print(f"number of prompted points: {len(all_points['input_point'])}")

                    # # DEBUG: Save the cropped image with points for debugging
                    # # Draw points on the cropped image and save the result
                    # debug_image = cropped_image.copy()
                    # for (x, y), label in zip(all_points["input_point"], all_points["input_label"]):
                    #     color = (0, 255, 0) if label == 1 else (255, 0, 0)  # Green for positive, Red for negative
                    #     cv2.circle(debug_image, (x, y), radius=5, color=color, thickness=-1)
                    # debug_output_path = os.path.join('data/outputs/debug_sam', 
                    #                                  f"{os.path.splitext(image_name)[0]}_x_start_{x_start}_y_start_{y_start}_{sampling_index}.png")
                    # cv2.imwrite(debug_output_path, cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))

                    break

                # Concatenate new points with existing points
                all_points["input_point"] += new_points['input_point'][0]
                all_points["input_label"] += new_points['input_label'][0]

                # Predict using SAM with logits as input
                with torch.inference_mode(), torch.autocast(device_type=sam_model.device.type, dtype=torch.bfloat16):
                    predicted_mask, _, prev_logits = sam_model.predict(
                        point_coords=np.array(all_points["input_point"]),
                        point_labels=np.array(all_points["input_label"]),
                        mask_input=prev_logits,  # Use logits from the previous iteration
                        multimask_output=False,
                    )

                    predicted_mask = predicted_mask[0]

            if predicted_mask is not None:
                refined_masks.append(predicted_mask)  # Collect the mask for averaging

        # Average the masks
        if refined_masks:
            predicted_mask = np.mean(refined_masks, axis=0)

        if not predicted_mask is None:
            # Merge the final predicted mask into the global refined mask
            refined_mask[y_start:y_end, x_start:x_end] = np.maximum(refined_mask[y_start:y_end, x_start:x_end], predicted_mask)

    return refined_mask
