

import random
import cv2
import numpy as np


class PointGenerator:
    def __init__(self, k, cropped_mask, kernel, prob_thresh=0.5):
        """
        Initialize the PointGenerator with pre-generated points and eroded masks.

        Args:
            k (int): Number of points to pre-generate.
            cropped_mask (np.ndarray): Cropped probability mask.
            kernel (np.ndarray): Kernel for morphological operations.
        """
        self.k = k

        # Prepare positive and negative masks
        contours_positive, _ = cv2.findContours((cropped_mask > prob_thresh).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        positive_mask = np.zeros_like(cropped_mask, dtype=np.uint8)
        cv2.drawContours(positive_mask, contours_positive, -1, 1, thickness=cv2.FILLED)

        negative_mask = (positive_mask == 0).astype(np.uint8)

        self.positive_mask = cv2.erode(positive_mask, kernel, iterations=1)
        self.negative_mask = cv2.erode(negative_mask, kernel, iterations=1)

        # Pre-generate random points
        h, w = cropped_mask.shape
        self.points = [(random.randint(0, w - 1), random.randint(0, h - 1)) for _ in range(k)]
        self.used_points = set()  # Track used points

        self.prev_fp_mask = None  # Cache for previous false positive mask
        self.prev_fn_mask = None  # Cache for previous false negative mask
        self.iou_thresh = 0.85  # IoU threshold to stop point generation

    def retrieve_key_points(self, predicted_mask=None, num_points=10):
        """
        Retrieve key points based on false positive and false negative regions.

        Args:
            predicted_mask (np.ndarray): Predicted mask from the previous iteration.
            num_points (int): Number of points to retrieve for each region.

        Returns:
            dict: Points and labels for SAM input.
        """
        false_positive = (predicted_mask > 0.5) & (self.negative_mask > 0)
        false_negative = (self.positive_mask > 0.5) & (predicted_mask <= 0.5)

        # Calculate IoU with previous masks
        # If false positive and false negative masks does not change significantly, 
        # New retrived points can be noised, so we can skip this iteration
        if self.prev_fp_mask is not None and self.prev_fn_mask is not None:
            fp_iou = np.sum(false_positive & self.prev_fp_mask) / np.sum(false_positive | self.prev_fp_mask)
            fn_iou = np.sum(false_negative & self.prev_fn_mask) / np.sum(false_negative | self.prev_fn_mask)
            if fp_iou > self.iou_thresh and fn_iou > self.iou_thresh:
                # Cache the condition met count
                if not hasattr(self, "condition_met_count"):
                    self.condition_met_count = 0
                self.condition_met_count += 1

                if self.condition_met_count >= 2:
                    return {"input_point": [[]], "input_label": [[]]}
            else:
                self.condition_met_count = 0  # Reset if condition is not met

        # Update cached masks
        self.prev_fp_mask = false_positive
        self.prev_fn_mask = false_negative

        fp_points = [(x, y) for x, y in self.points if false_positive[y, x]]
        fn_points = [(x, y) for x, y in self.points if false_negative[y, x]]

        selected_fp = random.sample(fp_points, min(len(fp_points), num_points))
        selected_fn = random.sample(fn_points, min(len(fn_points), num_points))

        # Mark selected points as used and remove them from self.points
        self.used_points.update(selected_fp + selected_fn)
        self.points = [p for p in self.points if p not in self.used_points]

        # Assign false positive points as negative and false negative points as positive
        input_points = selected_fp + selected_fn
        input_labels = [0] * len(selected_fp) + [1] * len(selected_fn)

        return {"input_point": [input_points], "input_label": [input_labels]}

    def retrieve_random_points(self, num_points=10):
        """
        Retrieve random points and classify them as positive or negative based on the mask.

        Args:
            num_points (int): Number of random points to retrieve.

        Returns:
            dict: Points and labels for SAM input.
        """
        random_points = random.sample(self.points, min(len(self.points), num_points))
        positive_points = [(x, y) for x, y in random_points if self.positive_mask[y, x] == 1]
        negative_points = [(x, y) for x, y in random_points if self.negative_mask[y, x] == 1]

        # Mark selected points as used and remove them from self.points
        self.used_points.update(random_points)
        self.points = [p for p in self.points if p not in self.used_points]

        input_points = positive_points + negative_points
        input_labels = [1] * len(positive_points) + [0] * len(negative_points)

        return {"input_point": [input_points], "input_label": [input_labels]}
    
    
