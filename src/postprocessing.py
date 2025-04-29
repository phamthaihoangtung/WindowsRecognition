import cv2
import numpy as np
import os
from utils.utils import draw_overlay

def apply_otsu_threshold(image: np.ndarray) -> np.ndarray:
    """
    Applies Gaussian blur followed by Otsu's thresholding to the input image.

    Args:
        image (np.ndarray): Grayscale image to threshold.

    Returns:
        np.ndarray: Binary image after applying Otsu's thresholding.
    """
    if len(image.shape) != 2:
        raise ValueError("Input image must be a grayscale image.")
    
    # # Apply Gaussian blur
    # blurred_image = cv2.GaussianBlur(image, (21, 21), 0)
    
    # Apply Otsu's thresholding
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image

def main():
    # Define directories
    mask_dir = "data/outputs/base_image_sam_l_strict_point_2/"
    image_dir = "data/processed/v3/images/test/"
    output_dir = "data/outputs/overlay_results/"
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all mask files in the directory
    for mask_filename in os.listdir(mask_dir):
        if not mask_filename.endswith(".png") or "_mask" not in mask_filename:
            continue

        mask_path = os.path.join(mask_dir, mask_filename)
        image_filename = mask_filename.replace("_mask.png", ".jpg")
        image_path = os.path.join(image_dir, image_filename)

        # Read the mask and input image
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Mask not found or invalid at {mask_path}")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Image not found or invalid at {image_path}")
            continue

        # Apply Otsu's thresholding on the mask
        binary_mask = apply_otsu_threshold(mask)

        # Save the binary mask with '_mask' postfix
        binary_mask_path = os.path.join(output_dir, f"{os.path.splitext(mask_filename)[0]}_mask.png")
        cv2.imwrite(binary_mask_path, binary_mask)

        # Draw the overlay
        overlay_image = draw_overlay(image, binary_mask)

        # Save the overlay with '_overlay' postfix
        overlay_path = os.path.join(output_dir, f"{os.path.splitext(image_filename)[0]}_overlay.jpg")
        cv2.imwrite(overlay_path, overlay_image)
        print(f"Processed and saved overlay: {overlay_path}")
        print(f"Saved binary mask: {binary_mask_path}")

if __name__ == "__main__":
    main()

