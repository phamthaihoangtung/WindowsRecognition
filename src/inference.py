import os
import cv2
import torch
import yaml
import numpy as np
from albumentations import Normalize
from albumentations.pytorch import ToTensorV2
from model import SegmentationModel
import glob
from tqdm import tqdm  # Add tqdm for progress bar

def load_model(model_path, model_config, device):
    """
    Load the trained segmentation model.

    Args:
        model_path (str): Path to the saved model checkpoint.
        model_config (dict): Model configuration dictionary.
        device (torch.device): Device to load the model on.

    Returns:
        SegmentationModel: Loaded model.
    """
    model = SegmentationModel.load_from_checkpoint(
                                                    model_path, map_location=device,
                                                    model_config=model_config,
                                                    learning_rate=0.001,  # Placeholder, not used during inference
                                                    scheduler_config=None  # Placeholder, not used during inference
                                                   )
    model.eval()
    model.to(device)
    return model

def preprocess_image(image, normalize_transform, patch_size, stride, image_size):
    """
    Preprocess the input image: resize, crop, and normalize.

    Args:
        image (np.ndarray): Input RGB image.
        normalize_transform (albumentations.Compose): Normalization transform.
        patch_size (int): Size of the square patches.
        stride (int): Stride for cropping patches.
        image_size (tuple): Target size to resize each patch (width, height).

    Returns:
        list: List of cropped and normalized image patches.
    """
    original_size = image.shape[:2]
    resized_image = cv2.resize(image, (1024, 1024))  # Resize to 1024x1024 before patching
    patches = []

    for y in range(0, 1024 - patch_size + 1, stride):
        for x in range(0, 1024 - patch_size + 1, stride):
            patches.append(resized_image[y:y+patch_size, x:x+patch_size])

    # Handle the bottommost row of patches
    if (1024 - patch_size) % stride != 0:
        for x in range(0, 1024 - patch_size + 1, stride):
            patches.append(resized_image[1024-patch_size:1024, x:x+patch_size])

    resized_patches = [cv2.resize(patch, image_size) for patch in patches]
    normalized_patches = [normalize_transform(image=patch)["image"] for patch in resized_patches]
    return normalized_patches, original_size

def postprocess_mask(patches, original_size, patch_size, stride):
    """
    Merge patches into a single mask and resize to the original image size.

    Args:
        patches (list): List of predicted mask patches.
        original_size (tuple): Original size of the input image (height, width).
        patch_size (int): Size of the square patches.
        stride (int): Stride used during preprocessing.

    Returns:
        np.ndarray: Resized mask to the original image size.
    """
    merged_mask = np.zeros((1024, 1024), dtype=np.float32)
    weight_map = np.zeros((1024, 1024), dtype=np.float32)

    idx = 0
    for y in range(0, 1024 - patch_size + 1, stride):
        for x in range(0, 1024 - patch_size + 1, stride):
            patch = patches[idx]
            if patch.shape != (patch_size, patch_size):  # Ensure patch size matches
                patch = cv2.resize(patch, (patch_size, patch_size))
            merged_mask[y:y+patch_size, x:x+patch_size] += patch
            weight_map[y:y+patch_size, x:x+patch_size] += 1
            idx += 1

    merged_mask /= np.maximum(weight_map, 1)  # Avoid division by zero
    resized_mask = cv2.resize(merged_mask, (original_size[1], original_size[0]))

    return (resized_mask * 255).astype(np.uint8)  # Scale to 0-255 for saving

def fill_holes_preserving_boundary(binary_mask):
    """
    Fill holes in the binary mask while preserving the outer boundary.
    """
    kernel = np.ones((101, 101), np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary_mask = np.zeros_like(binary_mask)
    cv2.drawContours(boundary_mask, contours, -1, 255, thickness=cv2.FILLED)
    dilated = cv2.dilate(binary_mask, kernel, iterations=1)
    corrected = cv2.erode(dilated, kernel, iterations=1)
    return cv2.bitwise_and(corrected, boundary_mask)

def draw_polygons(mask, original_image):
    """
    Draw polygons on the mask and overlay boundaries on the original image.

    Args:
        mask (np.ndarray): Binary mask.
        original_image (np.ndarray): Original RGB image.

    Returns:
        np.ndarray: Updated binary mask.
    """
    _, mask = cv2.threshold(mask, 159, 255, cv2.THRESH_BINARY)

    # Save the mask with only boundary contours
    overlay_original = original_image.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.drawContours(overlay_original, [contour], -1, (255, 0, 0), thickness=3)  # Blue boundary

    binary_smooth_mask = fill_holes_preserving_boundary(mask)

    # Find contours
    contours, _ = cv2.findContours(binary_smooth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    updated_mask = np.zeros_like(mask)
    overlay_image = original_image.copy()

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if circularity > 0.7:  
            epsilon = 0.005 * perimeter  
        else: 
            epsilon = 0.03 * perimeter   

        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(updated_mask, [approx], -1, (255), thickness=cv2.FILLED)
        cv2.drawContours(overlay_image, [approx], -1, (0, 255, 0), thickness=3)  # Green boundary
        cv2.drawContours(overlay_original, [approx], -1, (0, 0, 255), thickness=3)  # Red boundary for approx

    return overlay_image, updated_mask, overlay_original

def infer(image, model, device, patch_size_ratio, stride_ratio, image_size):
    """
    Perform inference on an input image.

    Args:
        image (np.ndarray): Input RGB image.
        model (SegmentationModel): Trained segmentation model.
        device (torch.device): Device to perform inference on.
        patch_size_ratio (float): Ratio of patch size to the image edge.
        stride_ratio (float): Ratio of stride to the image size.
        image_size (tuple): Target size to resize each patch (width, height).

    Returns:
        np.ndarray: Predicted binary mask.
    """
    # Calculate patch size and stride based on ratios
    patch_size = int(1024 * patch_size_ratio)
    stride = int(1024 * stride_ratio)  # Update stride to be a ratio of the image size

    # Preprocess the image
    normalize_transform = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    patches, original_size = preprocess_image(image, normalize_transform, patch_size, stride, image_size)

    # Perform inference on each patch
    patches = torch.stack([ToTensorV2()(image=patch)["image"] for patch in patches]).to(device)
    with torch.no_grad():
        predictions = model(patches)
        predictions = torch.sigmoid(predictions).cpu().numpy()

    # Postprocess the mask
    mask_patches = [pred[0] for pred in predictions]  # Extract the first channel
    final_mask = postprocess_mask(mask_patches, original_size, patch_size, stride)
    return final_mask

def infer_with_tta(image, model, device, patch_size_ratio, stride_ratio, image_size):
    """
    Perform inference with Test-Time Augmentation (TTA) using horizontal flip.

    Args:
        image (np.ndarray): Input RGB image.
        model (SegmentationModel): Trained segmentation model.
        device (torch.device): Device to perform inference on.
        patch_size_ratio (float): Ratio of patch size to the image edge.
        stride_ratio (float): Ratio of stride to the image size.
        image_size (tuple): Target size to resize each patch (width, height).

    Returns:
        np.ndarray: Predicted binary mask with TTA.
    """
    # Perform inference on the original image
    mask_original = infer(image, model, device, patch_size_ratio, stride_ratio, image_size)

    # Perform inference on the horizontally flipped image
    flipped_image = cv2.flip(image, 1)  # Horizontal flip
    mask_flipped = infer(flipped_image, model, device, patch_size_ratio, stride_ratio, image_size)

    # Flip the mask back to the original orientation
    mask_flipped = cv2.flip(mask_flipped, 1)

    # Combine the masks (average)
    final_mask = 0.5 * mask_original + 0.5 * mask_flipped
    return final_mask.astype(np.uint8)

if __name__ == "__main__":
    # Load configuration from YAML
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    model_path = 'models/base_model.ckpt'
    model_config = config["model"]
    input_folder = "data/processed/v2/images/test/"
    output_folder = "data/outputs/base_image/"
    use_tta = config.get("use_tta", True)  # Default to True if not specified
    combine_patches = config.get("combine_patches", False)  # Default to True if not specified
    image_size = tuple(config.get("image_size", (512, 512)))  # Default image size is (512, 512)

    # Create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(model_path, model_config, device)

    # Iterate over all image files in the input folder with progress bar
    for i, image_path in tqdm(enumerate(glob.glob(os.path.join(input_folder, "*.jpg"))), desc="Processing images"):
        # Prepare output paths
        image_name = os.path.basename(image_path)
        output_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}.png")
        overlay_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_overlay.png")
        
        # Read and preprocess the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Perform inference with or without TTA
        infer_function = infer_with_tta if use_tta else infer

        if combine_patches:
            mask_1 = infer_function(image, model, device, patch_size_ratio=0.5, stride_ratio=0.125, image_size=image_size)
            mask_2 = infer_function(image, model, device, patch_size_ratio=0.625, stride_ratio=0.125, image_size=image_size)
            mask_3 = infer_function(image, model, device, patch_size_ratio=0.75, stride_ratio=0.125, image_size=image_size)
            mask_4 = infer_function(image, model, device, patch_size_ratio=1.0, stride_ratio=1.0, image_size=image_size)

            mask = (
                    0.3 * mask_1 
                    + 0.3 * mask_2 
                    + 0.2 * mask_3 
                    + 0.2 * mask_4
                    )  # Combine mask with weights
        else:
            mask = infer_function(image, model, device, patch_size_ratio=1.0, stride_ratio=1.0, image_size=image_size)

        mask = mask.astype(np.uint8)

        overlay_image, _, overlay_original = draw_polygons(mask, image)

        # Save the output mask
        cv2.imwrite(output_path, mask)  # Save the original binary mask
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_original.png"), cv2.cvtColor(overlay_original, cv2.COLOR_RGB2BGR))