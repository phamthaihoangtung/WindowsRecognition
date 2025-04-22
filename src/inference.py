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

def preprocess_image(image, normalize_transform, stride):
    """
    Preprocess the input image: resize, crop, and normalize.

    Args:
        image (np.ndarray): Input RGB image.
        normalize_transform (albumentations.Compose): Normalization transform.
        stride (int): Stride for cropping patches.

    Returns:
        list: List of cropped and normalized image patches.
    """
    original_size = image.shape[:2]
    resized_image = cv2.resize(image, (1024, 1024))
    patches = [
        resized_image[y:y+512, x:x+512]
        for y in range(0, 1024 - 512 + 1, stride)
        for x in range(0, 1024 - 512 + 1, stride)
    ]
    normalized_patches = [normalize_transform(image=patch)["image"] for patch in patches]
    return normalized_patches, original_size

def postprocess_mask(patches, original_size, stride):
    """
    Merge patches into a single mask and resize to the original image size.

    Args:
        patches (list): List of predicted mask patches.
        original_size (tuple): Original size of the input image (height, width).
        stride (int): Stride used during preprocessing.

    Returns:
        np.ndarray: Resized mask to the original image size.
    """
    merged_mask = np.zeros((1024, 1024), dtype=np.float32)
    weight_map = np.zeros((1024, 1024), dtype=np.float32)

    idx = 0
    for y in range(0, 1024 - 512 + 1, stride):
        for x in range(0, 1024 - 512 + 1, stride):
            merged_mask[y:y+512, x:x+512] += patches[idx]
            weight_map[y:y+512, x:x+512] += 1
            idx += 1

    merged_mask /= weight_map  # Average probabilities in overlap areas
    resized_mask = cv2.resize(merged_mask, (original_size[1], original_size[0]))
    return (resized_mask * 255).astype(np.uint8)  # Scale to 0-255 for saving

def fill_holes_preserving_boundary(binary_mask):
    """
    Fill holes in the binary mask while preserving the outer boundary.
    """
    kernel = np.ones((100, 100), np.uint8)
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
        overlay_path (str): Path to save the overlay image.

    Returns:
        np.ndarray: Updated binary mask.
    """
    # Smooth the mask using Gaussian Blur

    mask = fill_holes_preserving_boundary(mask)

    smooth_mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, binary_smooth_mask = cv2.threshold(smooth_mask, 95, 255, cv2.THRESH_BINARY)
    
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
        cv2.drawContours(overlay_image, [approx], -1, (0, 255, 0), thickness=2)  # Green boundary

    # kernel = np.ones((3, 3), np.uint8)
    # updated_mask = cv2.morphologyEx(updated_mask, cv2.MORPH_CLOSE, kernel)

    return overlay_image

def infer(image, model, device, stride):
    """
    Perform inference on an input image.

    Args:
        image (np.ndarray): Input RGB image.
        model (SegmentationModel): Trained segmentation model.
        device (torch.device): Device to perform inference on.
        stride (int): Stride for cropping patches.

    Returns:
        np.ndarray: Predicted binary mask.
    """
    # Preprocess the image
    normalize_transform = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    patches, original_size = preprocess_image(image, normalize_transform, stride)

    # Perform inference on each patch
    patches = torch.stack([ToTensorV2()(image=patch)["image"] for patch in patches]).to(device)
    with torch.no_grad():
        predictions = model(patches)
        predictions = torch.sigmoid(predictions).cpu().numpy()

    # Postprocess the mask
    mask_patches = [pred[0] for pred in predictions]  # Extract the first channel
    final_mask = postprocess_mask(mask_patches, original_size, stride)
    return final_mask

if __name__ == "__main__":
    # Load configuration from YAML
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    model_path = 'models/unet_model.ckpt'
    model_config = config["model"]
    input_folder = "data/processed/v2/images/test/"
    output_folder = "data/outputs/v2"
    stride = config.get("stride", 128)  # Default stride is 256

    # Create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(model_path, model_config, device)

    # Iterate over all image files in the input folder with progress bar
    for image_path in tqdm(glob.glob(os.path.join(input_folder, "*.jpg")), desc="Processing images"):
        # Prepare output paths
        image_name = os.path.basename(image_path)
        output_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}.png")
        overlay_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_overlay.png")

        # Perform inference
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = infer(image, model, device, stride)
        overlay_image = draw_polygons(mask, image)

        # Save the output mask
        cv2.imwrite(output_path, mask)  # Save the original binary mask
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))