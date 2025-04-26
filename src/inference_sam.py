import os
import cv2
import torch
import yaml
import numpy as np
from albumentations import Normalize
from albumentations.pytorch import ToTensorV2
from model import SegmentationModel
from tqdm import tqdm
import glob
from ultralytics import SAM  # Import SAM model
import random
from utils.utils import draw_overlay

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
        model_path, map_location=device, model_config=model_config,
        learning_rate=0.001, scheduler_config=None  # Placeholder values
    )
    model.eval()
    model.to(device)
    return model

def load_sam_model(device):
    """
    Load the SAM 2.1 Tiny model.

    Args:
        device (torch.device): Device to load the model on.

    Returns:
        SAM: Loaded SAM model.
    """
    sam_model = SAM("models/SAM/sam2.1_l.pt").to(device)
    return sam_model

def infer(image, model, device, use_tta=False, target_size=None):
    """
    Perform inference on an input image with optional Test-Time Augmentation (TTA).

    Args:
        image (np.ndarray): Input RGB image.
        model (SegmentationModel): Trained segmentation model.
        device (torch.device): Device to perform inference on.
        use_tta (bool): Whether to use TTA with horizontal flip.
        target_size (tuple): Target size to resize the image (width, height).

    Returns:
        np.ndarray: Predicted probability mask.
    """
    # Resize the image to the target size
    resized_image = cv2.resize(image, target_size)

    normalize_transform = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    tensor_image = ToTensorV2()(image=normalize_transform(image=resized_image)["image"])["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(tensor_image)).squeeze(0).cpu().numpy()

    if use_tta:
        flipped_image = cv2.flip(resized_image, 1)
        flipped_tensor_image = ToTensorV2()(image=normalize_transform(image=flipped_image)["image"])["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            flipped_probs = torch.sigmoid(model(flipped_tensor_image)).squeeze(0).cpu().numpy()
        flipped_probs = np.flip(flipped_probs, axis=2)  # Flip back horizontally
        probs = (probs + flipped_probs) / 2  # Average the results

    return probs 

def extract_points(mask, k, positive_threshold=0.75, negative_threshold=0.25):
    """
    Extract k*k points from the mask and classify them as positive, negative, or neutral.

    Args:
        mask (np.ndarray): Probability mask (values between 0 and 1).
        k (int): Number of points per dimension.
        positive_threshold (float): Threshold above which a point is classified as positive.
        negative_threshold (float): Threshold below which a point is classified as negative.

    Returns:
        list: List of points with their classification (positive/negative/neutral).
    """
    # Ensure the mask has two dimensions
    if len(mask.shape) > 2:
        mask = mask.squeeze()  # Remove singleton dimensions if present

    h, w = mask.shape
    step_x, step_y = w // k, h // k
    remainder_x, remainder_y = w % k, h % k  # Calculate remainders
    points = []
    for i in range(k):
        for j in range(k):
            x_start = j * step_x + min(j, remainder_x)
            x_end = x_start + step_x + (1 if j < remainder_x else 0)
            y_start = i * step_y + min(i, remainder_y)
            y_end = y_start + step_y + (1 if i < remainder_y else 0)
            local_window = mask[y_start:y_end, x_start:x_end]
            prob_sum = np.sum(local_window) / (local_window.size or 1)
            if prob_sum > positive_threshold:
                classification = "positive"
            elif prob_sum < negative_threshold:
                classification = "negative"
            else:
                classification = "neutral"
            x = (x_start + x_end) // 2
            y = (y_start + y_end) // 2
            points.append((x, y, classification))
    return points


def process_patches(image, probs_mask, points, sam_model, patch_size, strides, device):
    """
    Process the image in patches, filter points, and use SAM for region refinement.

    Args:
        image (np.ndarray): Original RGB image.
        probs_mask (np.ndarray): Probability mask.
        points (list): List of points with classification.
        sam_model (SAM): SAM model for segmentation.
        patch_size (tuple): Patch size (width, height) as pixels or ratios.
        strides (tuple): Strides for patch sliding (width, height) as pixels or ratios.
        device (torch.device): Device to perform processing on.

    Returns:
        np.ndarray: Refined binary mask.
    """
    h, w = image.shape[:2]

    # Handle patch_size as float (ratio) or int (pixels)
    patch_w = int(patch_size[0] * w) if isinstance(patch_size[0], float) else patch_size[0]
    patch_h = int(patch_size[1] * h) if isinstance(patch_size[1], float) else patch_size[1]

    # Handle strides as float (ratio) or int (pixels)
    stride_w = int(strides[0] * w) if isinstance(strides[0], float) else strides[0]
    stride_h = int(strides[1] * h) if isinstance(strides[1], float) else strides[1]

    refined_mask = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)  # Initialize weight as zeros

    for y in range(0, h, stride_h):
        for x in range(0, w, stride_w):
            patch_x_end = min(x + patch_w, w)
            patch_y_end = min(y + patch_h, h)
            patch = image[y:patch_y_end, x:patch_x_end]
            patch_probs = probs_mask[y:patch_y_end, x:patch_x_end]

            # Filter points inside the patch
            patch_points = [
                (px, py, cls) for px, py, cls in points
                if x <= px < patch_x_end and y <= py < patch_y_end
            ]

            # Separate positive and negative points
            positive_points = [(px - x, py - y) for px, py, cls in patch_points if cls == "positive"]
            negative_points = [(px - x, py - y) for px, py, cls in patch_points if cls == "negative"]

            # Count positive and negative points
            num_positive = len(positive_points)
            num_negative = len(negative_points)
            total_points = len(patch_points)

            # Check if positive and negative points exceed 25% of total points
            if total_points > 0 and num_positive / total_points > 0.1 and num_negative / total_points > 0.1:
                # Randomly sample up to 10 points for each category
                positive_points = random.sample(positive_points, min(len(positive_points), 20))
                negative_points = random.sample(negative_points, min(len(negative_points), 10))

                # Use SAM with points as prompts
                prompts = {
                    "input_point": [positive_points + negative_points],
                    "input_label": [[1] * len(positive_points) + [0] * len(negative_points)],
                }
                sam_result = sam_model.predict(source=patch, 
                                               points=prompts['input_point'],
                                               labels=prompts['input_label'],
                                               verbose=False)  # Extract the first result from the generator
                sam_mask = sam_result[0].masks.data.cpu().numpy()[0]
                refined_patch = sam_mask
            else:
                # Skip this patch
                continue

            # Merge the refined patch into the global mask
            refined_mask[y:patch_y_end, x:patch_x_end] += refined_patch
            weight[y:patch_y_end, x:patch_x_end] += 1

    # Ensure minimum weight is 1
    weight = np.maximum(weight, 1)

    # Take the average probability for overlapping zones
    refined_mask = refined_mask / weight
    return refined_mask

if __name__ == "__main__":
    # Load configuration from YAML
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    model_path = 'models/base_model.ckpt'
    model_config = config["model"]
    input_folder = "data/processed/v2/images/test/"
    output_folder = "data/outputs/base_image_sam_l_strict_point_2/"
    use_tta = config.get("use_tta", False)
    k = config.get("num_points", 128)  # Number of points per dimension
    image_size = tuple(config.get("image_size", (512, 512)))
    patch_size = tuple(config.get("patch_size", (0.3, 0.3)))
    strides = tuple(config.get("strides", (0.05, 0.05)))

    os.makedirs(output_folder, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, model_config, device)
    sam_model = load_sam_model(device)

    for image_path in tqdm(glob.glob(os.path.join(input_folder, "*.jpg")), desc="Processing images"):
        image_name = os.path.basename(image_path)
        output_prompt_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_prompt.png")
        output_overlay_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_overlay.png")
        output_mask_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_mask.png")
        output_probs_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_probs.png")  # Save raw probs mask

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Perform inference
        probs_mask = infer(image, model, device, use_tta=use_tta, target_size=image_size).squeeze(0)
        probs_mask = cv2.resize(probs_mask, (image.shape[1], image.shape[0]))  # Resize to original image size
        
        # Save raw probability mask
        cv2.imwrite(output_probs_path, (probs_mask * 255).astype(np.uint8))

        # Resize image and probability mask to 1024x1024
        resized_image = cv2.resize(image, (2048, 2048))
        resized_probs_mask = cv2.resize(probs_mask, (2048, 2048))

        # Extract points
        points = extract_points(resized_probs_mask, k)

        # Draw overlay with points
        prompt_overlay = draw_overlay(resized_image, resized_probs_mask > 0.5, points)
        prompt_overlay = cv2.resize(prompt_overlay, (image.shape[1], image.shape[0]))  # Resize back to original size
        cv2.imwrite(output_prompt_path, cv2.cvtColor(prompt_overlay, cv2.COLOR_RGB2BGR))

        # Process patches and refine mask
        refined_mask = process_patches(resized_image, resized_probs_mask, points, sam_model, patch_size, strides, device)
        refined_mask = cv2.resize(refined_mask, (image.shape[1], image.shape[0]))  # Resize back to original size

        # Draw overlay with refined mask
        refined_overlay = draw_overlay(image, refined_mask > 0.5)
        cv2.imwrite(output_overlay_path, cv2.cvtColor(refined_overlay, cv2.COLOR_RGB2BGR))

        # Save refined mask
        cv2.imwrite(output_mask_path, (refined_mask * 255).astype(np.uint8))

        # break  # Remove this line to process all images
