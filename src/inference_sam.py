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
from ultralytics import SAM
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import random
from utils.utils import draw_overlay
import pytorch_lightning as pl
# from math import sqrt

pl.seed_everything(0)

def load_ultralytics_sam_model(device):
    """
    Load the SAM model using the Ultralytics implementation.

    Args:
        device (torch.device): Device to load the model on.

    Returns:
        SAM: Loaded SAM model.
    """
    return SAM("models/SAM/sam2.1_l.pt").to(device)

def load_hugging_face_sam_model(device):
    """
    Load the SAM 2 model using the original SAM 2 code.

    Args:
        device (torch.device): Device to load the model on.

    Returns:
        SAM2ImagePredictor: Loaded SAM 2 model.
    """
    checkpoint = "facebook/sam2.1-hiera-large"
    return SAM2ImagePredictor.from_pretrained(checkpoint)

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

def infer(image, model, device, use_tta=False, target_size=None):
    """
    Perform inference on an input image with optional Test-Time Augmentation (TTA).

    Args:
        image (np.ndarray): Input RGB image.
        model (SegmentationModel): Trained segmentation model.
        device (torch.device): Device to perform inference on.
        use_tta (bool): Whether to use TTA with horizontal flip and upscale.
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
        # Horizontal flip
        flipped_image = cv2.flip(resized_image, 1)
        flipped_tensor_image = ToTensorV2()(image=normalize_transform(image=flipped_image)["image"])["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            flipped_probs = torch.sigmoid(model(flipped_tensor_image)).squeeze(0).cpu().numpy()
        flipped_probs = np.flip(flipped_probs, axis=2)  # Flip back horizontally

        # Average all variants
        probs = np.mean([probs, flipped_probs], axis=0)

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

def prepare_sam_input(cropped_mask, kernel, config):
    """
    Prepare SAM input by randomly generating points and annotating them as positive or negative.

    Args:
        cropped_mask (np.ndarray): Cropped probability mask.
        kernel (np.ndarray): Kernel for morphological operations.
        config (dict): Configuration dictionary.
        debug_output_path (str): Path to save the debug visualization (optional).

    Returns:
        dict: SAM input prompts with points and labels.
    """
    # Find contours on the positive regions
    contours_positive, _ = cv2.findContours((cropped_mask > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    positive_mask = np.zeros_like(cropped_mask, dtype=np.uint8)
    cv2.drawContours(positive_mask, contours_positive, -1, 1, thickness=cv2.FILLED)

    # Classify areas outside the positive contours as negative
    negative_mask = (positive_mask == 0).astype(np.uint8)

    # Erode both positive and negative masks
    eroded_positive = cv2.erode(positive_mask, kernel, iterations=1)
    eroded_negative = cv2.erode(negative_mask, kernel, iterations=1)

    h, w = cropped_mask.shape
    k = config.get("num_points", 5)  # Total number of random points to generate
    points = [(random.randint(0, w - 1), random.randint(0, h - 1)) for _ in range(k)]

    positive_points = []
    negative_points = []

    for x, y in points:
        if eroded_positive[y, x] == 1:
            positive_points.append((x, y))
        elif eroded_negative[y, x] == 1:
            negative_points.append((x, y))

    prompts = {
        "input_point": [positive_points + negative_points],
        "input_label": [[1] * len(positive_points) + [0] * len(negative_points)],
    }

    return prompts

def clean_mask(probs_mask):
    """
    Clean the probability mask using morphological operations.

    Args:
        probs_mask (np.ndarray): Probability mask.

    Returns:
        np.ndarray: Cleaned mask.
    """
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(probs_mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
    return cleaned_mask

def find_and_filter_contours(cleaned_mask, config):
    """
    Find and filter contours based on area threshold.

    Args:
        cleaned_mask (np.ndarray): Cleaned binary mask.
        config (dict): Configuration dictionary.

    Returns:
        list: Filtered contours.
    """
    contours, _ = cv2.findContours((cleaned_mask > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = config.get("min_contour_area", 4000)
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

def infer_on_cropped_region(cropped_image, cropped_mask, sam_model, kernel, config):
    """
    Perform inference on a cropped region using SAM.

    Args:
        cropped_image (np.ndarray): Cropped RGB image.
        cropped_mask (np.ndarray): Cropped probability mask.
        sam_model (SAM): SAM model for segmentation.
        kernel (np.ndarray): Kernel for morphological operations.
        config (dict): Configuration dictionary.

    Returns:
        np.ndarray: Submask predicted by SAM.
    """
    prompts = prepare_sam_input(cropped_mask, kernel, config)
    sam_result = sam_model.predict(source=cropped_image,
                                   points=prompts['input_point'],
                                   labels=prompts['input_label'],
                                   verbose=False)
    submask = sam_result[0].masks.data.cpu().numpy()[0]
    return submask

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

def process_contours(image, probs_mask, sam_model: SAM2ImagePredictor, config):
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
    kernel = np.ones((31, 31), np.uint8)  # Kernel for morphological operations
    contours = find_and_filter_contours(probs_mask, config)

    refined_mask = np.zeros_like(probs_mask, dtype=np.float32)
    sam_consecutive_iterations = config.get("sam_consecutive_iterations", 20)
    num_sampling_runs = config.get("num_sample_runs", 10)  # Number of times to run the process

    for contour in contours:
        cropped_image, cropped_mask, (x_start, y_start, x_end, y_end) = crop_image_and_mask(image, probs_mask, contour, config)

        # Initialize PointGenerator
        point_generator = PointGenerator(config.get("num_points", 1000), 
                                         cropped_mask=cropped_mask, kernel=kernel, prob_thresh=0.5)

        # Iteratively refine the mask
        refined_masks = []  # Store masks for averaging

        for _ in range(num_sampling_runs):  # Run the process k times
            predicted_mask = None
            all_points = {"input_point": [], "input_label": []}

            prev_logits = None  # Initialize previous logits
            sam_model.set_image(cropped_image)  # Set the cropped image for SAM

            for i in range(sam_consecutive_iterations):
                if predicted_mask is None:
                    # Use PointGenerator to retrieve random points
                    new_points = point_generator.retrieve_random_points(num_points=5)
                else:
                    # Retrieve key points based on predicted_mask
                    new_points = point_generator.retrieve_key_points(predicted_mask, num_points=5)

                if not new_points["input_point"][0]:  # Stop if no new points
                    print(f"num iteration: {i}")
                    print(f"number of prompted points: {len(all_points['input_point'])}")
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

def post_process_refined_mask(refined_mask, area_threshold=1000, epsilon=0.01, prob_thresh=0.5):
    """
    Post-process the refined mask by filtering contours based on area and simplifying them.

    Args:
        refined_mask (np.ndarray): Refined binary mask.
        area_threshold (int): Minimum area for contours to be retained.
        epsilon (float): Approximation accuracy for contour simplification.

    Returns:
        np.ndarray: Post-processed binary mask.
    """
    # Smooth the mask using Gaussian blur
    # smoothed_mask = cv2.GaussianBlur(refined_mask, (5, 5), 0)

    # Convert mask to binary
    # binary_mask = (refined_mask > prob_thresh).astype(np.uint8)

    _, binary_mask = cv2.threshold((refined_mask * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank mask for the post-processed result
    post_processed_mask = np.zeros_like(binary_mask)

    for contour in contours:
        # Filter contours by area
        if cv2.contourArea(contour) >= area_threshold:
            # Simplify the contour
            approx_contour = cv2.approxPolyDP(contour, epsilon * cv2.arcLength(contour, True), True)
            # Draw the simplified contour on the mask
            cv2.drawContours(post_processed_mask, [approx_contour], -1, 1, thickness=cv2.FILLED)

    return post_processed_mask

if __name__ == "__main__":
    # Load configuration from YAML
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    model_path = 'models/base_model_b4/last.ckpt'
    model_config = config["model"]
    input_folder = "data/processed/v2/images/test/"
    output_folder = "data/outputs/best_b4_sam_l_cropped_box_interative_refinement_tta_less_aprrox/"
    use_tta = config.get("use_tta", False)
    k = config.get("num_points", 128)  # Number of points per dimension
    image_size = tuple(config.get("image_size", (512, 512)))
    patch_size = tuple(config.get("patch_size", (0.3, 0.3)))
    strides = tuple(config.get("strides", (0.05, 0.05)))
    refined_segmentation_mode = config.get("refined_segmentation_mode", "box")
    
    print(f"Config: {config}")

    os.makedirs(output_folder, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, model_config, device)
    if refined_segmentation_mode == "tiling":
        sam_model = load_ultralytics_sam_model(device)
    elif refined_segmentation_mode == "box":
        sam_model = load_hugging_face_sam_model(device)

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

        if refined_segmentation_mode == "tiling":
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
        elif refined_segmentation_mode == "box":
            # refined_masks = []
            # for _ in range(10):  # Run inference 5 times
            refined_mask = process_contours(image, probs_mask, sam_model, config)
            # refined_masks.append(refined_mask)
            
            # # Take the average of the 5 refined masks
            # refined_mask = np.mean(refined_masks, axis=0)

            # Post-process the refined mask
            postprocessing_mask = post_process_refined_mask(refined_mask, area_threshold=4000, epsilon=0.003, prob_thresh=0.8)

        # Draw overlay with refined mask
        refined_overlay = draw_overlay(image, postprocessing_mask)
        cv2.imwrite(output_overlay_path, cv2.cvtColor(refined_overlay, cv2.COLOR_RGB2BGR))

        # Save refined mask
        cv2.imwrite(output_mask_path, (refined_mask * 255).astype(np.uint8))

        # break  # Remove this line to process all images

