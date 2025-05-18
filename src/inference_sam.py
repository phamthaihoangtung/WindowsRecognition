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
from utils.utils import draw_overlay
import pytorch_lightning as pl
from inference.point_generator import PointGenerator
from inference.post_processor import post_process_refined_mask
from inference.tiling_processor import process_image_with_tiling
from inference.contour_processor import process_contours


class WindowsRecognitor:
    def __init__(self, config_path):
        # Load configuration
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.model_path = 'models/base_model_b4/last.ckpt'
        
        self.use_tta = self.config.get("use_tta", False)
        self.image_size = tuple(self.config.get("image_size", (512, 512)))
        self.refined_segmentation_mode = self.config.get("refined_segmentation_mode", "contour")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load models
        self.model = self._load_model(self.model_path, self.config["model"], self.device)
        if self.refined_segmentation_mode == "tiling":
            self.sam_model = self._load_ultralytics_sam_model(self.device)
        elif self.refined_segmentation_mode == "contour":
            self.sam_model = self._load_hugging_face_sam_model(self.device)

    def _load_ultralytics_sam_model(self, device):
        return SAM("models/SAM/sam2.1_l.pt").to(device)

    def _load_hugging_face_sam_model(self, device):
        checkpoint = "facebook/sam2.1-hiera-large"
        return SAM2ImagePredictor.from_pretrained(checkpoint)

    def _load_model(self, model_path, model_config, device):
        model = SegmentationModel.load_from_checkpoint(
            model_path, map_location=device, model_config=model_config,
            learning_rate=0.001, scheduler_config=None  # Placeholder values
        )
        model.eval()
        model.to(device)
        return model

    def _infer(self, image):
        resized_image = cv2.resize(image, self.image_size)
        normalize_transform = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        tensor_image = ToTensorV2()(image=normalize_transform(image=resized_image)["image"])["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = torch.sigmoid(self.model(tensor_image)).squeeze(0).cpu().numpy()

        if self.use_tta:
            flipped_image = cv2.flip(resized_image, 1)
            flipped_tensor_image = ToTensorV2()(image=normalize_transform(image=flipped_image)["image"])["image"].unsqueeze(0).to(self.device)
            with torch.no_grad():
                flipped_probs = torch.sigmoid(self.model(flipped_tensor_image)).squeeze(0).cpu().numpy()
            flipped_probs = np.flip(flipped_probs, axis=2)
            probs = np.mean([probs, flipped_probs], axis=0)

        return probs

    def recognize(self, image):
        """
        Perform inference on a single image and return the postprocessed mask.

        Args:
            image (np.ndarray): Input RGB image.

        Returns:
            np.ndarray: Postprocessed mask.
        """
        # Perform inference
        self.probs_mask = self._infer(image).squeeze(0)
        self.probs_mask = cv2.resize(self.probs_mask, (image.shape[1], image.shape[0]))

        if self.refined_segmentation_mode == "tiling":
            refined_mask = process_image_with_tiling(image, self.probs_mask, self.sam_model, self.config, self.device)
        elif self.refined_segmentation_mode == "contour":
            refined_mask = process_contours(image, self.probs_mask, self.sam_model, self.config)

        postprocessing_mask = post_process_refined_mask(
            refined_mask,
            area_threshold_ratio=0.001,
            epsilon=0.003,
            convex_hull_iou_threshold=0.95,
        )
        return postprocessing_mask

    def recognize_batch(self, input_folder=None, output_folder=None):
        """
        Process a batch of images from the input folder and save results.
        """
        os.makedirs(output_folder, exist_ok=True)

        for image_path in tqdm(glob.glob(os.path.join(input_folder, "*.jpg")), desc="Processing images"):
            image_name = os.path.basename(image_path)
            output_probs_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_probs.png")
            output_overlay_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_overlay.png")
            output_mask_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_mask.png")

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Get postprocessed mask
            postprocessing_mask = self.recognize(image)

            # Save raw probability mask
            cv2.imwrite(output_probs_path, (self.probs_mask * 255).astype(np.uint8))

            # Draw overlay with refined mask
            refined_overlay = draw_overlay(image, postprocessing_mask)
            cv2.imwrite(output_overlay_path, cv2.cvtColor(refined_overlay, cv2.COLOR_RGB2BGR))

            # Save refined mask
            cv2.imwrite(output_mask_path, (postprocessing_mask * 255).astype(np.uint8))


if __name__ == "__main__":
    pl.seed_everything(0)

    input_folder = "data/processed/v2/images/test/"
    output_folder = "data/outputs/best_b4_sam_cropped_box_iterative_refinement_convexhull_polydp_"

    recognitor = WindowsRecognitor("config/config.yaml")
    recognitor.recognize_batch(input_folder, output_folder)
