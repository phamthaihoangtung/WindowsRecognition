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
from PIL import Image as PILImage
from ultralytics import SAM
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from utils.utils import draw_overlay
import pytorch_lightning as pl
from inference.post_processor import post_process_refined_mask
from inference.tiling_processor import process_image_with_tiling
from inference.contour_processor import process_contours
from inference.cascade_processor import process_with_cascadepsp
from inference.crm_processor import process_with_crm, load_crm_model
from pathlib import Path


class WindowsRecognitor:
    def __init__(self, config_path):
        # Load configuration
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.model_path = self.config["inference"].get("model_path", None)
        self.sam_checkpoint = self.config["inference"].get("sam_checkpoint", "facebook/sam2.1-hiera-large")

        self.image_size = tuple(self.config["hyperparameters"].get("image_size", (512, 512)))
        self.coarse_segmentation_mode = self.config["inference"].get("coarse_segmentation_mode", "efficientnet")
        self.refined_segmentation_mode = self.config["inference"].get("refined_segmentation_mode", "contour")
        self.use_tta = self.config["inference"].get("use_tta", False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load coarse segmentation model
        if self.coarse_segmentation_mode == "sam3":
            self.sam3_model, self.sam3_processor = self._load_sam3_model()
        else:
            self.model = self._load_model(self.model_path, self.config["model"], self.device)

        # Load SAM2/SAM refinement model
        if self.refined_segmentation_mode == "tiling":
            self.sam_model = self._load_ultralytics_sam_model(self.sam_checkpoint, self.device)
        elif self.refined_segmentation_mode == "contour":
            self.sam_model = self._load_hugging_face_sam_model(self.sam_checkpoint)
        elif self.refined_segmentation_mode == "cascadepsp":
            self.sam_model = self._load_cascadepsp_model()
        elif self.refined_segmentation_mode == "crm":
            self.sam_model = self._load_crm_model()

    def _load_sam3_model(self):
        import sam3.model_builder as _sam3_mb
        bpe_path = os.path.join(os.path.dirname(_sam3_mb.__file__), "assets", "bpe_simple_vocab_16e6.txt.gz")
        model = build_sam3_image_model(bpe_path=bpe_path)
        processor = Sam3Processor(model)
        return model, processor

    def _infer_sam3(self, image):
        """
        Single-pass SAM3 inference on an RGB image.

        Args:
            image (np.ndarray): Input RGB image (H, W, 3) uint8.

        Returns:
            np.ndarray: Probability mask (H, W) float32 in [0, 1].
        """
        img_h, img_w = image.shape[:2]
        pil_image = PILImage.fromarray(image)
        text_prompt = self.config["inference"].get("sam3_text_prompt", "A window")
        score_threshold = self.config["inference"].get("sam3_score_threshold", 0.8)

        state = self.sam3_processor.set_image(pil_image)

        # Turn 1: text prompt only
        output = self.sam3_processor.set_text_prompt(state=state, prompt=text_prompt)

        # Turn 2: add high-confidence detections as exemplar prompts; skip if none found
        high_conf = [(s, b) for s, b in zip(output["scores"], output["boxes"]) if s > score_threshold]
        if high_conf:
            self.sam3_processor.reset_all_prompts(state)
            self.sam3_processor.set_text_prompt(state=state, prompt=text_prompt)
            for _, box in high_conf:
                x1, y1, x2, y2 = box.tolist()
                cx, cy = (x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h
                w, h = (x2 - x1) / img_w, (y2 - y1) / img_h
                output = self.sam3_processor.add_geometric_prompt(
                    box=[cx, cy, w, h], label=True, state=state
                )

        # Combine N instance masks into a single score-weighted probability map
        min_object_score = self.config["inference"].get("sam3_min_object_score", 0.6)
        probs = np.zeros((img_h, img_w), dtype=np.float32)
        for mask, score in zip(output["masks"], output["scores"]):
            score_val = float(score)
            if score_val < min_object_score:
                continue
            mask_np = mask.squeeze().detach().cpu().numpy().astype(bool)
            probs = np.where(mask_np, np.maximum(probs, score_val), probs)

        return probs

    def _load_ultralytics_sam_model(self, sam_checkpoint, device):
        return SAM(sam_checkpoint).to(device)

    def _load_hugging_face_sam_model(self, sam_checkpoint):
        return SAM2ImagePredictor.from_pretrained(sam_checkpoint)

    def _load_cascadepsp_model(self):
        import segmentation_refinement as refine
        return refine.Refiner(device=str(self.device))

    def _load_crm_model(self):
        checkpoint = self.config["inference"]["crm_checkpoint"]
        return load_crm_model(checkpoint, self.device)

    def _load_model(self, model_path, model_config, device):
        model = SegmentationModel.load_from_checkpoint(
            model_path, map_location=device, model_config=model_config,
            # learning_rate=0.001, scheduler_config=None  # Placeholder values
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

        return probs

    def _coarse_infer(self, image):
        """
        Run coarse segmentation for a single image (no TTA).

        Args:
            image (np.ndarray): Input RGB image (H, W, 3).

        Returns:
            np.ndarray: Probability mask (H, W) float32 in [0, 1] at original resolution.
        """
        if self.coarse_segmentation_mode == "sam3":
            return self._infer_sam3(image)
        else:
            probs = self._infer(image).squeeze(0)
            return cv2.resize(probs, (image.shape[1], image.shape[0]))

    def recognize(self, image):
        """
        Perform inference on a single image and return the postprocessed mask.

        Args:
            image (np.ndarray): Input RGB image.

        Returns:
            np.ndarray: Postprocessed mask.
        """
        # Perform coarse inference
        self.probs_mask = self._coarse_infer(image)
        if self.use_tta:
            flipped = cv2.flip(image, 1)
            flipped_probs = np.flip(self._coarse_infer(flipped), axis=1)
            self.probs_mask = np.mean([self.probs_mask, flipped_probs], axis=0)

        if not self.probs_mask.any():
            return np.zeros(image.shape[:2], dtype=np.float32)

        if self.refined_segmentation_mode == "tiling":
            refined_mask = process_image_with_tiling(image, self.probs_mask, self.sam_model, self.config, self.device)
        elif self.refined_segmentation_mode == "contour":
            refined_mask = process_contours(image, self.probs_mask, self.sam_model, self.config)
        elif self.refined_segmentation_mode == "cascadepsp":
            refined_mask = process_with_cascadepsp(image, self.probs_mask, self.sam_model, self.config)
            return refined_mask  # skip post-processing for CascadePSP experiment
        elif self.refined_segmentation_mode == "crm":
            refined_mask = process_with_crm(image, self.probs_mask, self.sam_model, self.config["inference"])
            return refined_mask  # skip post-processing for CRM experiment

        postprocessing_mask = post_process_refined_mask(
            refined_mask,
            area_threshold_ratio=0.001,
            epsilon=0.003,
            convex_hull_iou_threshold=0.95,
        )
        return postprocessing_mask
    
    def recognize_from_path(self, image_path):
        """
        Perform inference on an image from a given path and return the postprocessed mask.

        Args:
            image_path (str): Path to the input image.

        Returns:
            np.ndarray: Postprocessed mask.
        """
        self.image = cv2.imread(image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        return self.recognize(self.image)

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

            # Get postprocessed mask
            postprocessing_mask = self.recognize_from_path(image_path)

            # Save raw probability mask
            cv2.imwrite(output_probs_path, (self.probs_mask * 255).astype(np.uint8))

            # Draw overlay with refined mask
            refined_overlay = draw_overlay(self.image, postprocessing_mask)
            cv2.imwrite(output_overlay_path, cv2.cvtColor(refined_overlay, cv2.COLOR_RGB2BGR))

            # Save refined mask
            cv2.imwrite(output_mask_path, (postprocessing_mask * 255).astype(np.uint8))


if __name__ == "__main__":
    pl.seed_everything(0)

    input_folder = "data/official_test"
    output_folder = "data/outputs/official_test_sam3_crm"

    recognitor = WindowsRecognitor("config/config_crm.yaml")
    recognitor.recognize_batch(input_folder, output_folder)
