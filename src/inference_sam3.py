import os
import tempfile
import glob

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm
import pytorch_lightning as pl

from sam3.model_builder import build_sam3_video_predictor
from utils.utils import draw_overlay
from inference.post_processor import post_process_refined_mask
from inference.instance_processor import process_instances_sam3


class WindowsRecognitorSam3:
    """
    Two-stage window segmentation using SAM3 throughout.

    Stage 1 — Detection (video predictor, text + exemplar boxes):
        Full image → N instance masks + bounding boxes.

    Stage 2 — Refinement (video predictor, text + box + iterative points):
        Each instance crop → SAM3 text+box initial pass, then FP/FN point
        correction (1 run, sam_consecutive_iterations iterations).

    Stage 3 — Post-processing:
        Contour filtering, convex-hull / Douglas-Peucker simplification.

    A single Sam3VideoPredictorMultiGPU handles both Stage 1 and Stage 2
    (one weight load).
    """

    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = build_sam3_video_predictor()

    # ------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------

    def _infer_sam3(self, image, image_path=None):
        """
        Run SAM3 detection on the full image.

        Turn 1: text prompt only.
        Turn 2: if any detections exceed sam3_score_threshold, re-run with
                text + those boxes as exemplar prompts.

        Args:
            image (np.ndarray): RGB uint8 (H, W, 3).
            image_path (str | None): If provided, used directly as the
                session resource path (no temp file needed).

        Returns:
            list[dict]: Each dict has "mask" (H, W) bool, "box" [x1,y1,x2,y2],
                        "score" float, "obj_id" int.
        """
        text_prompt = self.config["inference"].get("sam3_text_prompt", "A window")
        score_threshold = self.config["inference"].get("sam3_score_threshold", 0.8)
        img_h, img_w = image.shape[:2]

        cleanup = False
        if image_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            image_path = tmp.name
            tmp.close()
            cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cleanup = True

        session_id = None
        instances = []
        try:
            response = self.predictor.handle_request({
                "type": "start_session",
                "resource_path": image_path,
            })
            session_id = response["session_id"]

            # Turn 1: text only
            response = self.predictor.handle_request({
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": text_prompt,
            })
            instances = self._parse_outputs(response["outputs"], img_h, img_w)

            # Turn 2: high-confidence detections as exemplar boxes
            high_conf = [inst for inst in instances if inst["score"] > score_threshold]
            if high_conf:
                # boxes_xywh: [x_min_norm, y_min_norm, w_norm, h_norm]
                exemplar_boxes = [
                    [
                        inst["box"][0] / img_w,
                        inst["box"][1] / img_h,
                        (inst["box"][2] - inst["box"][0]) / img_w,
                        (inst["box"][3] - inst["box"][1]) / img_h,
                    ]
                    for inst in high_conf
                ]
                response = self.predictor.handle_request({
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "text": text_prompt,
                    "bounding_boxes": exemplar_boxes,
                    "bounding_box_labels": [1] * len(exemplar_boxes),
                })
                instances = self._parse_outputs(response["outputs"], img_h, img_w)

        finally:
            if session_id is not None:
                self.predictor.handle_request({
                    "type": "close_session",
                    "session_id": session_id,
                })
            if cleanup:
                os.unlink(image_path)

        return instances

    def _parse_outputs(self, outputs, img_h, img_w):
        """Convert video predictor outputs to an instance list."""
        if outputs is None:
            return []
        out_masks = outputs["out_binary_masks"]    # (N, H, W) bool
        out_probs = outputs["out_probs"]           # (N,)
        out_boxes = outputs["out_boxes_xywh"]      # (N, 4) normalized [x,y,w,h]
        out_obj_ids = outputs["out_obj_ids"]       # (N,)

        instances = []
        for mask, score, box_xywh, obj_id in zip(
            out_masks, out_probs, out_boxes, out_obj_ids
        ):
            x_norm, y_norm, w_norm, h_norm = box_xywh.tolist()
            instances.append({
                "mask": mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask,
                "box": [
                    x_norm * img_w,
                    y_norm * img_h,
                    (x_norm + w_norm) * img_w,
                    (y_norm + h_norm) * img_h,
                ],
                "score": float(score),
                "obj_id": int(obj_id),
            })
        return instances

    # ------------------------------------------------------------------
    # Inference entry points
    # ------------------------------------------------------------------

    def recognize(self, image, image_path=None):
        """
        Full pipeline on a single RGB image.

        Args:
            image (np.ndarray): RGB uint8 (H, W, 3).
            image_path (str | None): Optional file path for Stage 1 session
                (avoids saving a temp file when the path is already known).

        Returns:
            np.ndarray: Post-processed binary mask (H, W) float32.
        """
        # Stage 1
        instances = self._infer_sam3(image, image_path=image_path)

        # Filter instances too small for reliable Stage 2 refinement
        inf_config = self.config.get("inference", {})
        img_area = image.shape[0] * image.shape[1]
        min_ratio = inf_config.get("min_instance_area_ratio", 0.001)
        instances = [inst for inst in instances if inst["mask"].sum() / img_area >= min_ratio]

        # Store score-weighted union for visualization
        self.probs_mask = np.zeros(image.shape[:2], dtype=np.float32)
        for inst in instances:
            self.probs_mask = np.where(
                inst["mask"],
                np.maximum(self.probs_mask, inst["score"]),
                self.probs_mask,
            )

        if not instances:
            return np.zeros(image.shape[:2], dtype=np.float32)

        # Stage 2
        refined_mask = process_instances_sam3(image, instances, self.predictor, inf_config)

        # Stage 3
        postprocessing_mask = post_process_refined_mask(
            refined_mask,
            area_threshold_ratio=0.001,
            epsilon=0.003,
            convex_hull_iou_threshold=0.99,
        )
        return postprocessing_mask

    def recognize_from_path(self, image_path):
        """Load image from path and run the full pipeline."""
        self.image = cv2.imread(image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        return self.recognize(self.image, image_path=image_path)

    def recognize_batch(self, input_folder, output_folder):
        """Process all .jpg images in input_folder and save results."""
        os.makedirs(output_folder, exist_ok=True)

        for image_path in tqdm(
            glob.glob(os.path.join(input_folder, "*.jpg")), desc="Processing images"
        ):
            image_name = os.path.basename(image_path)
            stem = os.path.splitext(image_name)[0]
            output_probs_path = os.path.join(output_folder, f"{stem}_probs.png")
            output_overlay_path = os.path.join(output_folder, f"{stem}_overlay.png")
            output_mask_path = os.path.join(output_folder, f"{stem}_mask.png")

            postprocessing_mask = self.recognize_from_path(image_path)

            cv2.imwrite(output_probs_path, (self.probs_mask * 255).astype(np.uint8))

            refined_overlay = draw_overlay(self.image, postprocessing_mask)
            cv2.imwrite(
                output_overlay_path, cv2.cvtColor(refined_overlay, cv2.COLOR_RGB2BGR)
            )

            cv2.imwrite(output_mask_path, (postprocessing_mask * 255).astype(np.uint8))


if __name__ == "__main__":
    import argparse

    pl.seed_everything(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="Path to a single image for testing")
    parser.add_argument("--input-folder", type=str, default="data/interiors_crawled")
    parser.add_argument("--output-folder", type=str, default="data/outputs/interiors_crawled_sam3_full")
    args = parser.parse_args()

    recognitor = WindowsRecognitorSam3("config/config_sam3.yaml")

    if args.image:
        output_folder = args.output_folder
        os.makedirs(output_folder, exist_ok=True)
        stem = os.path.splitext(os.path.basename(args.image))[0]

        mask = recognitor.recognize_from_path(args.image)

        cv2.imwrite(os.path.join(output_folder, f"{stem}_probs.png"),
                    (recognitor.probs_mask * 255).astype(np.uint8))
        overlay = draw_overlay(recognitor.image, mask)
        cv2.imwrite(os.path.join(output_folder, f"{stem}_overlay.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_folder, f"{stem}_mask.png"),
                    (mask * 255).astype(np.uint8))
        print(f"Saved to {output_folder}/{stem}_{{probs,overlay,mask}}.png")
    else:
        recognitor.recognize_batch(args.input_folder, args.output_folder)
