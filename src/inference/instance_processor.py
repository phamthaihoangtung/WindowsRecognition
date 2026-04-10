import os
import tempfile
import numpy as np
import cv2
import torch

from .point_generator import PointGenerator
from .contour_processor import crop_image_and_mask


def _to_numpy(x):
    return x.cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def process_instances_sam3(image, instances, predictor, config):
    """
    Refine each SAM3 instance using text + box (first call) then iterative
    point correction (subsequent calls) via the SAM3 video predictor.

    Text/box and point prompts are mutually exclusive per add_prompt call:
      - Call 1: text + box  →  initial detection, returns obj_id
      - Calls 2+: points only (with obj_id)  →  FP/FN correction

    Args:
        image (np.ndarray): RGB uint8 (H, W, 3).
        instances (list[dict]): Each dict has keys:
            "mask"   np.ndarray (H, W) bool  — Stage 1 coarse mask
            "box"    [x1, y1, x2, y2]        — pixel coords
            "score"  float
        predictor: Sam3VideoPredictorMultiGPU
        config (dict): inference config.

    Returns:
        np.ndarray: Refined mask (H, W) float32 in [0, 1].
    """
    text_prompt = config.get("sam3_text_prompt", "A window")
    sam_consecutive_iterations = config.get("sam_consecutive_iterations", 10)
    num_points = config.get("num_points", 1000)
    extend_ratio = config.get("extend_ratio", 0.2)
    kernel = np.ones((15, 15), np.uint8)

    img_h, img_w = image.shape[:2]
    refined_mask = np.zeros((img_h, img_w), dtype=np.float32)

    for instance in instances:
        coarse_mask = instance["mask"].astype(np.float32)  # (H, W)
        x1, y1, x2, y2 = instance["box"]
        bw, bh = x2 - x1, y2 - y1

        # Crop with padding
        x_start = max(0, int(x1 - bw * extend_ratio))
        y_start = max(0, int(y1 - bh * extend_ratio))
        x_end = min(img_w, int(x2 + bw * extend_ratio))
        y_end = min(img_h, int(y2 + bh * extend_ratio))

        cropped_image = image[y_start:y_end, x_start:x_end]
        cropped_mask = coarse_mask[y_start:y_end, x_start:x_end]

        crop_h, crop_w = cropped_image.shape[:2]
        if crop_h == 0 or crop_w == 0:
            continue

        # Instance box normalized to crop space [x_min, y_min, w, h]
        box_norm = [
            max(0.0, (x1 - x_start) / crop_w),
            max(0.0, (y1 - y_start) / crop_h),
            min(1.0, (x2 - x_start) / crop_w) - max(0.0, (x1 - x_start) / crop_w),
            min(1.0, (y2 - y_start) / crop_h) - max(0.0, (y1 - y_start) / crop_h),
        ]

        # Save crop to temp file (video predictor requires a file path)
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp_path = tmp.name
        tmp.close()
        cv2.imwrite(tmp_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))

        session_id = None
        predicted_mask = cropped_mask  # fallback: use Stage 1 coarse mask
        try:
            response = predictor.handle_request({
                "type": "start_session",
                "resource_path": tmp_path,
            })
            session_id = response["session_id"]

            # Call 1: text + box  →  initial detection
            response = predictor.handle_request({
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": text_prompt,
                "bounding_boxes": [box_norm],
                "bounding_box_labels": [1],
            })
            outputs = response["outputs"]

            if outputs is None or len(outputs["out_obj_ids"]) == 0:
                # No detection — fall back to Stage 1 coarse mask for this instance
                refined_mask[y_start:y_end, x_start:x_end] = np.maximum(
                    refined_mask[y_start:y_end, x_start:x_end], cropped_mask
                )
                continue

            # Use first (highest-score) object
            obj_id = int(outputs["out_obj_ids"][0])
            predicted_mask = _to_numpy(outputs["out_binary_masks"][0]).astype(np.float32)

            # Build point generator from the coarse mask of this crop
            point_generator = PointGenerator(
                num_points, cropped_mask=cropped_mask, kernel=kernel, prob_thresh=0.5
            )

            # Calls 2+: point-only correction (FP/FN)
            for iter_idx in range(sam_consecutive_iterations):
                if iter_idx == 0:
                    new_points = point_generator.retrieve_random_points(num_points=5)
                else:
                    new_points = point_generator.retrieve_key_points(
                        predicted_mask, num_points=5
                    )

                if not new_points["input_point"][0]:
                    break

                pts_px = new_points["input_point"][0]   # [(x, y), ...] pixel in crop
                labels = new_points["input_label"][0]   # [1/0, ...]

                # Normalize to [0, 1] — video predictor default is rel_coordinates=True
                pts_norm = [[x / crop_w, y / crop_h] for x, y in pts_px]

                response = predictor.handle_request({
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "points": pts_norm,
                    "point_labels": labels,
                    "obj_id": obj_id,
                })

                out = response["outputs"]
                if out is not None and len(out["out_obj_ids"]) > 0:
                    obj_ids_out = out["out_obj_ids"].tolist()
                    if obj_id in obj_ids_out:
                        idx = obj_ids_out.index(obj_id)
                        predicted_mask = (
                            _to_numpy(out["out_binary_masks"][idx]).astype(np.float32)
                        )

        finally:
            if session_id is not None:
                predictor.handle_request({
                    "type": "close_session",
                    "session_id": session_id,
                })
            os.unlink(tmp_path)

        refined_mask[y_start:y_end, x_start:x_end] = np.maximum(
            refined_mask[y_start:y_end, x_start:x_end], predicted_mask
        )

    return refined_mask
