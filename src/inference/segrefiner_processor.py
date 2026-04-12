import sys
import os
import numpy as np
import torch

_SEGREFINER_ROOT = os.path.join(os.path.dirname(__file__), "../../segrefiner")
sys.path.insert(0, _SEGREFINER_ROOT)

# Must be imported before any segrefiner/mmdet code to patch mmcv 1.x → 2.x symbols
from inference.segrefiner_compat import *  # noqa: F401, F403

_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def load_segrefiner_model(config_path, checkpoint_path, device):
    """
    Load a SegRefiner model directly (bypassing mmdet.apis.init_detector to
    avoid mmcv.parallel dependency).

    Args:
        config_path (str): Path to the mmdet config file (e.g. segrefiner/configs/segrefiner/segrefiner_hr.py).
        checkpoint_path (str): Path to the .pth checkpoint file.
        device (torch.device): Target device.

    Returns:
        nn.Module: Loaded SegRefiner model in eval mode.
    """
    from mmengine.config import Config
    from mmdet.models import build_detector
    from mmengine.runner import load_checkpoint

    cfg = Config.fromfile(config_path)
    if "pretrained" in cfg.model:
        cfg.model.pretrained = None
    cfg.model.train_cfg = None
    # batch_max=0 in the default test_cfg causes a range(0,N,0) error; set a safe default.
    if cfg.get("test_cfg") is not None and cfg.test_cfg.get("batch_max", 0) == 0:
        cfg.test_cfg.batch_max = 100
    if cfg.model.get("test_cfg") is not None and cfg.model.test_cfg.get("batch_max", 0) == 0:
        cfg.model.test_cfg.batch_max = 100
    # The base configs use type='SegRefiner' (abstract) + task='instance'.
    # Map to the concrete subclass so simple_test_instance is properly defined.
    if cfg.model.get("task") == "instance" and cfg.model.get("type") == "SegRefiner":
        cfg.model.type = "SegRefinerInstance"
    elif cfg.model.get("task") == "semantic" and cfg.model.get("type") == "SegRefiner":
        cfg.model.type = "SegRefinerSemantic"
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, checkpoint_path, map_location="cpu")
    model.to(device)
    model.eval()
    return model


def process_with_segrefiner(image, probs_mask, model, config):
    """
    Refine a coarse probability mask using SegRefiner (instance mode).

    The HR model refines each connected component (window instance) independently
    then composes the results back into a full-image mask.

    Args:
        image (np.ndarray): RGB image (H, W, 3) uint8.
        probs_mask (np.ndarray): Coarse probability mask (H, W) float32 in [0, 1].
        model: Loaded SegRefiner model instance.
        config (dict): Reads keys:
            - segrefiner_threshold (float, default 0.5): binarisation threshold.

    Returns:
        np.ndarray: Refined mask (H, W) float32 in [0, 1].
    """
    import cv2
    from mmdet.core.mask import BitmapMasks

    threshold = config.get("segrefiner_threshold", 0.5)
    H, W = probs_mask.shape
    coarse_binary = (probs_mask >= threshold).astype(np.uint8)

    # Find connected components — each is one window instance
    num_labels, label_map = cv2.connectedComponents(coarse_binary)
    instance_masks = []
    bboxes = []  # [x1, y1, x2, y2, score, class_id]
    for label_id in range(1, num_labels):
        mask = (label_map == label_id).astype(np.uint8)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        instance_masks.append(mask)
        bboxes.append([x1, y1, x2, y2, 1.0, 0])

    # If nothing found, return zeros
    if not instance_masks:
        return np.zeros((H, W), dtype=np.float32)

    device = next(model.parameters()).device

    # Normalize: RGB → BGR, subtract ImageNet mean, divide std, HWC → CHW, add batch dim
    bgr = image[:, :, ::-1].copy().astype(np.float32)
    bgr = (bgr - _MEAN) / _STD
    img_tensor = torch.from_numpy(bgr.transpose(2, 0, 1)).unsqueeze(0).to(device)

    img_meta = dict(
        ori_shape=(H, W, 3),
        img_shape=(H, W, 3),
        pad_shape=(H, W, 3),
        scale_factor=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        flip=False,
        flip_direction=None,
        img_norm_cfg=dict(mean=_MEAN, std=_STD, to_rgb=False),
    )

    # Instance mode expects: coarse_masks=[BitmapMasks], dt_bboxes=[tensor(N,6)],
    # img_metas=[meta_dict] — all list-wrapped for batch=1
    coarse_masks = [BitmapMasks(instance_masks, H, W)]
    dt_bboxes = [np.array(bboxes, dtype=np.float32)]

    with torch.no_grad():
        results = model(
            return_loss=False,
            img=img_tensor,
            img_metas=[img_meta],
            coarse_masks=coarse_masks,
            dt_bboxes=dt_bboxes,
        )

    # results = [(bbox_results, mask_results)]
    # mask_results is [[mask, ...], [], ...] — one list per class, masks are (H, W) uint8
    _, mask_results = results[0]
    refined = np.zeros((H, W), dtype=np.float32)
    for cls_masks in mask_results:
        for m in cls_masks:
            refined = np.maximum(refined, np.array(m, dtype=np.float32))
    return refined
