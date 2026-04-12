import sys
import os
import tempfile
import numpy as np
from PIL import Image as PILImage

import importlib.util as _ilu

_SAMREFINER_ROOT = os.path.join(os.path.dirname(__file__), "../../samrefiner")

# sam-hq/ first so its segment_anything (HQ-capable) is used
sys.path.insert(0, os.path.join(_SAMREFINER_ROOT, "sam-hq"))
sys.path.insert(0, _SAMREFINER_ROOT)

# Force samrefiner's utils.py into sys.modules before sam_refiner.py is imported,
# otherwise the project's src/utils/ (already cached) would be used instead.
_utils_spec = _ilu.spec_from_file_location("utils", os.path.join(_SAMREFINER_ROOT, "utils.py"))
_utils_mod = _ilu.module_from_spec(_utils_spec)
sys.modules["utils"] = _utils_mod
_utils_spec.loader.exec_module(_utils_mod)


def load_samrefiner_model(checkpoint_path, model_type, device):
    """
    Load a SAM-HQ model for use with SAMRefiner.

    Args:
        checkpoint_path (str): Path to SAM-HQ .pth checkpoint.
        model_type (str): One of "vit_h", "vit_l", "vit_b", "vit_tiny".
        device (torch.device): Target device.

    Returns:
        Sam: Loaded SAM-HQ model in eval mode.
    """
    from segment_anything import sam_model_registry
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    sam.eval()
    return sam


def process_with_samrefiner(image, probs_mask, sam_model, config):
    """
    Refine a coarse probability mask using SAMRefiner with SAM-HQ.

    Args:
        image (np.ndarray): RGB image (H, W, 3) uint8.
        probs_mask (np.ndarray): Coarse probability mask (H, W) float32 in [0, 1].
        sam_model: Loaded Sam-HQ instance.
        config (dict): Reads keys:
            - samrefiner_threshold (float, default 0.5): binarisation threshold.
            - samrefiner_iters (int, default 5): refinement iterations.
            - samrefiner_gamma (float, default 4.0): geodesic distance gamma.
            - samrefiner_strength (float, default 30): mask-prompt strength.
            - samrefiner_margin (float, default 0.0): bounding-box margin.
            - samrefiner_use_point (bool, default True)
            - samrefiner_use_box (bool, default True)
            - samrefiner_use_mask (bool, default True)
            - samrefiner_add_neg (bool, default True)

    Returns:
        np.ndarray: Refined mask (H, W) float32 in [0, 1].
    """
    from sam_refiner import sam_refiner

    threshold = config.get("samrefiner_threshold", 0.5)
    coarse_mask = (probs_mask >= threshold).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp_path = f.name
    try:
        PILImage.fromarray(image).save(tmp_path)
        refined_masks, _, _ = sam_refiner(
            image_path=tmp_path,
            coarse_masks=[coarse_mask],
            sam=sam_model,
            use_samhq=True,
            use_point=config.get("samrefiner_use_point", True),
            use_box=config.get("samrefiner_use_box", True),
            use_mask=config.get("samrefiner_use_mask", True),
            add_neg=config.get("samrefiner_add_neg", True),
            iters=config.get("samrefiner_iters", 5),
            gamma=config.get("samrefiner_gamma", 4.0),
            strength=config.get("samrefiner_strength", 30),
            margin=config.get("samrefiner_margin", 0.0),
        )
    finally:
        os.unlink(tmp_path)

    # refined_masks: (N, H, W); we passed one mask so N=1
    return refined_masks[0].astype(np.float32)
