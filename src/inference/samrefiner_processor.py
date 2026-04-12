import sys
import os
import tempfile
import numpy as np
from PIL import Image as PILImage

import importlib.util as _ilu

_SAMREFINER_ROOT = os.path.join(os.path.dirname(__file__), "../../samrefiner")
_SAM2HQ_ROOT = os.path.join(_SAMREFINER_ROOT, "sam-hq", "sam-hq2")

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


def load_sam2hq_model(checkpoint_path, config_file, device):
    """
    Load a SAM2-HQ model wrapped in SAM2ImagePredictor.

    Temporarily swaps the 'sam2' package in sys.modules to load from the SAM2-HQ
    submodule at samrefiner/sam-hq/sam-hq2/, then restores the host project's sam2.
    After loading, the returned predictor object works independently of sys.path.

    Args:
        checkpoint_path (str): Path to SAM2-HQ .pt checkpoint.
        config_file (str): Config name relative to sam2hq configs/ dir
            (e.g. "sam2.1/sam2.1_hq_hiera_l.yaml").
        device: Target device.

    Returns:
        SAM2ImagePredictor: Loaded SAM2-HQ predictor in eval mode.
    """
    # build_sam2 from sam2hq uses compose(config_name=...) which searches pkg://sam2
    # (the installed main sam2 package). Since sam2hq is not pip-installed we must
    # manually initialise Hydra with sam2hq's own configs/ directory instead.
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    sam2hq_config_dir = os.path.abspath(os.path.join(_SAM2HQ_ROOT, "sam2", "configs"))

    saved_modules = {
        k: sys.modules.pop(k)
        for k in list(sys.modules)
        if k == "sam2" or k.startswith("sam2.")
    }
    sys.path.insert(0, _SAM2HQ_ROOT)
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.build_sam import _load_checkpoint

        hydra_overrides = [
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=sam2hq_config_dir, version_base=None):
            cfg = compose(config_name=config_file, overrides=hydra_overrides)
            OmegaConf.resolve(cfg)
            model = instantiate(cfg.model, _recursive_=True)

        _load_checkpoint(model, checkpoint_path)
        model = model.to(device)
        model.eval()
        predictor = SAM2ImagePredictor(model)
    finally:
        sys.path.remove(_SAM2HQ_ROOT)
        for k in list(sys.modules):
            if k == "sam2" or k.startswith("sam2."):
                del sys.modules[k]
        sys.modules.update(saved_modules)
    return predictor


def process_with_samrefiner(image, probs_mask, sam_model, config):
    """
    Refine a coarse probability mask using SAMRefiner.

    Dispatches to SAM-HQ or SAM2-HQ backend based on
    ``config["samrefiner_backend"]`` (default ``"sam_hq"``).

    Args:
        image (np.ndarray): RGB image (H, W, 3) uint8.
        probs_mask (np.ndarray): Coarse probability mask (H, W) float32 in [0, 1].
        sam_model: Loaded model — Sam-HQ instance for "sam_hq" backend,
            SAM2ImagePredictor for "sam2_hq" backend.
        config (dict): Inference config dict.

    Returns:
        np.ndarray: Refined mask (H, W) float32 in [0, 1].
    """
    backend = config.get("samrefiner_backend", "sam_hq")
    threshold = config.get("samrefiner_threshold", 0.5)
    coarse_mask = (probs_mask >= threshold).astype(np.float32)

    kwargs = dict(
        use_point=config.get("samrefiner_use_point", True),
        use_box=config.get("samrefiner_use_box", True),
        use_mask=config.get("samrefiner_use_mask", True),
        add_neg=config.get("samrefiner_add_neg", True),
        iters=config.get("samrefiner_iters", 5),
        gamma=config.get("samrefiner_gamma", 4.0),
        strength=config.get("samrefiner_strength", 30),
        margin=config.get("samrefiner_margin", 0.0),
    )

    if backend == "sam2_hq":
        return _process_sam2hq(image, coarse_mask, sam_model, kwargs)
    else:
        return _process_samhq(image, coarse_mask, sam_model, kwargs)


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------

def _process_samhq(image, coarse_mask, sam_model, kwargs):
    from sam_refiner import sam_refiner

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp_path = f.name
    try:
        PILImage.fromarray(image).save(tmp_path)
        refined_masks, _, _ = sam_refiner(
            image_path=tmp_path,
            coarse_masks=[coarse_mask],
            sam=sam_model,
            use_samhq=True,
            **kwargs,
        )
    finally:
        os.unlink(tmp_path)

    return refined_masks[0].astype(np.float32)


def _process_sam2hq(image, coarse_mask, predictor, kwargs):
    from sam_refiner import sam2_refiner

    refined_masks, _, _ = sam2_refiner(
        image=image,
        coarse_masks=[coarse_mask],
        predictor=predictor,
        **kwargs,
    )
    return refined_masks[0].astype(np.float32)
