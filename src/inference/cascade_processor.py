import numpy as np


def process_with_cascadepsp(image, probs_mask, cascade_model, config):
    """
    Refine a coarse probability mask using CascadePSP.

    Args:
        image (np.ndarray): RGB image (H, W, 3) uint8.
        probs_mask (np.ndarray): Coarse probability mask (H, W) float32 in [0, 1].
        cascade_model: CascadePSP Refiner instance.
        config (dict): Reads keys:
            - cascadepsp_threshold (float, default 0.5): binarisation threshold for coarse mask.
            - cascadepsp_L (int, default 900): global step resolution.
            - cascadepsp_fast (bool, default False): skip local refinement step if True.

    Returns:
        np.ndarray: Refined mask (H, W) float32 in [0, 1].
    """
    threshold = config.get("cascadepsp_threshold", 0.5)
    L = config.get("cascadepsp_L", 900)
    fast = config.get("cascadepsp_fast", False)

    # CascadePSP expects uint8 mask with values 0 or 255
    binary_mask = (probs_mask >= threshold).astype(np.uint8) * 255

    output = cascade_model.refine(image, binary_mask, fast=fast, L=L)
    # output: uint8 (H, W), values 0 or 255

    return (output / 255.0).astype(np.float32)
