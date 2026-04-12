import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../crm/High-Quality-Segmention"))

import torch
import torch.nn as nn
import numpy as np


def _make_coord(shape):
    """Generate LIIF coordinate grid of shape (H*W, 2) with values in [-1, 1]."""
    H, W = shape
    ys = torch.linspace(-1 + 1 / H, 1 - 1 / H, H)
    xs = torch.linspace(-1 + 1 / W, 1 - 1 / W, W)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_y, grid_x], dim=-1).view(-1, 2)  # (H*W, 2) — [height, width] to match CRM convention


def load_crm_model(checkpoint_path, device):
    """
    Load a CRMNet model from a local checkpoint.

    Args:
        checkpoint_path (str): Path to the .pth checkpoint file.
        device (torch.device): Target device.

    Returns:
        nn.DataParallel: Loaded CRMNet in eval mode.
    """
    from models.network.crm_transferCoord_transferFeat import CRMNet

    model = nn.DataParallel(CRMNet(backend="resnet50"))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def process_with_crm(image, probs_mask, crm_model, config):
    """
    Refine a coarse probability mask using CRM.

    Args:
        image (np.ndarray): RGB image (H, W, 3) uint8.
        probs_mask (np.ndarray): Coarse probability mask (H, W) float32 in [0, 1].
        crm_model: Loaded CRMNet (nn.DataParallel) instance.
        config (dict): Reads keys:
            - crm_threshold (float, default 0.5): binarisation threshold for coarse mask.
            - crm_scales (list, default [0.25, 0.5, 1.0]): multi-scale inference ratios.

    Returns:
        np.ndarray: Refined mask (H, W) float32 in [0, 1].
    """
    threshold = config.get("crm_threshold", 0.5)
    scales = config.get("crm_scales", [0.25, 0.5, 1.0])
    memory_chunk = 50176 * 4  # safe chunk size for GPU memory

    device = next(crm_model.parameters()).device

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    H, W = image.shape[:2]
    binary_mask = (probs_mask >= threshold).astype(np.float32)

    im_t = torch.from_numpy(image).permute(2, 0, 1).float().to(device) / 255.0
    im_t = (im_t - mean) / std  # (3, H, W)

    seg_t = torch.from_numpy(binary_mask).unsqueeze(0).float().to(device)
    seg_t = (seg_t - 0.5) / 0.5  # (1, H, W), normalised to [-1, 1]

    # Pad by 32px with reflect — matches the training pipeline (OfflineDataset_crm_pad32).
    # Without this, boundary convolutions see zeros instead of reflected content, causing
    # corner/edge artifacts in the output.
    pad = 32
    im_t = torch.nn.functional.pad(im_t.unsqueeze(0), (pad, pad, pad, pad), mode="reflect").squeeze(0)
    seg_t = torch.nn.functional.pad(seg_t.unsqueeze(0), (pad, pad, pad, pad), mode="reflect").squeeze(0)

    pH, pW = H + 2 * pad, W + 2 * pad
    coord = _make_coord((pH, pW)).to(device)  # (pH*pW, 2)
    cell = torch.tensor([2 / pH, 2 / pW], device=device).unsqueeze(0).expand(pH * pW, -1)  # (pH*pW, 2)

    # seg_t is updated after each scale (iterative refinement)
    cur_seg = seg_t.unsqueeze(0)  # (1, 1, pH, pW)

    with torch.no_grad():
        for s in scales:
            sH = max(1, int(pH * s))
            sW = max(1, int(pW * s))
            im_s = torch.nn.functional.interpolate(
                im_t.unsqueeze(0), size=(sH, sW), mode="bilinear", align_corners=True
            )
            seg_s = torch.nn.functional.interpolate(
                cur_seg, size=(sH, sW), mode="bilinear", align_corners=True
            )

            # Reset transfer cache for each scale
            transferCoord = None
            transferFeat = None

            coord_chunks = coord.split(memory_chunk, dim=0)
            cell_chunks = cell.split(memory_chunk, dim=0)
            pred_parts = []
            for c_coord, c_cell in zip(coord_chunks, cell_chunks):
                out = crm_model(
                    im_s,
                    seg_s,
                    c_coord.unsqueeze(0),
                    c_cell.unsqueeze(0),
                    transferCoord,
                    transferFeat,
                )
                if isinstance(out, tuple):
                    out_dict, transferCoord, transferFeat = out
                else:
                    out_dict = out
                pred_parts.append(out_dict["pred_224"].squeeze(0))  # (chunk, 1)

            # pred_224 is already in [0, 1] — no sigmoid needed
            pred = torch.cat(pred_parts, dim=0).view(pH, pW)  # (pH, pW)

            # Update seg for next scale: convert [0,1] prediction to [-1,1]
            cur_seg = ((pred.unsqueeze(0).unsqueeze(0) - 0.5) * 2)  # (1, 1, pH, pW)

    # Crop padding back to original resolution
    pred_cropped = pred[pad:pad + H, pad:pad + W]
    refined = pred_cropped.cpu().numpy()
    return refined