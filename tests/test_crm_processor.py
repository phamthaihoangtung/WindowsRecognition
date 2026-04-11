import os
import sys
import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from inference.crm_processor import process_with_crm, _make_coord

OFFICIAL_TEST_DIR = "data/official_test"
TEST_IMAGES = ["0O9A9803.jpg"]


# --- Helpers ---

def make_mock_model(H, W, device="cpu"):
    """CRMNet mock that returns a plausible pred_224 tensor."""
    model = MagicMock(spec=nn.DataParallel)
    model.parameters.return_value = iter([torch.zeros(1, device=device)])

    def forward(im, seg, coord, cell, transferCoord, transferFeat):
        n = coord.shape[1]
        pred = torch.zeros(1, n, 1, device=device)
        out_dict = {"pred_224": pred}
        if transferCoord is None:
            return out_dict, torch.zeros(1, device=device), torch.zeros(1, device=device)
        return out_dict

    model.side_effect = forward
    model.__call__ = forward
    return model


# --- Unit tests (no weights needed) ---

def test_make_coord_shape():
    coord = _make_coord((32, 48))
    assert coord.shape == (32 * 48, 2)


def test_make_coord_range():
    coord = _make_coord((16, 16))
    assert coord.min().item() >= -1.0
    assert coord.max().item() <= 1.0


def test_output_dtype_and_range():
    H, W = 64, 64
    image = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    probs_mask = np.random.rand(H, W).astype(np.float32)
    model = make_mock_model(H, W)
    result = process_with_crm(image, probs_mask, model, {"crm_scales": [1.0]})
    assert result.dtype == np.float32
    assert 0.0 <= result.min() and result.max() <= 1.0
    assert result.shape == (H, W)


def test_output_shape_matches_input():
    H, W = 50, 80
    image = np.zeros((H, W, 3), dtype=np.uint8)
    probs_mask = np.ones((H, W), dtype=np.float32) * 0.7
    model = make_mock_model(H, W)
    result = process_with_crm(image, probs_mask, model, {"crm_scales": [0.5, 1.0]})
    assert result.shape == (H, W)


def test_threshold_binarises_mask():
    """Below-threshold pixels should contribute a zero binary mask."""
    H, W = 32, 32
    image = np.zeros((H, W, 3), dtype=np.uint8)
    probs_mask = np.full((H, W), 0.3, dtype=np.float32)  # all below default threshold 0.5

    captured = {}

    def forward(im, seg, coord, cell, transferCoord, transferFeat):
        captured["seg"] = seg.clone()
        n = coord.shape[1]
        pred = torch.zeros(1, n, 1)
        out_dict = {"pred_224": pred}
        if transferCoord is None:
            return out_dict, torch.zeros(1), torch.zeros(1)
        return out_dict

    model = MagicMock(spec=nn.DataParallel)
    model.parameters.return_value = iter([torch.zeros(1)])
    model.__call__ = forward
    model.side_effect = forward

    process_with_crm(image, probs_mask, model, {"crm_threshold": 0.5, "crm_scales": [1.0]})
    seg = captured["seg"]
    # normalised with mean=0.5 std=0.5 → binary 0 → (0 - 0.5)/0.5 = -1.0
    assert torch.allclose(seg, torch.full_like(seg, -1.0))


def test_scales_averaged():
    """Multiple scales should be averaged in the output."""
    H, W = 16, 16
    image = np.zeros((H, W, 3), dtype=np.uint8)
    probs_mask = np.ones((H, W), dtype=np.float32)
    call_count = {"n": 0}

    def forward(im, seg, coord, cell, transferCoord, transferFeat):
        call_count["n"] += 1
        n = coord.shape[1]
        pred = torch.full((1, n, 1), float(call_count["n"]))
        out_dict = {"pred_224": pred}
        if transferCoord is None:
            return out_dict, torch.zeros(1), torch.zeros(1)
        return out_dict

    model = MagicMock(spec=nn.DataParallel)
    model.parameters.return_value = iter([torch.zeros(1)])
    model.__call__ = forward
    model.side_effect = forward

    process_with_crm(image, probs_mask, model, {"crm_scales": [0.5, 1.0]})
    assert call_count["n"] == 2


# --- Integration test: full WindowsRecognitor flow on real images ---

@pytest.mark.skipif(not os.path.isdir(OFFICIAL_TEST_DIR), reason="test images not available")
@pytest.mark.timeout(300)
def test_full_flow_real_images():
    """
    Runs complete Stage1 (SAM3) -> Stage2 (CRM) pipeline on real images.
    Requires SAM3 + CRM weights to be installed.
    Saves outputs to data/outputs/test_crm/ for visual inspection.
    """
    from inference_sam import WindowsRecognitor
    from utils.utils import draw_overlay

    output_dir = "data/outputs/test_crm"
    os.makedirs(output_dir, exist_ok=True)

    recognitor = WindowsRecognitor("config/config_crm.yaml")

    for fname in TEST_IMAGES:
        image_path = os.path.join(OFFICIAL_TEST_DIR, fname)
        if not os.path.exists(image_path):
            pytest.skip(f"{fname} not found")

        mask = recognitor.recognize_from_path(image_path)
        stem = os.path.splitext(fname)[0]

        cv2.imwrite(
            os.path.join(output_dir, f"{stem}_mask.png"),
            (mask * 255).astype(np.uint8),
        )
        overlay = draw_overlay(recognitor.image, mask)
        cv2.imwrite(
            os.path.join(output_dir, f"{stem}_overlay.png"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        )

        assert mask.shape == recognitor.image.shape[:2], f"Shape mismatch for {fname}"
        assert mask.dtype == np.float32
        assert 0.0 <= mask.min() and mask.max() <= 1.0
