import os
import sys
import cv2
import numpy as np
import pytest
from unittest.mock import MagicMock, ANY

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from inference.cascade_processor import process_with_cascadepsp

OFFICIAL_TEST_DIR = "data/official_test"
TEST_IMAGES = ["0O9A9803.jpg", "0V7A2364.jpg", "_DSC0526.jpg", "_DSC4368.jpg"]


# --- Unit tests (mocked model, no weights needed) ---

def make_mock_model(output_mask):
    model = MagicMock()
    model.refine.return_value = output_mask
    return model


def test_output_dtype_and_range():
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    probs_mask = np.random.rand(256, 256).astype(np.float32)
    raw_output = (probs_mask > 0.5).astype(np.uint8) * 255
    result = process_with_cascadepsp(image, probs_mask, make_mock_model(raw_output), {})
    assert result.dtype == np.float32
    assert 0.0 <= result.min() and result.max() <= 1.0


def test_input_mask_binarised():
    """probs_mask must arrive at the model as uint8 0/255, not float."""
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    probs_mask = np.full((64, 64), 0.6, dtype=np.float32)
    model = make_mock_model(np.full((64, 64), 255, dtype=np.uint8))
    process_with_cascadepsp(image, probs_mask, model, {"cascadepsp_threshold": 0.5})
    passed_mask = model.refine.call_args[0][1]
    assert passed_mask.dtype == np.uint8
    assert set(np.unique(passed_mask)) <= {0, 255}


def test_config_keys_forwarded():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    probs_mask = np.zeros((64, 64), dtype=np.float32)
    model = make_mock_model(np.zeros((64, 64), dtype=np.uint8))
    process_with_cascadepsp(image, probs_mask, model, {"cascadepsp_L": 1200, "cascadepsp_fast": True})
    model.refine.assert_called_once_with(ANY, ANY, fast=True, L=1200)


# --- Integration test: full WindowsRecognitor flow on 3 real images ---

@pytest.mark.skipif(not os.path.isdir(OFFICIAL_TEST_DIR), reason="test images not available")
@pytest.mark.timeout(300)
def test_full_flow_real_images():
    """
    Runs complete Stage1 (SAM3) -> Stage2 (CascadePSP) pipeline on 3 images.
    Requires SAM3 + CascadePSP model weights to be installed.
    Saves outputs to data/outputs/test_cascadepsp/ for visual inspection.
    """
    from inference_sam import WindowsRecognitor
    from utils.utils import draw_overlay

    output_dir = "data/outputs/test_cascadepsp"
    os.makedirs(output_dir, exist_ok=True)

    recognitor = WindowsRecognitor("config/config_cascadepsp.yaml")

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
