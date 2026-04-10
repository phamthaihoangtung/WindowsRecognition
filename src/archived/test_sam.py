import argparse
import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

SAM_CHECKPOINT = "facebook/sam2.1-hiera-large"
IMAGE_PATH = "data/interiors_crawled/pexels_3284980.jpg"
TEXT_PROMPT = "A window"
POINT_COORDS = np.array([[100, 150], [200, 250], [300, 350], [400, 450], [450, 400]])
POINT_LABELS = np.ones(len(POINT_COORDS), dtype=np.int32)


def test_sam2_model(use_mask_prompt: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_bgr = cv2.imread(IMAGE_PATH)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found at path: {IMAGE_PATH}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    mask_prompt = None
    if use_mask_prompt:
        from inference import load_model

        with open("config/config.yaml", "r") as file:
            config = yaml.safe_load(file)

        base_model = load_model("models/base_model.ckpt", config["model"], device)

        image = cv2.resize(img_rgb, (512, 512))
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        image_tensor = (image_tensor - imagenet_mean) / imagenet_std

        logit = base_model(image_tensor.to(device))
        logit_resized = torch.nn.functional.interpolate(logit, size=(256, 256), mode="bilinear", align_corners=False)
        mask_prompt = logit_resized.squeeze().detach().cpu().numpy()

        logit_path = "data/test_sam/logit_output.npy"
        np.save(logit_path, mask_prompt > 0.5)
        print(f"Logit saved to '{logit_path}'")

    predictor = SAM2ImagePredictor.from_pretrained(SAM_CHECKPOINT)

    with torch.inference_mode():
        predictor.set_image(img_rgb)
        masks, scores, _ = predictor.predict(
            point_coords=POINT_COORDS,
            point_labels=POINT_LABELS,
            mask_input=mask_prompt[None] if mask_prompt is not None else None,
            multimask_output=False,
        )

    for idx, mask in enumerate(masks):
        mask_path = f"data/test_sam/refined_mask_{idx}.png"
        cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
        print(f"Mask {idx} saved to '{mask_path}' (score: {scores[idx]:.3f})")


COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255)]


def _save_masks_with_overlay(img_bgr, masks, boxes, scores, mask_prefix, overlay_path):
    overlay = img_bgr.copy()
    for idx, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
        mask_arr = (np.squeeze(mask.detach().cpu().numpy()) * 255).astype(np.uint8)
        cv2.imwrite(f"data/test_sam/{mask_prefix}_{idx}.png", mask_arr)
        contours, _ = cv2.findContours(mask_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, COLORS[idx % len(COLORS)], thickness=3)
        print(f"  Mask {idx}: score={score:.3f}, box={box}")
    cv2.imwrite(f"data/test_sam/{overlay_path}", overlay)
    print(f"  Overlay saved to 'data/test_sam/{overlay_path}'")


def test_sam3_model():
    img_bgr = cv2.imread(IMAGE_PATH)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found at path: {IMAGE_PATH}")
    img_h, img_w = img_bgr.shape[:2]
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    state = processor.set_image(img_pil)

    # Turn 1: text only
    print("=== Turn 1: text prompt only ===")
    output1 = processor.set_text_prompt(state=state, prompt=TEXT_PROMPT)
    _save_masks_with_overlay(img_bgr, output1["masks"], output1["boxes"], output1["scores"],
                             mask_prefix="sam3_t1_mask", overlay_path="sam3_t1_overlay.png")

    # Turn 2: best confident box (score > 0.8) from turn 1 + same text
    high_conf = [(s, b) for s, b in zip(output1["scores"], output1["boxes"]) if s > 0.8]
    if not high_conf:
        print("No masks with score > 0.8 in turn 1, skipping turn 2.")
        return

    print(f"\n=== Turn 2: text + {len(high_conf)} exemplar box(es) (score >= 0.8) ===")
    processor.reset_all_prompts(state)
    processor.set_text_prompt(state=state, prompt=TEXT_PROMPT)
    for score, box in high_conf:
        x1, y1, x2, y2 = box.tolist()
        cx, cy = (x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h
        w, h = (x2 - x1) / img_w, (y2 - y1) / img_h
        output2 = processor.add_geometric_prompt(box=[cx, cy, w, h], label=True, state=state)
        print(f"  Added exemplar box with score={score:.3f}")
    _save_masks_with_overlay(img_bgr, output2["masks"], output2["boxes"], output2["scores"],
                             mask_prefix="sam3_t2_mask", overlay_path="sam3_t2_overlay.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["sam2", "sam3"], default="sam2", help="Which SAM model to test")
    parser.add_argument("--mask-prompt", action="store_true", help="Use mask prompt for SAM2 inference")
    args = parser.parse_args()

    if args.model == "sam3":
        test_sam3_model()
    else:
        test_sam2_model(use_mask_prompt=args.mask_prompt)
