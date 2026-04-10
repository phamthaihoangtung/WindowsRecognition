import argparse
import cv2
import numpy as np
import torch
import yaml
from sam2.sam2_image_predictor import SAM2ImagePredictor

SAM_CHECKPOINT = "facebook/sam2.1-hiera-large"
IMAGE_PATH = "data/interiors_crawled/pexels_3284980.jpg"
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-mask-prompt", action="store_true", help="Disable mask prompt for SAM inference")
    args = parser.parse_args()
    test_sam2_model(use_mask_prompt=not args.no_mask_prompt)
