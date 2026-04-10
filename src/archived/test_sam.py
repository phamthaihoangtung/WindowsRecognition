import cv2  # Import OpenCV for saving masks
import numpy as np  # Import NumPy for array manipulation
import torch  # Import PyTorch for model handling
from ultralytics import SAM
from inference import load_model  # Assuming load_model is defined in inference.py
import yaml  # Import YAML to load configuration
from ultralytics.models.sam.predict import Predictor  # Import Predictor for SAM inference
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def test_sam2_model():
    # Load configuration from YAML
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Load the base model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = load_model('models/base_model.ckpt', config['model'], device)

    # # Initialize the SAM2 model
    # model = SAM('models/SAM/sam2.1_t.pt')

    # Example input for testing
    input_data = {
        'image': 'data/processed/v3/images/test/_DSC0521.jpg',  # Updated with an example image path
        'prompt': {
            'type': 'point',  # Specify the prompt type as 'point'
            'coordinates': [[(100, 150), (200, 250), (300, 350), (400, 450), (450, 400)]]  # Added more point coordinates (x, y)
        }
    }

    # Read image using OpenCV
    image = cv2.imread(input_data['image'])
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {input_data['image']}")

    image = cv2.resize(image, (512, 512))

    # Preprocess image and compute logit
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0  # Convert to tensor

    # Normalize by ImageNet statistics
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    image_tensor = (image_tensor - imagenet_mean) / imagenet_std

    logit = base_model(image_tensor.to(device))  # Get logit from the base model
    logit_resized = torch.nn.functional.interpolate(logit, size=(256, 256), mode='bilinear', align_corners=False)  # Resize logit to 256x256

    mask_prompt = logit_resized.squeeze().detach().cpu().numpy()

    # Save logit to a file
    logit_path = 'data/test_sam/logit_output.npy'
    np.save(logit_path, mask_prompt > 0.5)
    print(f"Logit saved to '{logit_path}'")

    # Step 1: Initialize SAM Predictor and model using overrides
    predictor = Predictor(overrides={"model": "models/SAM/sam2.1_b.pt"})  # Adjust model path as needed
    predictor.setup_model(verbose=True)  # Enable verbose output during model setup

    # Step 2: Set image
    img_bgr = cv2.imread(input_data['image'])
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found at path: {input_data['image']}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    print("Image shape:", img_rgb.shape)
    features = predictor.set_image(img_rgb)

    # 3. Retrieve the preprocessed tensor for inference
    # 3. Cache features (returns None)
    predictor.set_image(img_rgb)

    # 4. Preprocess into (1,C,H,W) tensor
    im_tensor = predictor.preprocess([img_rgb])

    # print("Features shape:", features.shape)

    # Step 3: First inference using points as a prompt
    points = input_data['prompt']['coordinates']

    # Step 4: Refine prediction by using logits as mask prompt
    refined_masks, refined_scores, refined_logits = predictor.inference(
        im_tensor,
        points=points,
        masks=mask_prompt,  # Use the logits from the previous output as the prompt
        multimask_output=False
    )

    # Save refined masks to files
    for idx, refined_mask in enumerate(refined_masks):
        mask_path = f'data/test_sam/refined_mask_{idx}.png'
        cv2.imwrite(mask_path, (refined_mask * 255).astype(np.uint8))
        print(f"Refined mask {idx} saved to '{mask_path}'")

if __name__ == "__main__":
    test_sam2_model()
