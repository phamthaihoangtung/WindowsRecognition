import os
import torch
from data_loader import SegmentationDataset
from torch.utils.data import DataLoader
import segmentation_models_pytorch.metrics.functional as smp_metrics_functional
from model import SegmentationModel
import yaml

def evaluate_model(config=None, model=None, test_dataset=None, device="cuda"):
    """
    Evaluate the model on the test set.

    Args:
        config (dict, optional): Configuration dictionary. If None, it will load from the default config file.
        model (SegmentationModel, optional): Trained model. If None, it will be loaded from the checkpoint.
        test_dataset (Dataset, optional): Test dataset. If None, it will be constructed from the config.
        device (str, optional): Device to run the evaluation on. Default is "cuda".

    Returns:
        dict: Evaluation metrics including loss and IoU.
    """
    if config is None:
        with open("config/config.yaml", "r") as file:
            config = yaml.safe_load(file)

    # Load the trained model if not provided
    if model is None:
        model = SegmentationModel(
            model_config=config["model"],
            learning_rate=config["hyperparameters"]["learning_rate"],
            scheduler_config=config["hyperparameters"]["scheduler"],
            use_wandb=False,  # Disable wandb for evaluation
        )
        model.load_state_dict(torch.load(config["paths"]["model_save_path"]))
    
    model.eval()
    model = model.to(device)  # Ensure model is on the correct device

    # Create the test loader from the provided dataset or config
    if test_dataset is None:
        test_dataset = SegmentationDataset(
            images_dir=config["paths"]["test_images"],
            masks_dir=config["paths"]["test_masks"],
            transform=None,  # Use the same transforms as validation if needed
            classes=config["model"]["classes"],
        )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Initialize metrics
    total_loss = 0
    total_iou = 0
    num_samples = 0
    loss_fn = model.loss_fn

    # Evaluate the model
    with torch.no_grad():
        for images, masks in test_loader:
            masks = masks.float()  # Ensure masks are float
            images, masks = images.to(device), masks.to(device)  # Move data to the correct device
            outputs = model(images)
            activated_outputs = torch.sigmoid(outputs)

            # Compute loss
            loss = loss_fn(outputs, masks)
            total_loss += loss.item()

            # Compute IoU
            tp, fp, fn, tn = smp_metrics_functional.get_stats(
                activated_outputs, masks.long(), mode="binary", threshold=model.iou_threshold
            )
            iou = smp_metrics_functional.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
            total_iou += iou.item()

            num_samples += 1

    # Compute average metrics
    avg_loss = total_loss / num_samples
    avg_iou = total_iou / num_samples

    # Print final metrics
    print(f"Test Loss: {avg_loss}")
    print(f"Test IoU: {avg_iou}")

    return {"loss": avg_loss, "iou": avg_iou}

if __name__ == "__main__":
    evaluate_model()
