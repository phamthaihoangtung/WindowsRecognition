import os
import torch
import pytorch_lightning as pl
from dotenv import load_dotenv
from data_loader import SegmentationDataModule, get_transforms
from model import SegmentationModel
from utils import DebugSampleLogger, setup_wandb
import yaml
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint  # Import ModelCheckpoint for saving the best model
from evaluate import evaluate_model  # Import evaluation function
import wandb  # Import wandb for logging evaluation results
from data_loader import SegmentationDataset

def train_model():
    # Set float32 matmul precision for Tensor Cores
    torch.set_float32_matmul_precision('medium')  # Use 'high' for better precision if needed

    # Load environment variables
    load_dotenv()

    # Load configuration
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Initialize wandb logger
    logger = setup_wandb(config)

    # Data module
    data_module = SegmentationDataModule(
        train_images=config["paths"]["train_images"],
        train_masks=config["paths"]["train_masks"],
        val_images=config["paths"]["val_images"],  # Can be a single path or a list
        val_masks=config["paths"]["val_masks"],  # Can be a single path or a list
        train_batch_size=config["hyperparameters"]["train_batch_size"],
        val_batch_size=config["hyperparameters"]["val_batch_size"],
        image_size=config["hyperparameters"]["image_size"],
        classes=config["model"]["classes"],
    )

    # Test dataset
    test_transform = get_transforms(config["hyperparameters"]["image_size"], mode="test")
    test_dataset = SegmentationDataset(
        images_dir=config["paths"]["test_images"],
        masks_dir=config["paths"]["test_masks"],
        transform=test_transform,
        classes=config["model"]["classes"],
    )

    # Model
    model = SegmentationModel(
        model_config=config["model"],
        learning_rate=config["hyperparameters"]["learning_rate"],
        scheduler_config=config["hyperparameters"]["scheduler"],
    )

    # Move model to the appropriate device
    device = torch.device("cuda" if config["trainer"]["accelerator"] == "gpu" and torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Initialize the debug sample logger callback
    debug_sample_logger = DebugSampleLogger(
        use_wandb=config["logging"]["use_wandb"],
        frequency=config["logging"].get("debug_sample_frequency", 2),
    )

    # Determine validation loss name dynamically
    val_loss_name = "val_loss/dataloader_idx_0" if isinstance(config["paths"]["val_images"], list) and len(config["paths"]["val_images"]) > 1 else "val_loss"

    # Initialize the early stopping callback if enabled
    early_stopping = None
    if config["hyperparameters"].get("enable_early_stopping", True):
        early_stopping = EarlyStopping(
            monitor=val_loss_name,
            patience=config["hyperparameters"].get("early_stopping_patience", 5),
            mode="min",
        )

    # Initialize the model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor=val_loss_name,
        dirpath=os.path.dirname(config["paths"]["model_save_path"]),
        filename="best_model",
        save_top_k=1 if early_stopping else -1,  # Save only the best if early stopping is enabled
        mode="min",
        save_last=True,
    )

    # Trainer
    trainer = pl.Trainer(
        accelerator=config["trainer"]["accelerator"],
        devices=config["trainer"]["devices"],
        max_epochs=config["hyperparameters"]["epochs"],
        logger=logger,
        callbacks=[cb for cb in [debug_sample_logger, early_stopping, checkpoint_callback] if cb],
        enable_checkpointing=True,
    )
    trainer.fit(model, data_module)

    # Save final model
    trainer.save_checkpoint(config["paths"]["model_save_path"])

    # Evaluate after training
    print("Evaluating on the test set...")
    metrics = evaluate_model(config, model=model, test_dataset=test_dataset, device=device)  # Pass device

    # Log evaluation results to wandb
    if config["logging"]["use_wandb"]:
        wandb.log({
            "test_loss": metrics["loss"],
            "test_iou": metrics["iou"]
        })

if __name__ == "__main__":
    train_model()
