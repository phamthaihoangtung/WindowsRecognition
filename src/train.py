import os
import torch
import pytorch_lightning as pl
from dotenv import load_dotenv
from data_loader import SegmentationDataModule
from model import SegmentationModel
from utils import DebugSampleLogger, setup_wandb
import yaml
from pytorch_lightning.callbacks.early_stopping import EarlyStopping  # Import EarlyStopping callback

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
        val_images=config["paths"]["val_images"],
        val_masks=config["paths"]["val_masks"],
        train_batch_size=config["hyperparameters"]["train_batch_size"],  # Separate train batch size
        val_batch_size=config["hyperparameters"]["val_batch_size"],      # Separate val batch size
        image_size=config["hyperparameters"]["image_size"],
        classes=config["model"]["classes"],  # Pass classes from config
    )

    # Model
    model = SegmentationModel(
        model_config=config["model"],
        learning_rate=config["hyperparameters"]["learning_rate"],
        scheduler_config=config["hyperparameters"]["scheduler"],
        use_wandb=config["logging"]["use_wandb"],
    )

    # Initialize the debug sample logger callback
    debug_sample_logger = DebugSampleLogger(
        use_wandb=config["logging"]["use_wandb"],
        frequency=config["logging"].get("debug_sample_frequency", 2),  # Set frequency from config with default 1
    )

    # Initialize the early stopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss",  # Metric to monitor
        patience=config["hyperparameters"].get("early_stopping_patience", 5),  # Number of epochs to wait
        mode="min",  # Stop when the monitored metric stops decreasing
    )

    # Trainer
    trainer = pl.Trainer(
        accelerator=config["trainer"]["accelerator"],  # Read from config
        devices=config["trainer"]["devices"],          # Read from config
        max_epochs=config["hyperparameters"]["epochs"],
        logger=logger,
        callbacks=[debug_sample_logger, early_stopping],  # Add early stopping to callbacks
        enable_checkpointing=True,  # Ensure checkpointing is enabled for early stopping
    )
    trainer.fit(model, data_module)

    # Save model
    trainer.save_checkpoint(config["paths"]["model_save_path"])

if __name__ == "__main__":
    train_model()
