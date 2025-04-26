import wandb
import torch
import pytorch_lightning as pl
import os

def log_debug_samples(images, masks, outputs, max_samples=3, prefix=""):
    """
    Logs debug samples (input images, ground truth masks, and predictions) to wandb.

    Args:
        images (torch.Tensor): Batch of input images.
        masks (torch.Tensor): Batch of ground truth masks.
        outputs (torch.Tensor): Batch of model predictions.
        max_samples (int): Maximum number of samples to log.
        prefix (str): Prefix for log keys to differentiate between train and val logs.
    """
    # Convert tensors to CPU and detach for logging
    images_np = images.cpu().detach().numpy().transpose(0, 2, 3, 1)  # Convert to HWC format
    masks_np = masks.cpu().detach().numpy()
    preds_np = outputs.cpu().detach().numpy() > 0.5  # Threshold predictions (sigmoid already applied)

    # Log the first `max_samples` samples
    for i in range(min(max_samples, len(images_np))):
        wandb.log({
            f"{prefix}sample_{i}/input_image": wandb.Image(images_np[i], caption="Input"),
            f"{prefix}sample_{i}/ground_truth": wandb.Image(masks_np[i], caption="Ground Truth"),
            f"{prefix}sample_{i}/prediction": wandb.Image(preds_np[i], caption="Prediction"),
        })

def setup_wandb(config):
    """
    Sets up the WandbLogger based on the configuration.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        pl.loggers.WandbLogger or None: WandbLogger instance if enabled, otherwise None.
    """
    if config["logging"]["use_wandb"]:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb_mode = config["logging"].get("wandb_mode", "online")  # Default to "online"
        wandb_logger_args = {
            "project": config["logging"]["wandb_project"],
            "config": config,
            "mode": wandb_mode,  # Add mode to WandbLogger arguments
        }
        if "wandb_experiment" in config["logging"] and config["logging"]["wandb_experiment"]:
            wandb_logger_args["name"] = config["logging"]["wandb_experiment"]
        return pl.loggers.WandbLogger(**wandb_logger_args)
    return None

class DebugSampleLogger(pl.Callback):
    """
    A PyTorch Lightning callback to log debug samples (input images, ground truth masks, and predictions)
    during the validation phase, supporting multiple validation datasets.
    """
    def __init__(self, use_wandb=True, max_samples=8, frequency=1):
        super().__init__()
        self.use_wandb = use_wandb
        self.max_samples = max_samples
        self.frequency = frequency  # Add frequency parameter
        self.logged_samples = []  # Store samples for logging at epoch end
        self.epoch_counter = 0  # Track epochs for frequency

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Log the first batch's samples during training
        if self.use_wandb and batch_idx == 0:
            images, masks = batch
            # print(images.mean(), images.std())  # Debugging line to check image stats
            outputs = torch.sigmoid(pl_module(images))  # Apply sigmoid activation
            log_debug_samples(images, masks, outputs, max_samples=self.max_samples, prefix="train_")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Collect the first sample from each batch, up to max_samples batches
        if self.use_wandb and len(self.logged_samples) < self.max_samples:
            images, masks = batch
            # print(images.mean(), images.std())  # Debugging line to check image stats
            outputs = torch.sigmoid(pl_module(images))  # Apply sigmoid activation
            self.logged_samples.append((images[0:1], masks[0:1], outputs[0:1], dataloader_idx))  # Include dataloader_idx

    def on_validation_epoch_end(self, trainer, pl_module):
        # Log samples only if the epoch matches the frequency
        self.epoch_counter += 1
        if self.use_wandb and self.epoch_counter % self.frequency == 0 and self.logged_samples:
            # Group samples by dataloader_idx
            grouped_samples = {}
            for sample in self.logged_samples:
                dataloader_idx = sample[3]
                if dataloader_idx not in grouped_samples:
                    grouped_samples[dataloader_idx] = []
                grouped_samples[dataloader_idx].append(sample[:3])  # Exclude dataloader_idx

            # Log samples for each validation dataset
            for dataloader_idx, samples in grouped_samples.items():
                images = torch.cat([sample[0] for sample in samples], dim=0)
                masks = torch.cat([sample[1] for sample in samples], dim=0)
                outputs = torch.cat([sample[2] for sample in samples], dim=0)
                prefix = f"val_sample/dataloader_idx_{dataloader_idx}/"  # Add dataloader_idx to prefix
                log_debug_samples(images, masks, outputs, max_samples=self.max_samples, prefix=prefix)

            self.logged_samples = []  # Clear after logging

        # Log learning rate at the end of the epoch
        if self.use_wandb:
            current_lr = trainer.optimizers[0].param_groups[0]['lr']
            wandb.log({"learning_rate": current_lr, "epoch": self.epoch_counter})
