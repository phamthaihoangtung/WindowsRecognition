import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.metrics.functional as smp_metrics_functional
import wandb
from utils import log_debug_samples
from segmentation_models_pytorch.losses import DiceLoss

class SegmentationModel(pl.LightningModule):
    def __init__(self, model_config, learning_rate, scheduler_config, use_wandb):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=model_config["backbone"],
            encoder_weights=model_config["encoder_weights"],
            in_channels=model_config["in_channels"],
            classes=model_config["classes"],
        )
        self.loss_fn = DiceLoss(mode="binary", from_logits=True)
        self.learning_rate = learning_rate
        self.scheduler_config = scheduler_config
        self.iou_threshold = 0.5  # Threshold for IoU computation

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        masks = masks.float()  # Ensure masks are float
        outputs = self(images)
        # print(f"[Training] Images shape: {images.shape}, Masks shape: {masks.shape}, Outputs shape: {outputs.shape}")
        loss = self.loss_fn(outputs, masks)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        masks = masks.float()  # Ensure masks are float
        outputs = self(images)
        # Apply activation function to outputs
        activated_outputs = torch.sigmoid(outputs)
        loss = self.loss_fn(outputs, masks)

        # Compute IoU using smp.metrics.functional
        tp, fp, fn, tn = smp_metrics_functional.get_stats(
            activated_outputs, masks.long(), mode="binary", threshold=self.iou_threshold
        )
        iou = smp_metrics_functional.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

        self.log("val_loss", loss)
        self.log("val_iou", iou)

        return {"val_loss": loss, "val_iou": iou}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.scheduler_config["type"] == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=self.scheduler_config["step_size"], 
                gamma=self.scheduler_config["gamma"]
            )
        elif self.scheduler_config["type"] == "ExponentialLR":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, 
                gamma=self.scheduler_config["gamma"]
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {self.scheduler_config['type']}")
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
