import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

def get_transforms(image_size, mode="train"):
    """
    Returns the appropriate transforms for training, validation, or testing.

    Args:
        image_size (tuple): Target image size as (height, width).
        mode (str): Mode of transformation - "train", "val", or "test".

    Returns:
        albumentations.Compose: Composed transformations.
    """
    if mode == "train":
        return A.Compose([
            A.Resize(1024, 1024),  # Resize to 1024x1024
            A.RandomCrop(image_size[0], image_size[1]),  # Random crop to image_size
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    elif mode in ["val", "test"]:
        return A.Compose([
            A.Resize(1024, 1024),  # Resize to 1024x1024
            A.CenterCrop(image_size[0], image_size[1]),  # Center crop to image_size
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        raise ValueError(f"Unsupported mode: {mode}")

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, classes=1):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.classes = classes
        self.image_files = sorted(os.listdir(images_dir))
        self.mask_files = sorted(os.listdir(masks_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Process mask based on the number of classes
        if self.classes == 1:
            mask = mask[..., None]  # Add channel dimension for binary segmentation
            mask = (mask > 0).astype("float32")  # Ensure mask values are in the 0-1 range
        else:
            mask = cv2.oneHotEncode(mask, self.classes)  # Example for multi-class (adjust as needed)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Ensure mask has shape [C, H, W]
        if mask.ndim == 3 and mask.shape[-1] == 1:  # If mask is [H, W, 1]
            mask = mask.permute(2, 0, 1)  # Change to [C, H, W]

        return image, mask

class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, train_images, train_masks, val_images, val_masks, train_batch_size, val_batch_size, image_size, classes=1):
        super().__init__()
        self.train_images = train_images
        self.train_masks = train_masks
        self.val_images = val_images if isinstance(val_images, list) else [val_images]
        self.val_masks = val_masks if isinstance(val_masks, list) else [val_masks]
        self.train_batch_size = train_batch_size  # Separate train batch size
        self.val_batch_size = val_batch_size      # Separate val batch size
        self.image_size = image_size
        self.classes = classes

        self.train_transform = get_transforms(self.image_size, mode="train")
        self.val_transform = get_transforms(self.image_size, mode="val")

    def setup(self, stage=None):
        self.train_dataset = SegmentationDataset(
            self.train_images, self.train_masks, transform=self.train_transform, classes=self.classes
        )
        self.val_datasets = [
            SegmentationDataset(val_images, val_masks, transform=self.val_transform, classes=self.classes)
            for val_images, val_masks in zip(self.val_images, self.val_masks)
        ]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.train_batch_size,  # Use train batch size
            shuffle=True, 
            num_workers=4,
            persistent_workers=True
        )

    def val_dataloader(self):
        return [
            DataLoader(
                val_dataset,
                batch_size=self.val_batch_size,
                shuffle=False,
                num_workers=4,
                persistent_workers=True,
            )
            for val_dataset in self.val_datasets
        ]
