import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

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
        self.val_images = val_images
        self.val_masks = val_masks
        self.train_batch_size = train_batch_size  # Separate train batch size
        self.val_batch_size = val_batch_size      # Separate val batch size
        self.image_size = image_size
        self.classes = classes

        self.train_transform = A.Compose([
            A.Resize(self.image_size[0], self.image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        self.val_transform = A.Compose([
            A.Resize(self.image_size[0], self.image_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def setup(self, stage=None):
        self.train_dataset = SegmentationDataset(
            self.train_images, self.train_masks, transform=self.train_transform, classes=self.classes
        )
        self.val_dataset = SegmentationDataset(
            self.val_images, self.val_masks, transform=self.val_transform, classes=self.classes
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.train_batch_size,  # Use train batch size
            shuffle=True, 
            num_workers=8,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.val_batch_size,  # Use val batch size
            shuffle=False, 
            num_workers=8,
            persistent_workers=True
        )
