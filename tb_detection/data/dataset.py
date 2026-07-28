import os
import sys
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add parent directory to path so config can be imported when running script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

def get_train_transforms():
    """
    Returns albumentations transforms for training.
    Includes CLAHE, rotation, flipping, and brightness/contrast adjustments.
    """
    return A.Compose([
        A.Resize(height=Config.IMAGE_SIZE[0], width=Config.IMAGE_SIZE[1]),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),  # Enhances contrast for X-rays
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.95, 1.05),
            rotate=(-10, 10),
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            p=0.5
        ),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_val_transforms():
    """
    Returns deterministic transforms for validation/testing.
    """
    return A.Compose([
        A.Resize(height=Config.IMAGE_SIZE[0], width=Config.IMAGE_SIZE[1]),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),  # Apply same visual prep as training
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

class TBDataset(Dataset):
    """
    Custom PyTorch Dataset for loading Tuberculosis chest X-ray images
    from organized train/val/test directories.
    """
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Load images based on Config classes
        for class_idx, class_name in enumerate(Config.CLASS_NAMES):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for file in os.listdir(class_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, file))
                    self.labels.append(class_idx)
                    
    def __len__(self) -> int:
        return len(self.image_paths)
        
    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Unable to read file: {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, torch.tensor(label, dtype=torch.long)
