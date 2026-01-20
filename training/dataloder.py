import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
from config import format_config, training_config, pretraining_config

from training.dataset import SegmentationDataset, SRPretrainDataset, EvaluationDataset

downsample_transform = transforms.Compose([
    transforms.Resize(
        (format_config.ultra_low_resolution,format_config.ultra_low_resolution),
        interpolation=transforms.InterpolationMode.BICUBIC
    ),
    transforms.Resize(
        (format_config.low_resolution, format_config.low_resolution),
        interpolation=transforms.InterpolationMode.BICUBIC
    ),
])

evaluate_transform = transforms.Compose([
    transforms.CenterCrop((format_config.high_resolution, format_config.high_resolution)),
    downsample_transform,
    transforms.ToTensor()
])

degradation_transform = transforms.Compose([
    transforms.CenterCrop((format_config.high_resolution, format_config.high_resolution)),
    downsample_transform
])

# Define transformations
# RGB Images: Resize -> Tensor -> Normalize
train_transform = transforms.Compose([
    transforms.CenterCrop((format_config.high_resolution, format_config.high_resolution)),
    transforms.ToTensor()
])

def to_long_tensor(x):
    """Helper function to convert input to a Long Tensor for masks."""
    return torch.as_tensor(np.array(x), dtype=torch.long)

# Masks: Resize -> Convert to LongTensor (Keep integer class values)
# We use Nearest Neighbor interpolation for masks to avoid creating new "decimal" classes
mask_transform = transforms.Compose([
    transforms.CenterCrop((format_config.high_resolution, format_config.high_resolution)),
    # using because lambda functions are not serializable
    transforms.Lambda(to_long_tensor)
])


# Instantiate the Dataset
train_dataset = SegmentationDataset(
    image_dir=training_config.image_dir,
    mask_dir=training_config.mask_dir,
    transform=train_transform,
    mask_transform=mask_transform
)

# Instantiate the DataLoader
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=training_config.batch_size,         # Number of samples per batch (Decrease if you hit OOM on GPU)
    shuffle=True,         # Shuffle data every epoch (Important for training)
    num_workers=1,        # Number of CPU subprocesses to load data in parallel
    pin_memory=False,      # Speeds up transfer to GPU (set True if using CUDA)
    drop_last=True        # Drop the last incomplete batch (optional, helps with batchnorm)
)

# 3. Initialize Dataset
pretrain_dataset = SRPretrainDataset(
    hr_image_dir=pretraining_config.hr_image_dir,
    hr_transform=train_transform,
    degradation_transform=degradation_transform
)

# 4. DataLoader
pretrain_loader = DataLoader(
    pretrain_dataset, 
    batch_size=pretraining_config.batch_size, 
    shuffle=True, 
    num_workers=4,
    pin_memory=True
)


def create_eval_loader(test_dir, gt_dir, batch_size=1):
    """
    Create evaluation DataLoader.
    
    Args:
        test_dir: Path to test images
        gt_dir: Path to ground truth masks
        batch_size: Batch size (default 1 for evaluation)
        
    Returns:
        DataLoader for evaluation
    """
    eval_dataset = EvaluationDataset(
        test_dir=test_dir,
        gt_dir=gt_dir,
        lr_transform=evaluate_transform,
        mask_transform=mask_transform
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep order for evaluation
        num_workers=2,
        pin_memory=True
    )
    
    return eval_loader