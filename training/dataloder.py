import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
from config import format_config, training_config, pretraining_config, perf_config

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
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

degradation_transform = transforms.Compose([
    transforms.CenterCrop((format_config.high_resolution, format_config.high_resolution)),
    downsample_transform
])

# Define transformations
# RGB Images: Crop -> Tensor -> Normalize to [-1, 1] (mean=0.5, std=0.5)
# Matches the inference postprocess which denormalises via tensor * 0.5 + 0.5
train_transform = transforms.Compose([
    transforms.CenterCrop((format_config.high_resolution, format_config.high_resolution)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
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


def create_train_loader():
    """
    Create training DataLoader for joint SR + segmentation training.
    Call this function only when training (not in eval-only mode).
    
    Returns:
        DataLoader for training
    """
    train_dataset = SegmentationDataset(
        image_dir=training_config.image_dir,
        mask_dir=training_config.mask_dir,
        transform=train_transform,
        mask_transform=mask_transform,
        compute_distance_maps=training_config.use_abl_loss  # Only compute distance maps if ABL is enabled
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=perf_config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=perf_config.num_workers > 0,
        prefetch_factor=2 if perf_config.num_workers > 0 else None,
        drop_last=True
    )
    return train_loader


def create_pretrain_loader():
    """
    Create pretraining DataLoader for SR pretraining phase.
    Call this function only when pretraining (not in eval-only mode).
    
    Returns:
        DataLoader for pretraining
    """
    pretrain_dataset = SRPretrainDataset(
        hr_image_dir=pretraining_config.hr_image_dir,
        hr_transform=train_transform,
        degradation_transform=degradation_transform
    )

    pretrain_loader = DataLoader(
        pretrain_dataset, 
        batch_size=pretraining_config.batch_size, 
        shuffle=True, 
        num_workers=perf_config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=perf_config.num_workers > 0,
        prefetch_factor=2 if perf_config.num_workers > 0 else None,
    )
    return pretrain_loader


def create_eval_loader(test_dir, gt_dir, batch_size=1, include_hr=False):
    """
    Create evaluation DataLoader.
    
    Args:
        test_dir: Path to test images
        gt_dir: Path to ground truth masks
        batch_size: Batch size (default 1 for evaluation)
        include_hr: If True, also return HR ground truth tensors
            (needed for SR quality metrics: PSNR, SSIM, LPIPS, FID)
        
    Returns:
        DataLoader for evaluation
    """
    eval_dataset = EvaluationDataset(
        test_dir=test_dir,
        gt_dir=gt_dir,
        lr_transform=evaluate_transform,
        mask_transform=mask_transform,
        hr_transform=train_transform if include_hr else None
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=perf_config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=perf_config.num_workers > 0,
        prefetch_factor=2 if perf_config.num_workers > 0 else None,
    )
    
    return eval_loader