import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
from config import training_config


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None, compute_distance_maps=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        
        # Determine if we need distance maps based on ABL flag
        if compute_distance_maps is None:
            self.compute_distance_maps = getattr(training_config, 'use_abl_loss', False)
        else:
            self.compute_distance_maps = compute_distance_maps
        
        # Initialize ABL instance for distance map computation if needed
        self.abl_helper = None
        if self.compute_distance_maps:
            from abl.abl import ABL
            # Create lightweight ABL instance just for distance map computation
            # We don't need the full loss criterion, just the helper methods
            self.abl_helper = ABL(ignore_label=255)
        
        self.images = []
        self.masks = []

        # 1. Create a dictionary of the masks for O(1) lookup
        # Key: filename without extension ('image_01'), Value: full filename ('image_01.png')
        mask_map = {}
        for f in os.listdir(mask_dir):
            if f.endswith(('.png', '.jpg', '.jpeg')):
                stem = os.path.splitext(f)[0]
                mask_map[stem] = f

        # 2. Iterate through images and find the matching mask
        # We sort here just so the dataset order is deterministic
        for img_name in sorted(os.listdir(image_dir)):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                img_stem = os.path.splitext(img_name)[0]
                
                if img_stem in mask_map:
                    # Success: We found a mask with the same name (ignoring extension)
                    self.images.append(img_name)
                    self.masks.append(mask_map[img_stem])
                else:
                    # Optional: specific warning helps you debug your data
                    print(f"Warning: Image '{img_name}' ignored (no matching mask found in {mask_dir})")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") 

        if self.transform:
            image = self.transform(image)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)

        # Conditionally compute ABL distance maps using ABL class methods
        if self.compute_distance_maps and self.abl_helper is not None:
            # Convert mask to tensor format expected by ABL methods
            # mask is [H, W], we need [1, H, W] for batch processing
            mask_tensor = mask.unsqueeze(0)  # [H, W] -> [1, H, W]
            
            # Use ABL's gt2boundary method 
            # Note: gt2boundary expects ignore_label parameter, use the same as ABL instance
            gt_boundary = self.abl_helper.gt2boundary(mask_tensor, ignore_label=self.abl_helper.ignore_label)
            
            # Use ABL's get_dist_maps method to get distance maps
            dist_maps = self.abl_helper.get_dist_maps(gt_boundary)
            
            # dist_maps should be [1, H, W], but squeeze to [H, W] to match expected format
            # The training loop will add batch dimension when batching multiple samples
            dist_maps = dist_maps.squeeze(0)  # [1, H, W] -> [H, W]
            
            return image, mask, dist_maps
        else:
            return image, mask


class SRPretrainDataset(Dataset):
    def __init__(self, hr_image_dir, hr_transform, degradation_transform):
        """
        Args:
            hr_image_dir (str): Path to folder containing High-Res images.
            hr_transform (callable): REQUIRED. Transforms for HR image (Crop + ToTensor).
            degradation_transform (callable): REQUIRED. Pipeline to create LR from HR.
        """
        self.hr_image_dir = hr_image_dir
        self.hr_transform = hr_transform
        self.degradation_transform = degradation_transform
        
        self.images = []
        valid_extensions = ('.png', '.jpg', '.jpeg')
        
        # Load file list
        for f in sorted(os.listdir(hr_image_dir)):
            if f.lower().endswith(valid_extensions):
                self.images.append(f)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.hr_image_dir, img_name)
        
        # 1. Load HR Image
        hr_image = Image.open(img_path).convert("RGB")

        # 2. Apply HR Transforms (Must include ToTensor)
        hr_tensor = self.hr_transform(hr_image)

        # 3. Apply Degradation (HR Tensor -> LR Tensor)
        lr_tensor = self.degradation_transform(hr_tensor)

        return lr_tensor, hr_tensor


class EvaluationDataset(Dataset):
    """
    Dataset for evaluation that returns:
    - LR input tensor (degraded from HR)
    - GT mask tensor
    - HR image tensor (for SR quality metrics: PSNR, SSIM, LPIPS, FID)
    - filename (for saving results)
    """
    def __init__(self, test_dir, gt_dir, lr_transform, mask_transform, hr_transform=None):
        """
        Args:
            test_dir (str): Path to test images (HR images to be degraded)
            gt_dir (str): Path to ground truth segmentation masks
            lr_transform (callable): Transform to create LR input from HR image
            mask_transform (callable): Transform for GT masks
            hr_transform (callable, optional): Transform for HR ground truth
                (CenterCrop + ToTensor). If None, HR tensor is not returned.
        """
        self.test_dir = test_dir
        self.gt_dir = gt_dir
        self.lr_transform = lr_transform
        self.mask_transform = mask_transform
        self.hr_transform = hr_transform
        
        self.images = []
        self.masks = []
        self.filenames = []
        valid_extensions = ('.png', '.jpg', '.jpeg')
        
        # Build mask lookup
        mask_map = {}
        for f in os.listdir(gt_dir):
            if f.lower().endswith(valid_extensions):
                stem = os.path.splitext(f)[0]
                mask_map[stem] = f
        
        # Match images with masks
        for img_name in sorted(os.listdir(test_dir)):
            if img_name.lower().endswith(valid_extensions):
                img_stem = os.path.splitext(img_name)[0]
                
                if img_stem in mask_map:
                    self.images.append(img_name)
                    self.masks.append(mask_map[img_stem])
                    self.filenames.append(img_name)
                else:
                    print(f"Warning: Image '{img_name}' ignored (no matching GT mask)")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = self.masks[idx]
        
        img_path = os.path.join(self.test_dir, img_name)
        mask_path = os.path.join(self.gt_dir, mask_name)
        
        # Load and transform image
        image = Image.open(img_path).convert("RGB")
        lr_tensor = self.lr_transform(image)
        
        # Load and transform mask
        mask = Image.open(mask_path).convert("L")
        mask_tensor = self.mask_transform(mask)
        
        # Optionally return HR ground truth for SR metrics
        if self.hr_transform is not None:
            hr_image = Image.open(img_path).convert("RGB")
            hr_tensor = self.hr_transform(hr_image)
            return lr_tensor, mask_tensor, hr_tensor, img_name
        
        return lr_tensor, mask_tensor, img_name