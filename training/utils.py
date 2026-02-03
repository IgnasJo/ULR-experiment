"""Utility functions for dataset handling and experiment setup."""

import os
import shutil
import random
from typing import Tuple, List


def create_train_test_split(
    source_rgb: str,
    source_label: str,
    train_rgb: str,
    train_label: str,
    test_rgb: str,
    test_label: str,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Create train/test split from source dataset.
    
    Args:
        source_rgb: Path to source RGB images
        source_label: Path to source label images
        train_rgb: Destination path for training RGB images
        train_label: Destination path for training labels
        test_rgb: Destination path for test RGB images
        test_label: Destination path for test labels
        train_ratio: Ratio of data to use for training (default 0.8)
        seed: Random seed for reproducibility (default 42)
    
    Returns:
        Tuple of (train_files, test_files) lists
    """
    # Create directories
    for d in [train_rgb, train_label, test_rgb, test_label]:
        os.makedirs(d, exist_ok=True)
    
    # Get all image files and shuffle
    image_files = sorted([
        f for f in os.listdir(source_rgb) 
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ])
    
    random.seed(seed)
    random.shuffle(image_files)
    
    # Split
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    test_files = image_files[split_idx:]
    
    print(f"Total images: {len(image_files)}")
    print(f"Training set: {len(train_files)} images ({train_ratio*100:.0f}%)")
    print(f"Test set: {len(test_files)} images ({(1-train_ratio)*100:.0f}%)")
    
    # Copy training files
    for f in train_files:
        shutil.copy(os.path.join(source_rgb, f), os.path.join(train_rgb, f))
        label_name = os.path.splitext(f)[0] + '.png'
        label_src = os.path.join(source_label, label_name)
        if os.path.exists(label_src):
            shutil.copy(label_src, os.path.join(train_label, label_name))
    
    # Copy test files
    for f in test_files:
        shutil.copy(os.path.join(source_rgb, f), os.path.join(test_rgb, f))
        label_name = os.path.splitext(f)[0] + '.png'
        label_src = os.path.join(source_label, label_name)
        if os.path.exists(label_src):
            shutil.copy(label_src, os.path.join(test_label, label_name))
    
    print("Data split complete!")
    return train_files, test_files
