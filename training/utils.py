"""Utility functions for dataset handling and experiment setup."""

import os
import shutil
import random
import re


def extract_number(filename):
    """Extract the last integer value from filename for matching."""
    matches = re.findall(r'\d+', filename)
    return int(matches[-1]) if matches else 0


def create_train_test_split(
    source_rgb,
    source_label,
    train_rgb,
    train_label,
    test_rgb,
    test_label,
    train_ratio=0.8,
    seed=42,
    max_samples=None
):
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
        max_samples: Maximum number of samples to use (default None, uses all)
    
    Returns:
        Tuple of (train_files, test_files) lists
    """
    # Create directories
    for d in [train_rgb, train_label, test_rgb, test_label]:
        os.makedirs(d, exist_ok=True)
    
    # Get all image files and sort by numeric value
    image_files = [
        f for f in os.listdir(source_rgb) 
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ]
    image_files.sort(key=extract_number)
    
    # Build label lookup dictionary once (maps number -> label filename)
    print(f"Building label lookup from {source_label}...")
    label_files = [
        f for f in os.listdir(source_label)
        if f.endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'))
    ]
    label_lookup = {}
    for label_file in label_files:
        number = extract_number(label_file)
        label_lookup[number] = label_file
    print(f"Found {len(label_lookup)} label files")
    
    # Helper function to find label file by matching numeric value
    def find_label_file(image_filename):
        image_number = extract_number(image_filename)
        if image_number in label_lookup:
            return os.path.join(source_label, label_lookup[image_number])
        return None
    
    # Validate that all images have corresponding labels
    print(f"Validating {len(image_files)} images...")
    for f in image_files:
        label_path = find_label_file(f)
        if label_path is None:
            image_number = extract_number(f)
            raise FileNotFoundError(
                f"Label file not found for image '{f}' (number: {image_number}). "
                f"No label file with matching number found in {source_label}"
            )
    
    random.seed(seed)
    random.shuffle(image_files)
    
    # Limit samples if specified
    if max_samples is not None and max_samples > 0:
        image_files = image_files[:max_samples]
    
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
        label_src = find_label_file(f)
        if label_src:
            # Keep original extension of label file
            label_ext = os.path.splitext(label_src)[1]
            label_name = os.path.splitext(f)[0] + label_ext
            shutil.copy(label_src, os.path.join(train_label, label_name))
    
    # Copy test files
    for f in test_files:
        shutil.copy(os.path.join(source_rgb, f), os.path.join(test_rgb, f))
        label_src = find_label_file(f)
        if label_src:
            # Keep original extension of label file
            label_ext = os.path.splitext(label_src)[1]
            label_name = os.path.splitext(f)[0] + label_ext
            shutil.copy(label_src, os.path.join(test_label, label_name))
    
    print("Data split complete!")
    return train_files, test_files
