"""Global configuration values, various training, evaluation parameters, magic values. Can override with local_config.py"""

import torch
import os
from datetime import datetime
from types import SimpleNamespace


# ============================================================================
# Checkpoint Directory Management
# ============================================================================
def get_checkpoint_dir():
    """
    Get the checkpoint directory for today's date.
    Creates folder structure: checkpoints/MM-DD/
    """
    date_str = datetime.now().strftime("%m-%d")
    checkpoint_dir = os.path.join("checkpoints", date_str)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir

def get_checkpoint_path(filename):
    """
    Get full path for a checkpoint file in today's dated folder.
    
    Args:
        filename: Name of the checkpoint file (e.g., 'joint_checkpoint_final.pth')
        
    Returns:
        Full path like 'checkpoints/01-20/joint_checkpoint_final.pth'
    """
    return os.path.join(get_checkpoint_dir(), filename)


# ============================================================================
# Configuration
# ============================================================================
evaluation_config = SimpleNamespace(
  test_dir = r'datasets\custom_demo\rgb',
  test_dir_gt = r'datasets\custom_demo\label',
  checkpoint_path = r'joint_checkpoint_final.pth',  # Will be resolved via get_checkpoint_path()
  evaluation_dir = 'evaluation_output',
)

format_config = SimpleNamespace(
  ultra_low_resolution = 16,
  low_resolution = 96,
  img_channels = 3
)
format_config.high_resolution = format_config.low_resolution * 4


training_config = SimpleNamespace(
  num_classes = 14, # Count with default "background" class
  num_epochs = 2,
  batch_size = 1,
  generator_lr = 1e-4,
  discriminator_lr = 1e-4,
  segmentor_lr = 1e-2,
  lr_scheduler = 'poly',
  LEARNING_RATE = 1e-4,
  # Loss Weights
  alpha = 0.3,     # Balancing parameter between Generative and Segmentation loss
  lambda_1 = 0.5,  # Weight for L2 Pixel Loss
  lambda_2 = 0.01, # Weight for Feature Loss
  lambda_3 = 0.01, # Weight for Adversarial Loss
  lambda_abl = 0.02, # start small (ABL is strong)
  image_dir = r'datasets\custom_demo\rgb',
  mask_dir = r'datasets\custom_demo\label'
)

pretraining_config = SimpleNamespace(
  num_epochs = 2,
  vgg_weight = 5e-3,
  gan_weight = 1e-2,
  batch_size = 16,
  hr_image_dir=r'datasets\custom_demo\rgb',
  generator_lr = training_config.generator_lr,
  discriminator_lr = training_config.discriminator_lr,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Local Configuration Override
# ============================================================================
local_config_path = os.path.join(os.path.dirname(__file__), 'local_config.py')

if os.path.exists(local_config_path):
    print(f"Loading local configuration from {local_config_path}...")
    with open(local_config_path) as f:
        # Execute the file's content inside the current namespace
        exec(f.read())