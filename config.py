"""Global configuration values, various training, evaluation parameters, magic values. Can override with local_config.py"""

import torch
from types import SimpleNamespace


evaluation_config = SimpleNamespace(
  test_dir = r'custom\customULR\custom_rgb',
  test_dir_gt = r'custom\customULR\custom_label',
  checkpoint_path = r'checkpoints\best\joint_checkpoint_best.pth',
  evaluation_dir = 'evaluation_output',
  evaluation_checkpoint_path = r'checkpoints\evaluation_checkpoint'
)

format_config = SimpleNamespace(
  ultra_low_resolution = 16,
  low_resolution = 96,
  img_channels = 3
)
format_config.high_resolution = format_config.low_resolution * 4


training_config = SimpleNamespace(
  num_classes = 14, # Count with default "background" class
  num_epochs = 100,
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
  image_dir = r'datasets\custom_demo\custom_rgb',
  mask_dir = r'datasets\custom_demo\custom_label'
)

pretraining_config = SimpleNamespace(
  num_epochs = 50,
  vgg_weight = 5e-3,
  gan_weight = 1e-2,
  batch_size = 16,
  hr_image_dir=r'datasets\custom_demo\custom_rgb',
  generator_lr = training_config.generator_lr,
  discriminator_lr = training_config.discriminator_lr,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Try to override config with local_config.py
try:
    from local_config import *
except ImportError:
    pass