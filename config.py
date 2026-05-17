"""
Global configuration - Pure data only, no functions.

Architecture:
- model_config: Immutable model architecture (num_classes, channels, resolutions)
- training_config: Training hyperparameters (can be modified per experiment)
- pretraining_config: Pretraining specific parameters
- evaluation_config: Evaluation/inference paths
- checkpoint_config: Checkpoint directory settings

Usage:
    # Option 1: Direct modification (same process)
    import config
    config.checkpoint_config.base_dir = '/path/to/checkpoints'
    
    # Option 2: Environment variables (for subprocesses)
    export ULR_CHECKPOINT_DIR=/path/to/checkpoints
    export ULR_USE_ABL=True
    export ULR_USE_AMP=True
    export ULR_USE_COMPILE=True
"""

import os
import torch
from types import SimpleNamespace

# Absolute path to the experiment directory (where this file lives).
# Used to anchor all default paths so they resolve correctly regardless of CWD.
_EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))


def _rel(*parts: str) -> str:
    """Resolve a path relative to the experiment directory."""
    return os.path.join(_EXPERIMENT_DIR, *parts)


def _env(key: str, default, type_fn=str):
    """Get environment variable with type conversion."""
    val = os.environ.get(key)
    if val is None:
        return default
    if type_fn == bool:
        return val.lower() in ('true', '1', 'yes')
    return type_fn(val)


# ============================================================================
# Model Architecture (IMMUTABLE - shared across all experiments)
# ============================================================================
model_config = SimpleNamespace(
    num_classes = 14,           # Segmentation classes (including background)
    img_channels = 3,           # RGB
    ultra_low_resolution = 16,  # ULR input size
    low_resolution = 96,        # LR intermediate size  
    high_resolution = 384,      # HR output size (low_resolution * 4)
)


# ============================================================================
# Checkpoint Settings
# ============================================================================
checkpoint_config = SimpleNamespace(
    base_dir = _env('ULR_CHECKPOINT_DIR', _rel('checkpoints')),
    joint_filename = "joint_checkpoint_final.pth",
    pretrained_gen_filename = "pretrained_generator.pth",
    pretrained_disc_filename = "pretrained_discriminator.pth",
    eval_checkpoint_filename = "evaluation_checkpoint.pkl",
)


# ============================================================================
# Training Configuration
# ============================================================================
training_config = SimpleNamespace(
    # Epochs
    num_epochs = _env('ULR_TRAIN_EPOCHS', 2, int),
    
    # Batch size
    batch_size = _env('ULR_BATCH_SIZE', 1, int),
    
    # Learning rates
    generator_lr = _env('ULR_GENERATOR_LR', 1e-4, float),
    discriminator_lr = _env('ULR_DISCRIMINATOR_LR', 1e-5, float),
    segmentor_lr = _env('ULR_SEGMENTOR_LR', 1e-2, float),
    lr_scheduler = _env('ULR_LR_SCHEDULER', 'poly'),
    
    # Loss weights (Eq 1: L_tot = (1-α)(λ1*L2 + λ2*L_fea + λ3*L_adv) + α*L_ce + λ_abl*L_abl)
    alpha = _env('ULR_ALPHA', 0.3, float),
    lambda_1 = _env('ULR_LAMBDA_1', 0.5, float),
    lambda_2 = _env('ULR_LAMBDA_2', 0.01, float),
    lambda_3 = _env('ULR_LAMBDA_3', 0.005, float),
    
    # Active Boundary Loss
    use_abl_loss = _env('ULR_USE_ABL', False, bool),
    lambda_abl = _env('ULR_LAMBDA_ABL', 0.02, float),

    # Speed / precision flags
    # use_amp: enables bfloat16 mixed precision (significant speedup on Ampere+/Blackwell)
    use_amp = _env('ULR_USE_AMP', False, bool),
    # use_compile: enables torch.compile for generator and segmentor
    use_compile = _env('ULR_USE_COMPILE', False, bool),
    
    # GAN Stability
    label_smoothing_real = _env('ULR_LABEL_SMOOTHING', 0.9, float),
    d_update_freq = _env('ULR_D_UPDATE_FREQ', 1, int),
    
    # Data paths (override per experiment)
    image_dir = _env('ULR_TRAIN_RGB', _rel('datasets', 'custom_demo', 'rgb')),
    mask_dir = _env('ULR_TRAIN_LABEL', _rel('datasets', 'custom_demo', 'label')),
)


# ============================================================================
# Performance Configuration
# ============================================================================
perf_config = SimpleNamespace(
    # DataLoader workers (set 0 to disable multiprocessing, e.g. for debugging)
    num_workers = _env('ULR_NUM_WORKERS', 4, int),
    # deterministic=True disables cudnn.benchmark for reproducible results
    deterministic = _env('ULR_DETERMINISTIC', False, bool),
)


# ============================================================================
# Pretraining Configuration
# ============================================================================
pretraining_config = SimpleNamespace(
    num_epochs = _env('ULR_PRETRAIN_EPOCHS', 2, int),
    batch_size = _env('ULR_PRETRAIN_BATCH_SIZE', 2, int),
    
    # Loss weights
    vgg_weight = _env('ULR_VGG_WEIGHT', 5e-3, float),
    gan_weight = _env('ULR_GAN_WEIGHT', 1e-2, float),
    
    # Learning rates
    generator_lr = _env('ULR_PRETRAIN_GENERATOR_LR', 1e-4, float),
    discriminator_lr = _env('ULR_PRETRAIN_DISCRIMINATOR_LR', 1e-5, float),
    
    # Data path
    hr_image_dir = _env('ULR_TRAIN_RGB', _rel('datasets', 'custom_demo', 'rgb')),
)


# ============================================================================
# Evaluation Configuration
# ============================================================================
evaluation_config = SimpleNamespace(
    test_dir = _env('ULR_TEST_RGB', _rel('datasets', 'custom_demo', 'rgb')),
    test_dir_gt = _env('ULR_TEST_LABEL', _rel('datasets', 'custom_demo', 'label')),
    output_dir = _env('ULR_EVAL_OUTPUT_DIR', _rel('evaluation_output')),
)


# ============================================================================
# Batch Inference Configuration
# ============================================================================
batch_inference_config = SimpleNamespace(
    rgb_dir = _env('ULR_BATCH_INFERENCE_RGB', None),
    label_dir = _env('ULR_BATCH_INFERENCE_LABEL', None),
)


# ============================================================================
# Device
# ============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# Global performance settings (applied once at import time)
# ============================================================================
# TF32: ~2-3x faster matmuls on Ampere+ GPUs with negligible precision loss
torch.set_float32_matmul_precision('high')

# cuDNN auto-tunes convolution algorithms for fixed input sizes (16→96→384)
# Disable for fully reproducible runs via ULR_DETERMINISTIC=True
torch.backends.cudnn.benchmark = not _env('ULR_DETERMINISTIC', False, bool)
torch.backends.cudnn.deterministic = _env('ULR_DETERMINISTIC', False, bool)


# ============================================================================
# Backward Compatibility Aliases
# ============================================================================
format_config = SimpleNamespace(
    ultra_low_resolution = model_config.ultra_low_resolution,
    low_resolution = model_config.low_resolution,
    high_resolution = model_config.high_resolution,
    img_channels = model_config.img_channels,
)