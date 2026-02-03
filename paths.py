"""
Path utilities for checkpoint management.
Reads from config.checkpoint_config - no state of its own.

Usage:
    from paths import get_checkpoint_path, get_checkpoint_name
    
    # These read from config.checkpoint_config.base_dir
    path = get_checkpoint_path("model.pth")  
    name = get_checkpoint_name("joint", is_final=True)
"""

import os
from config import checkpoint_config


def get_checkpoint_path(filename: str) -> str:
    """
    Get full path for a checkpoint file.
    
    Args:
        filename: Name of the checkpoint file (e.g., 'joint_checkpoint_final.pth')
        
    Returns:
        Full path like 'checkpoints/joint_checkpoint_final.pth'
    """
    os.makedirs(checkpoint_config.base_dir, exist_ok=True)
    return os.path.join(checkpoint_config.base_dir, filename)


def get_checkpoint_name(model_type: str, epoch: int = None, is_final: bool = False, 
                        is_best: bool = False, suffix: str = None) -> str:
    """
    Generate a standardized checkpoint filename.
    
    Args:
        model_type: Type of model ('joint', 'generator', 'discriminator', 
                    'pretrained_generator', 'pretrained_discriminator')
        epoch: Current epoch number (optional)
        is_final: Whether this is the final checkpoint
        is_best: Whether this is the best checkpoint (by validation metric)
        suffix: Optional additional suffix string
        
    Returns:
        Checkpoint filename like 'joint_checkpoint_ep30.pth'
        
    Examples:
        >>> get_checkpoint_name('joint', epoch=30)
        'joint_checkpoint_ep30.pth'
        >>> get_checkpoint_name('joint', is_final=True)
        'joint_checkpoint_final.pth'
        >>> get_checkpoint_name('pretrained_generator')
        'pretrained_generator.pth'
    """
    # Handle pretrained models (no 'checkpoint' in name)
    if model_type.startswith('pretrained_'):
        base = model_type
    else:
        base = f"{model_type}_checkpoint"
    
    # Build name parts
    parts = [base]
    
    if is_best:
        parts.append("best")
    elif is_final:
        parts.append("final")
    elif epoch is not None:
        parts.append(f"ep{epoch}")
    
    if suffix:
        parts.append(suffix)
    
    # Join with underscores and add extension
    if len(parts) == 1:
        return f"{parts[0]}.pth"
    else:
        return f"{'_'.join(parts)}.pth"


# Convenience functions using standard filenames from config
def get_joint_checkpoint_path() -> str:
    """Get path to the joint (final) checkpoint."""
    return get_checkpoint_path(checkpoint_config.joint_filename)


def get_pretrained_generator_path() -> str:
    """Get path to the pretrained generator."""
    return get_checkpoint_path(checkpoint_config.pretrained_gen_filename)


def get_pretrained_discriminator_path() -> str:
    """Get path to the pretrained discriminator."""
    return get_checkpoint_path(checkpoint_config.pretrained_disc_filename)


def get_eval_checkpoint_path() -> str:
    """Get path to the evaluation state checkpoint."""
    return get_checkpoint_path(checkpoint_config.eval_checkpoint_filename)
