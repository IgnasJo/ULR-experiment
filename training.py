import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from training.feature_extractor import RADIOFeatureExtractor
from abl.abl import ABL   # wherever you placed it

from modeling.deeplab import DeepLab
from utils2.loss import SegmentationLosses
from utils2.lr_scheduler import LR_Scheduler
from esrgan import Generator, Discriminator, disc_config
from training.dataloder import train_loader 
from config import training_config, format_config, get_checkpoint_path

def apply_spectral_norm(module):
    """Recursively applies spectral normalization to Conv2d and Linear layers."""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.utils.spectral_norm(module)


def load_pretrained_discriminator_weights(discriminator, pretrained_path, num_classes, device='cuda'):
    """
    Load Phase 1 pretrained discriminator (3-channel input) into Phase 2 discriminator 
    (3 + num_classes channel input) with smart weight initialization.
    
    Strategy:
    - Copy RGB channel weights (first 3 channels) from pretrained model
    - Zero-initialize weights for new segmentation mask channels
    - This ensures the discriminator initially ignores segmentation masks,
      behaving exactly like pretrained model at step 0
    
    Args:
        discriminator: Phase 2 Discriminator model (in_channels = 3 + num_classes)
        pretrained_path: Path to pretrained_discriminator.pth from Phase 1
        num_classes: Number of segmentation classes
        device: Device to load weights onto
    
    Returns:
        discriminator: Model with loaded weights
    """
    if not os.path.exists(pretrained_path):
        print(f"[Warning] Pretrained discriminator not found at: {pretrained_path}")
        print("[Warning] Training discriminator from scratch...")
        return discriminator
    
    print(f"[Joint] Loading pretrained discriminator from: {pretrained_path}")
    
    # Load pretrained state dict (3-channel input)
    pretrained_state = torch.load(pretrained_path, map_location=device)
    
    # Get current model state dict
    model_state = discriminator.state_dict()
    
    # The first conv layer key - check for both regular and spectral norm wrapped versions
    # Regular: 'blocks.0.conv.weight'
    # Spectral norm wrapped: 'blocks.0.conv.weight_orig'
    first_layer_weight_key = 'blocks.0.conv.weight'
    first_layer_weight_key_sn = 'blocks.0.conv.weight_orig'  # spectral norm version
    first_layer_bias_key = 'blocks.0.conv.bias'
    
    # Determine which key exists in pretrained weights
    if first_layer_weight_key_sn in pretrained_state:
        # Spectral norm was applied during pretraining
        using_spectral_norm = True
        actual_weight_key = first_layer_weight_key_sn
        print(f"  [Info] Detected spectral norm wrapped weights (using '{actual_weight_key}')")
    elif first_layer_weight_key in pretrained_state:
        # Regular weights without spectral norm
        using_spectral_norm = False
        actual_weight_key = first_layer_weight_key
        print(f"  [Info] Detected regular weights (using '{actual_weight_key}')")
    else:
        print(f"[Error] Could not find first layer weights in pretrained state")
        print(f"  Looked for: '{first_layer_weight_key}' or '{first_layer_weight_key_sn}'")
        print(f"  Available keys: {list(pretrained_state.keys())[:10]}...")
        print("[Warning] Training discriminator from scratch...")
        return discriminator
    
    # Process each key in pretrained state
    new_state = {}
    for key, pretrained_tensor in pretrained_state.items():
        if key == actual_weight_key:
            # Shape: [out_channels, in_channels, kernel_h, kernel_w]
            # Pretrained: [64, 3, 3, 3]
            # Target:     [64, 3+num_classes, 3, 3]
            
            pretrained_shape = pretrained_tensor.shape  # [64, 3, 3, 3]
            
            # Model doesn't have spectral norm yet (applied after loading),
            # so always use 'blocks.0.conv.weight' as the target key
            target_key = first_layer_weight_key  # Always 'blocks.0.conv.weight'
            target_shape = model_state[target_key].shape  # [64, 17, 3, 3] for num_classes=14
            
            out_channels = pretrained_shape[0]
            rgb_channels = pretrained_shape[1]  # 3
            kernel_h, kernel_w = pretrained_shape[2], pretrained_shape[3]
            
            print(f"  First layer shape mismatch: pretrained={list(pretrained_shape)} -> target={list(target_shape)}")
            
            # Create new weight tensor with zeros
            new_weight = torch.zeros(target_shape, dtype=pretrained_tensor.dtype, device=device)
            
            # Copy RGB weights (first 3 channels)
            new_weight[:, :rgb_channels, :, :] = pretrained_tensor
            
            # Remaining channels (segmentation masks) stay zero-initialized
            # This ensures discriminator ignores mask channels initially
            
            print(f"  Copied RGB weights (channels 0-2), zero-initialized mask weights (channels 3-{target_shape[1]-1})")
            new_state[target_key] = new_weight
            
        else:
            # All other layers: direct copy (shapes should match)
            # Handle spectral norm key mapping: weight_orig -> weight
            # because model doesn't have spectral norm yet (applied after loading)
            mapped_key = key
            if '_orig' in key:
                mapped_key = key.replace('_orig', '')  # weight_orig -> weight
            
            # Skip spectral norm internal buffers (weight_u, weight_v) - not needed before SN is applied
            if key.endswith('_u') or key.endswith('_v'):
                continue
            
            if mapped_key in model_state:
                if pretrained_tensor.shape == model_state[mapped_key].shape:
                    new_state[mapped_key] = pretrained_tensor
                else:
                    print(f"  [Warning] Shape mismatch for '{mapped_key}': {pretrained_tensor.shape} vs {model_state[mapped_key].shape}, skipping")
            else:
                print(f"  [Warning] Key '{mapped_key}' not found in model, skipping")
    
    # Load the processed state dict
    # Use strict=False to allow for any missing keys (shouldn't happen, but safe)
    missing_keys, unexpected_keys = discriminator.load_state_dict(new_state, strict=False)
    
    if missing_keys:
        print(f"  [Info] Keys not loaded from pretrained (using default init): {missing_keys}")
    if unexpected_keys:
        print(f"  [Warning] Unexpected keys in pretrained: {unexpected_keys}")
    
    print("[Joint] Pretrained discriminator weights loaded successfully!")
    print(f"  -> RGB channels: copied from pretrained")
    print(f"  -> Mask channels: zero-initialized (discriminator ignores masks at step 0)")
    
    return discriminator

def to_one_hot(tensor, num_classes):
    """
    Converts label tensor [B, H, W] to one-hot tensor [B, C, H, W]
    Used for concatenating mask with image for Discriminator input (Eq 10).
    """
    tensor = tensor.unsqueeze(1) # [B, 1, H, W]
    one_hot = torch.zeros(tensor.size(0), num_classes, tensor.size(2), tensor.size(3), device=tensor.device)
    one_hot.scatter_(1, tensor, 1.0)
    return one_hot

def feature_loss_calc(f_real, f_fake):
    """
    Calculates L_fea = L1 + L_cos (Eq 4, 5, 6)
    With numerical stability to prevent NaN
    """
    eps = 1e-8
    
    # L1 Component
    l1 = F.l1_loss(f_fake, f_real)
    
    # Cosine Component
    # Flatten features to [B, D] for cosine similarity if they are 4D, 
    # or handle 3D [B, Tokens, Channels] from ViT/RADIO
    if f_real.dim() > 2:
        f_real_flat = f_real.view(f_real.size(0), -1)
        f_fake_flat = f_fake.view(f_fake.size(0), -1)
    else:
        f_real_flat = f_real
        f_fake_flat = f_fake
    
    # Normalize to prevent numerical instability
    f_real_norm = F.normalize(f_real_flat, p=2, dim=1, eps=eps)
    f_fake_norm = F.normalize(f_fake_flat, p=2, dim=1, eps=eps)
    
    cos_sim = (f_real_norm * f_fake_norm).sum(dim=1).mean()
    l_cos = 1 - cos_sim
    
    # Clamp to prevent extreme values
    l_cos = torch.clamp(l_cos, min=0.0, max=2.0)
    
    return l1 + l_cos


def train_joint(pretrained_generator_path=None, pretrained_discriminator_path=None):
    """
    Joint training of Generator and Segmentor.
    
    Args:
        pretrained_generator_path: Path to pretrained generator weights (optional)
        pretrained_discriminator_path: Path to pretrained discriminator weights (optional)
            Note: Phase 1 discriminator has 3 input channels (RGB only).
            Phase 2 discriminator has 3 + num_classes channels (RGB + masks).
            The loading function handles this mismatch automatically.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Joint Training on {device}...")

    # 1. Initialize Models
    
    # A. Generator (Super Resolution)
    generator = Generator().to(device)
    
    # Load pretrained weights if provided
    if pretrained_generator_path and os.path.exists(pretrained_generator_path):
        print(f"[Joint] Loading pretrained generator from: {pretrained_generator_path}")
        generator.load_state_dict(torch.load(pretrained_generator_path, map_location=device))
        print("[Joint] Pretrained weights loaded successfully!")
    elif pretrained_generator_path:
        print(f"[Warning] Pretrained weights not found at: {pretrained_generator_path}")
        print("[Warning] Training generator from scratch...")

    # B. Discriminator 
    # as per Eq (10): z = concat(I, S)
    # Calculate input channels: 3 (RGB) + num_classes (Mask Channels)
    # Example: 3 + 14 = 17 channels
    disc_in_channels = 3 + training_config.num_classes
    
    discriminator = Discriminator(in_channels=disc_in_channels, disc_config=disc_config).to(device)

    # Load pretrained discriminator weights if provided
    # Handles shape mismatch: Phase 1 (3ch) -> Phase 2 (3 + num_classes ch)
    if pretrained_discriminator_path:
        discriminator = load_pretrained_discriminator_weights(
            discriminator, 
            pretrained_discriminator_path, 
            training_config.num_classes, 
            device
        )

    # IMPLEMENTATION: Apply Spectral Normalization to SAD weights
    # This bounds the Lipschitz constant to stabilize training
    discriminator.apply(apply_spectral_norm)
    
    # C. Feature Extractor
    feature_extractor = RADIOFeatureExtractor().to(device)
    
    # D. Segmentation Model
    segmentor = DeepLab(num_classes=training_config.num_classes,
                        backbone='resnet',
                        output_stride=16,
                        sync_bn=False,
                        freeze_bn=True).to(device)

    # 2. Optimizers
    opt_g = optim.Adam(generator.parameters(), lr=training_config.generator_lr, betas=(0.9, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=training_config.discriminator_lr, betas=(0.9, 0.999))
    
    train_params = [{'params': segmentor.get_1x_lr_params(), 'lr': training_config.segmentor_lr},
                    {'params': segmentor.get_10x_lr_params(), 'lr': training_config.segmentor_lr * 10}]
    opt_seg = optim.SGD(train_params, momentum=0.9, weight_decay=5e-4, nesterov=True)

    # 3. Loss Functions
    criterion_l2 = nn.MSELoss()        # Eq (2)


    criterion_abl = ABL(
        isdetach=True,
        max_N_ratio=1/100,
        ignore_label=255,
        label_smoothing=0.0,     # IMPORTANT: disable smoothing for stability
        max_clip_dist=20.0
    ).to(device)

    criterion_gan = nn.BCEWithLogitsLoss() # Eq (7, 8)
    criterion_ce = SegmentationLosses(weight=None, cuda=torch.cuda.is_available()).build_loss(mode='ce') # Eq (3)

    # 4. Scheduler
    scheduler = LR_Scheduler(mode=training_config.lr_scheduler, 
                             base_lr=training_config.segmentor_lr, 
                             num_epochs=training_config.num_epochs, 
                             iters_per_epoch=len(train_loader))

    # 5. Training Loop
    best_pred = 0.0

    def freeze_bn_layers(model):
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()

    for epoch in range(training_config.num_epochs):
        generator.train()
        discriminator.train()
        segmentor.train()
        freeze_bn_layers(segmentor)
        
        tbar = tqdm(train_loader)
        
        for i, (images, masks) in enumerate(tbar):
            scheduler(opt_seg, i, epoch, best_pred)

            # Data Prep
            real_img = images.to(device)  # I_gt
            masks_gt = masks.to(device)   # S_gt (Indices)
            
            # Create ULR Input
            lr_img = F.interpolate(real_img, size=(format_config.ultra_low_resolution, format_config.ultra_low_resolution), mode='bicubic', align_corners=False)
            lr_img = F.interpolate(lr_img, size=(format_config.low_resolution, format_config.low_resolution), mode='bicubic', align_corners=False)

            # ===================================================================================
            #  STEP 1: GENERATE & SEGMENT (Forward Pass)
            # ===================================================================================
            
            # 1. Generate SR Image (I_sr)
            fake_sr = generator(lr_img) 
            
            # 2. Segment SR Image (S_pred)
            # segmentor returns logits
            seg_logits = segmentor(fake_sr) 
            # Apply softmax to get probability map for Discriminator concatenation
            seg_probs = torch.softmax(seg_logits, dim=1) 
            
            # 3. Prepare Joint Inputs for Discriminator (Eq 10)
            # z_real = concat(I_gt, S_gt_onehot)
            masks_onehot = to_one_hot(masks_gt, training_config.num_classes)
            z_real = torch.cat([real_img, masks_onehot], dim=1)
            
            # z_fake = concat(I_sr, S_pred_probs)
            z_fake = torch.cat([fake_sr.detach(), seg_probs.detach()], dim=1) # Detach for D training

            # Add instance noise to discriminator inputs (decays over training)
            # This prevents the discriminator from memorizing and overfitting
            noise_std = max(0.1 * (1 - epoch / training_config.num_epochs), 0.02)
            z_real_noisy = z_real + noise_std * torch.randn_like(z_real)
            z_fake_noisy = z_fake + noise_std * torch.randn_like(z_fake)

            # ===================================================================================
            #  STEP 2: TRAIN DISCRIMINATOR (Eq 7)
            # ===================================================================================
            opt_d.zero_grad()
            
            # Real Branch - One-sided label smoothing (0.9 instead of 1.0)
            # This prevents the discriminator from becoming overconfident
            pred_d_real = discriminator(z_real_noisy)
            real_labels = torch.ones_like(pred_d_real) * training_config.label_smoothing_real
            loss_d_real = criterion_gan(pred_d_real, real_labels)
            
            # Fake Branch - Keep labels at 0.0 (no smoothing for fake)
            pred_d_fake = discriminator(z_fake_noisy)
            loss_d_fake = criterion_gan(pred_d_fake, torch.zeros_like(pred_d_fake))
            
            loss_d = loss_d_real + loss_d_fake
            
            # Only update discriminator if it's not already too strong
            # Skip D update if it's already winning too hard (L_D < 0.1)
            should_update_d = loss_d.item() > 0.1
            
            # Check for NaN before backward
            if not (torch.isnan(loss_d) or torch.isinf(loss_d)) and should_update_d:
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                opt_d.step()
            elif not should_update_d:
                pass  # Skip D update silently when D is too strong
            else:
                print(f"[Warning] NaN/Inf in discriminator loss, skipping D update")
                loss_d = torch.tensor(0.0, device=device)  # For logging

            # ===================================================================================
            #  STEP 3: TRAIN GENERATOR & SEGMENTOR JOINTLY (Eq 1)
            # ===================================================================================
            opt_g.zero_grad()
            opt_seg.zero_grad()
            
            # NOTE: We must re-create z_fake WITHOUT detach to allow gradients to flow back to G and Seg
            z_fake_grad = torch.cat([fake_sr, torch.softmax(seg_logits, dim=1)], dim=1)
            
            # A. Calculate Generator Losses
            
            # 1. Pixel Loss (L2) - Eq (2)
            loss_2 = criterion_l2(fake_sr, real_img)
            
            # 2. Feature Loss (L1 + Cos) - Eq (4)
            # ---------------------------------------------------------
            # RADIO requires size divisible by 14 (e.g., 378).
            # Generator outputs 384 (divisible by 4).
            # We bridge this gap by resizing just for this specific loss calculation.
            # ---------------------------------------------------------
            radio_size = (378, 378)
            
            # Resize both Real and Fake images to 378 just for the feature extractor
            # This does NOT affect the generator's training resolution (which stays 384)
            real_for_radio = F.interpolate(real_img, size=radio_size, mode='bilinear', align_corners=False)
            fake_for_radio = F.interpolate(fake_sr, size=radio_size, mode='bilinear', align_corners=False)
            # Real features: no gradients needed (target)
            # Fake features: gradients needed (to train generator)
            real_feat = feature_extractor(real_for_radio, no_grad=True)
            fake_feat = feature_extractor(fake_for_radio, no_grad=False)
            loss_fea = feature_loss_calc(real_feat.detach(), fake_feat)
            
            # 3. Adversarial Loss - Eq (11)
            pred_d_fake_g = discriminator(z_fake_grad)
            loss_adv = criterion_gan(pred_d_fake_g, torch.ones_like(pred_d_fake_g))
            
            # B. Calculate Segmentation Loss (L_ce) - Eq (3)
            loss_ce = criterion_ce(seg_logits, masks_gt)

            loss_abl = criterion_abl(seg_logits, masks_gt)

            # ABL may return None if no boundaries are detected
            if loss_abl is None:
                loss_abl = torch.tensor(0.0, device=device)


            # C. Total Loss - Eq (1)
            # L_tot = (1 - alpha) * (lam1*L2 + lam2*L_fea + lam3*L_adv) + alpha * L_ce
            
            gen_part = (training_config.lambda_1 * loss_2) + \
                       (training_config.lambda_2 * loss_fea) + \
                       (training_config.lambda_3 * loss_adv)
            
            total_loss = ((1 - training_config.alpha) * gen_part) + (training_config.alpha * loss_ce) + (training_config.lambda_abl * loss_abl)
            
            # Check for NaN before backward pass
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"[Warning] NaN/Inf detected in total_loss, skipping batch")
                print(f"  L_2: {loss_2.item()}, L_fea: {loss_fea.item()}, L_adv: {loss_adv.item()}, L_ce: {loss_ce.item()}, L_abl: {loss_abl.item()}")
                opt_g.zero_grad()
                opt_seg.zero_grad()
                continue
            
            total_loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(segmentor.parameters(), max_norm=1.0)
            
            opt_g.step()
            opt_seg.step()

            # Display
            current_lr = opt_seg.param_groups[0]['lr']
            tbar.set_description(f"Ep {epoch+1} | L_D: {loss_d.item():.3f} | L_2: {loss_2.item():.3f} | L_CE: {loss_ce.item():.3f} | L_Adv: {loss_adv.item():.3f} | L_abl: {loss_abl.item():.3f}")

        # Checkpointing - save as single file compatible with inference.py load_models()
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'gen_state_dict': generator.state_dict(),
                'seg_state_dict': segmentor.state_dict(),
                'epoch': epoch + 1
            }
            ckpt_path = get_checkpoint_path(f"joint_checkpoint_ep{epoch+1}.pth")
            torch.save(checkpoint, ckpt_path)
            print(f"[Joint] Checkpoint saved to: {ckpt_path}")
    
    # Save final checkpoint
    final_checkpoint = {
        'gen_state_dict': generator.state_dict(),
        'seg_state_dict': segmentor.state_dict(),
        'epoch': training_config.num_epochs
    }
    final_path = get_checkpoint_path("joint_checkpoint_final.pth")
    torch.save(final_checkpoint, final_path)
    print(f"[Joint] Training complete. Final checkpoint saved to: {final_path}")

if __name__ == "__main__":
    train_joint()