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
from training.dataloder import create_train_loader
from config import training_config, format_config, model_config
from paths import get_checkpoint_path, get_checkpoint_name
def strip_module_state_dict(sd):
    """Strip 'module.' (DataParallel) and '_orig_mod.' (torch.compile) prefixes from state dict keys."""
    return {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in sd.items()}


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


def train_joint(pretrained_generator_path=None, pretrained_discriminator_path=None, pretrained_checkpoint_path=None):
    """
    Joint training of Generator and Segmentor.
    
    Args:
        pretrained_generator_path: Path to pretrained generator weights (raw state dict, optional)
        pretrained_discriminator_path: Path to pretrained discriminator weights (optional)
            Note: Phase 1 discriminator has 3 input channels (RGB only).
            Phase 2 discriminator has 3 + num_classes channels (RGB + masks).
            The loading function handles this mismatch automatically.
        pretrained_checkpoint_path: Path to a full joint checkpoint dict containing
            'gen_state_dict' and 'seg_state_dict'. When set, both generator and
            segmentor are initialised from the checkpoint (finetune mode).
            Raises FileNotFoundError / ValueError / RuntimeError on any loading failure.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = training_config.use_amp and device.type == 'cuda'
    use_compile = training_config.use_compile
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    print(f"Starting Joint Training on {device}...")
    print(f"[Joint] Mixed precision (AMP): {use_amp} (dtype={amp_dtype})")
    print(f"[Joint] torch.compile: {use_compile}")

    # Pre-load finetune checkpoint state dicts (in-memory) before model init
    _finetune_gen_sd = None
    _finetune_seg_sd = None
    if pretrained_checkpoint_path:
        if not os.path.exists(pretrained_checkpoint_path):
            raise FileNotFoundError(f"[Finetune] Checkpoint not found: {pretrained_checkpoint_path}")
        _ckpt = torch.load(pretrained_checkpoint_path, map_location=device, weights_only=False)
        if not isinstance(_ckpt, dict) or 'gen_state_dict' not in _ckpt or 'seg_state_dict' not in _ckpt:
            raise ValueError(
                f"[Finetune] Invalid checkpoint format. Expected 'gen_state_dict' and 'seg_state_dict', "
                f"got: {list(_ckpt.keys()) if isinstance(_ckpt, dict) else type(_ckpt)}"
            )
        print(f"[Joint] Finetuning from: {pretrained_checkpoint_path}")
        if 'epoch' in _ckpt:
            print(f"  Checkpoint epoch : {_ckpt['epoch']}")
        if 'miou' in _ckpt:
            print(f"  Checkpoint mIoU  : {_ckpt['miou']:.4f}")
        _finetune_gen_sd = strip_module_state_dict(_ckpt['gen_state_dict'])
        _finetune_seg_sd = strip_module_state_dict(_ckpt['seg_state_dict'])
        # Discriminator is always re-initialised fresh when finetuning — pretrained checkpoints
        # don't contain disc_state_dict and we want a clean discriminator for the new dataset.

    # 1. Initialize Models
    
    # A. Generator (Super Resolution)
    generator = Generator().to(device)
    
    # Load pretrained weights — finetune checkpoint takes priority over pretrained_generator_path
    if _finetune_gen_sd is not None:
        missing, unexpected = generator.load_state_dict(_finetune_gen_sd, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"[Finetune] Generator state dict mismatch. Missing={missing}, Unexpected={unexpected}"
            )
        print("[Joint] Generator loaded from finetune checkpoint.")
    elif pretrained_generator_path and os.path.exists(pretrained_generator_path):
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
    disc_in_channels = 3 + model_config.num_classes
    
    discriminator = Discriminator(in_channels=disc_in_channels, disc_config=disc_config).to(device)

    # Load pretrained discriminator weights if provided
    # Handles shape mismatch: Phase 1 (3ch) -> Phase 2 (3 + num_classes ch)
    if pretrained_discriminator_path:
        discriminator = load_pretrained_discriminator_weights(
            discriminator, 
            pretrained_discriminator_path, 
            model_config.num_classes, 
            device
        )
    # IMPLEMENTATION: Apply Spectral Normalization to SAD weights
    # This bounds the Lipschitz constant to stabilize training
    discriminator.apply(apply_spectral_norm)
    
    # C. Feature Extractor
    feature_extractor = RADIOFeatureExtractor().to(device)
    
    # D. Segmentation Model
    segmentor = DeepLab(num_classes=model_config.num_classes,
                        backbone='resnet',
                        output_stride=16,
                        sync_bn=False,
                        freeze_bn=True).to(device)

    # Load segmentor weights from finetune checkpoint if available
    if _finetune_seg_sd is not None:
        missing, unexpected = segmentor.load_state_dict(_finetune_seg_sd, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"[Finetune] Segmentor state dict mismatch. Missing={missing}, Unexpected={unexpected}"
            )
        print("[Joint] Segmentor loaded from finetune checkpoint.")

    # Optionally compile generator and segmentor for kernel fusion speedup.
    # Discriminator is excluded due to spectral norm hooks causing graph breaks.
    # RADIO is excluded as it is loaded via torch.hub with dynamic shapes.
    if use_compile:
        # torch.compile with the default inductor backend requires Triton,
        # which is not available on Windows. Use the "eager" backend as a
        # safe fallback that still benefits from Dynamo graph capture.
        import platform
        compile_backend = "eager" if platform.system() == "Windows" else "inductor"
        print(f"[Joint] Compiling generator and segmentor (backend={compile_backend})...")
        try:
            generator = torch.compile(generator, backend=compile_backend)
            segmentor = torch.compile(segmentor, backend=compile_backend)
            print("[Joint] Compilation complete.")
        except Exception as e:
            print(f"[Joint] torch.compile failed ({e}); falling back to eager mode.")

    # 2. Optimizers
    opt_g = optim.Adam(generator.parameters(), lr=training_config.generator_lr, betas=(0.9, 0.999), foreach=True)
    opt_d = optim.Adam(discriminator.parameters(), lr=training_config.discriminator_lr, betas=(0.9, 0.999), foreach=True)
    
    train_params = [{'params': segmentor.get_1x_lr_params(), 'lr': training_config.segmentor_lr},
                    {'params': segmentor.get_10x_lr_params(), 'lr': training_config.segmentor_lr * 10}]
    opt_seg = optim.SGD(train_params, momentum=0.9, weight_decay=5e-4, nesterov=True, foreach=True)

    # 3. Loss Functions
    criterion_l2 = nn.MSELoss()        # Eq (2)

    # Initialize ABL criterion only if enabled
    criterion_abl = None
    if training_config.use_abl_loss:
        criterion_abl = ABL(
            isdetach=True,
            max_N_ratio=1/100,
            ignore_label=255,
            label_smoothing=0.2,     # Paper recommends ~0.2 for conflict suppression
            max_clip_dist=20.0
        ).to(device)

    criterion_gan = nn.BCEWithLogitsLoss() # Eq (7, 8)
    criterion_ce = SegmentationLosses(weight=None, cuda=torch.cuda.is_available()).build_loss(mode='ce') # Eq (3)

    # 4. DataLoader (created here to avoid import-time path validation)
    train_loader = create_train_loader()

    # 5. Scheduler
    scheduler = LR_Scheduler(mode=training_config.lr_scheduler, 
                             base_lr=training_config.segmentor_lr, 
                             num_epochs=training_config.num_epochs, 
                             iters_per_epoch=len(train_loader))

    # 6. Training Loop
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
        
        for i, batch_data in enumerate(tbar):
            scheduler(opt_seg, i, epoch, best_pred)

            # Handle conditional distance maps based on ABL flag
            if training_config.use_abl_loss:
                # When ABL is enabled: (images, masks, distance_maps)
                images, masks, distance_maps = batch_data
                distance_maps = distance_maps.to(device, non_blocking=True)
            else:
                # When ABL is disabled: (images, masks)
                images, masks = batch_data

            # Data Prep
            real_img = images.to(device, non_blocking=True)  # I_gt
            masks_gt = masks.to(device, non_blocking=True)   # S_gt (Indices)
            
            # Create ULR Input
            lr_img = F.interpolate(real_img, size=(format_config.ultra_low_resolution, format_config.ultra_low_resolution), mode='bicubic', align_corners=False)
            lr_img = F.interpolate(lr_img, size=(format_config.low_resolution, format_config.low_resolution), mode='bicubic', align_corners=False)

            # ===================================================================================
            #  STEP 1: GENERATE & SEGMENT (Forward Pass)
            # ===================================================================================
            # AMP autocast only on generator forward (biggest compute win from bfloat16).
            # Everything else stays float32 to match baseline numerical behaviour.
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                fake_sr = generator(lr_img)
            # Immediately upcast to float32 so all downstream ops are full precision.
            fake_sr = fake_sr.float()

            # 2. Segment SR Image (S_pred)
            seg_logits = segmentor(fake_sr)
            seg_probs = torch.softmax(seg_logits, dim=1)

            # 3. Prepare Joint Inputs for Discriminator (Eq 10)
            masks_onehot = to_one_hot(masks_gt, model_config.num_classes)
            z_real = torch.cat([real_img, masks_onehot], dim=1)
            z_fake = torch.cat([fake_sr.detach(), seg_probs.detach()], dim=1)

            # Add instance noise (decays over training)
            noise_std = max(0.1 * (1 - epoch / training_config.num_epochs), 0.02)
            z_real_noisy = z_real + noise_std * torch.randn_like(z_real)
            z_fake_noisy = z_fake + noise_std * torch.randn_like(z_fake)

            # ===================================================================================
            #  STEP 2: TRAIN DISCRIMINATOR (Eq 7)
            # ===================================================================================
            opt_d.zero_grad()
            
            # Real Branch - One-sided label smoothing
            pred_d_real = discriminator(z_real_noisy)
            real_labels = torch.ones_like(pred_d_real) * training_config.label_smoothing_real
            loss_d_real = criterion_gan(pred_d_real, real_labels)
            
            # Fake Branch
            pred_d_fake = discriminator(z_fake_noisy)
            loss_d_fake = criterion_gan(pred_d_fake, torch.zeros_like(pred_d_fake))
            
            loss_d = loss_d_real + loss_d_fake
            
            # Only update discriminator if it's not already too strong
            should_update_d = loss_d.item() > 0.1
            
            if not (torch.isnan(loss_d) or torch.isinf(loss_d)) and should_update_d:
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                opt_d.step()
            elif not should_update_d:
                pass  # Skip D update silently when D is too strong
            else:
                print(f"[Warning] NaN/Inf in discriminator loss, skipping D update")
                loss_d = torch.tensor(0.0, device=device)

            # ===================================================================================
            #  STEP 3: TRAIN GENERATOR & SEGMENTOR JOINTLY (Eq 1)
            # ===================================================================================
            opt_g.zero_grad()
            opt_seg.zero_grad()
            
            # NOTE: Re-use fake_sr/seg_logits from Step 1 WITHOUT detach for gradient flow.
            # seg_probs is detached here: adversarial gradient should only reach the generator,
            # not the segmentor.  The segmentor trains exclusively via CE loss (alpha term).
            z_fake_grad = torch.cat([fake_sr, seg_probs.detach()], dim=1)
            
            # A. Calculate Generator Losses
            
            # 1. Pixel Loss (L2) - Eq (2)
            loss_2 = criterion_l2(fake_sr, real_img)
            
            # 2. Feature Loss (L1 + Cos) - Eq (4)
            # RADIO requires size divisible by 14 (378). Generator outputs 384.
            radio_size = (378, 378)
            real_for_radio = F.interpolate(real_img, size=radio_size, mode='bilinear', align_corners=False)
            fake_for_radio = F.interpolate(fake_sr, size=radio_size, mode='bilinear', align_corners=False)
            # Real features: no gradients needed (target)
            # Fake features: gradients needed (to train generator via L_fea)
            real_feat = feature_extractor(real_for_radio, no_grad=True)
            fake_feat = feature_extractor(fake_for_radio, no_grad=False)
            loss_fea = feature_loss_calc(real_feat.detach(), fake_feat)
            
            # 3. Adversarial Loss - Eq (11)
            pred_d_fake_g = discriminator(z_fake_grad)
            loss_adv = criterion_gan(pred_d_fake_g, torch.ones_like(pred_d_fake_g))

            # B. Calculate Segmentation Loss (L_ce) - Eq (3)
            loss_ce = criterion_ce(seg_logits, masks_gt)

            # ABL loss
            if training_config.use_abl_loss and criterion_abl is not None:
                loss_abl = criterion_abl(seg_logits, masks_gt, dist_maps=distance_maps)
                if loss_abl is None:
                    loss_abl = torch.tensor(0.0, device=device)
            else:
                loss_abl = torch.tensor(0.0, device=device)

            # C. Total Loss - Eq (1) — all float32
            gen_part = (training_config.lambda_1 * loss_2) + \
                       (training_config.lambda_2 * loss_fea) + \
                       (training_config.lambda_3 * loss_adv)
            
            abl_component = (training_config.lambda_abl * loss_abl) if training_config.use_abl_loss else 0
            total_loss = ((1 - training_config.alpha) * gen_part) + (training_config.alpha * loss_ce) + abl_component
            
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"[Warning] NaN/Inf detected in total_loss, skipping batch")
                print(f"  L_2: {loss_2.item()}, L_fea: {loss_fea.item()}, L_adv: {loss_adv.item()}, L_ce: {loss_ce.item()}, L_abl: {loss_abl.item()}")
                opt_g.zero_grad()
                opt_seg.zero_grad()
                continue
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(segmentor.parameters(), max_norm=1.0)
            opt_g.step()
            opt_seg.step()

            # Display
            current_lr = opt_seg.param_groups[0]['lr']
            if training_config.use_abl_loss:
                tbar.set_description(f"Ep {epoch+1} | L_D: {loss_d.item():.3f} | L_2: {loss_2.item():.3f} | L_CE: {loss_ce.item():.3f} | L_Adv: {loss_adv.item():.3f} | L_abl: {loss_abl.item():.3f}")
            else:
                tbar.set_description(f"Ep {epoch+1} | L_D: {loss_d.item():.3f} | L_2: {loss_2.item():.3f} | L_CE: {loss_ce.item():.3f} | L_Adv: {loss_adv.item():.3f}")

        # Checkpointing - save as single file compatible with inference.py load_models()
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'gen_state_dict': generator.state_dict(),
                'seg_state_dict': segmentor.state_dict(),
                'epoch': epoch + 1
            }
            ckpt_name = get_checkpoint_name('joint', epoch=epoch + 1)
            ckpt_path = get_checkpoint_path(ckpt_name)
            torch.save(checkpoint, ckpt_path)
            print(f"[Joint] Checkpoint saved to: {ckpt_path}")
    
    # Save final checkpoint
    final_checkpoint = {
        'gen_state_dict': generator.state_dict(),
        'seg_state_dict': segmentor.state_dict(),
        'epoch': training_config.num_epochs
    }
    final_name = get_checkpoint_name('joint', is_final=True)
    final_path = get_checkpoint_path(final_name)
    if os.path.exists(final_path):
        print(f"[WARNING] Overwriting existing checkpoint: {final_path}")
    torch.save(final_checkpoint, final_path)
    print(f"[Joint] Training complete. Final checkpoint saved to: {final_path}")

if __name__ == "__main__":
    train_joint()