import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from training.feature_extractor import RADIOFeatureExtractor

from modeling.deeplab import DeepLab
from utils2.loss import SegmentationLosses
from utils2.lr_scheduler import LR_Scheduler
from esrgan import Generator, Discriminator, disc_config
from training.dataloder import train_loader 
from config import training_config, format_config

def apply_spectral_norm(module):
    """Recursively applies spectral normalization to Conv2d and Linear layers."""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.utils.spectral_norm(module)

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
    """
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
        
    cos_sim = F.cosine_similarity(f_real_flat, f_fake_flat, dim=1).mean()
    l_cos = 1 - cos_sim
    
    return l1 + l_cos

def train_joint():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Joint Training on {device}...")

    # 1. Initialize Models
    
    # A. Generator (Super Resolution)
    generator = Generator().to(device)

    # B. Discriminator 
    # as per Eq (10): z = concat(I, S)
    # Calculate input channels: 3 (RGB) + num_classes (Mask Channels)
    # Example: 3 + 14 = 17 channels
    disc_in_channels = 3 + training_config.num_classes
    
    discriminator = Discriminator(in_channels=disc_in_channels, disc_config=disc_config).to(device)

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

            # ===================================================================================
            #  STEP 2: TRAIN DISCRIMINATOR (Eq 7)
            # ===================================================================================
            opt_d.zero_grad()
            
            # Real Branch
            pred_d_real = discriminator(z_real)
            loss_d_real = criterion_gan(pred_d_real, torch.ones_like(pred_d_real))
            
            # Fake Branch
            pred_d_fake = discriminator(z_fake)
            loss_d_fake = criterion_gan(pred_d_fake, torch.zeros_like(pred_d_fake))
            
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            opt_d.step()

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
            with torch.no_grad():
                real_feat = feature_extractor(real_for_radio)
            fake_feat = feature_extractor(fake_for_radio)
            loss_fea = feature_loss_calc(real_feat, fake_feat)
            
            # 3. Adversarial Loss - Eq (11)
            pred_d_fake_g = discriminator(z_fake_grad)
            loss_adv = criterion_gan(pred_d_fake_g, torch.ones_like(pred_d_fake_g))
            
            # B. Calculate Segmentation Loss (L_ce) - Eq (3)
            loss_ce = criterion_ce(seg_logits, masks_gt)

            # C. Total Loss - Eq (1)
            # L_tot = (1 - alpha) * (lam1*L2 + lam2*L_fea + lam3*L_adv) + alpha * L_ce
            
            gen_part = (training_config.lambda_1 * loss_2) + \
                       (training_config.lambda_2 * loss_fea) + \
                       (training_config.lambda_3 * loss_adv)
            
            total_loss = ((1 - training_config.alpha) * gen_part) + (training_config.alpha * loss_ce)
            
            total_loss.backward()
            
            opt_g.step()
            opt_seg.step()

            # Display
            current_lr = opt_seg.param_groups[0]['lr']
            tbar.set_description(f"Ep {epoch+1} | L_D: {loss_d.item():.3f} | L_2: {loss_2.item():.3f} | L_CE: {loss_ce.item():.3f} | L_Adv: {loss_adv.item():.3f}")

        # Checkpointing
        if (epoch + 1) % 5 == 0:
            torch.save(generator.state_dict(), f"generator_ep{epoch+1}.pth")
            torch.save(segmentor.state_dict(), f"segmentor_ep{epoch+1}.pth")

if __name__ == "__main__":
    train_joint()