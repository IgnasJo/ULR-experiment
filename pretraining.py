import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import pretraining_config, training_config
from paths import get_checkpoint_path
from esrgan import Generator, Discriminator, disc_config
from training.dataloder import create_pretrain_loader
from training.feature_extractor import VGG19FeatureExtractor  # standard VGG perceptual


def apply_spectral_norm(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.utils.spectral_norm(module)



def pretrain_sr(save_path=None, save_disc_path=None):
    """
    Pretrain the SR Generator.
    
    Args:
        save_path: Path to save the final pretrained generator weights
        save_disc_path: Path to save the final pretrained discriminator weights
        
    Returns:
        Tuple of (generator_path, discriminator_path)
    """
    if save_path is None:
        save_path = get_checkpoint_path("pretrained_generator.pth")
    if save_disc_path is None:
        save_disc_path = get_checkpoint_path("pretrained_discriminator.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = training_config.use_amp and device.type == 'cuda'
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    print(f"[SR PRETRAIN] Device: {device}")
    print(f"[SR PRETRAIN] Mixed precision (AMP): {use_amp} (dtype={amp_dtype})")
    print(f"[SR PRETRAIN] Will save to: {save_path}")

    # =========================
    # Models
    # =========================

    generator = Generator().to(device)

    discriminator = Discriminator(in_channels=3, disc_config=disc_config).to(device)
    discriminator.apply(apply_spectral_norm)

    # Initialize the Feature Extractor
    feature_extractor = VGG19FeatureExtractor().to(device)

    # =========================
    # Optimizers
    # =========================

    opt_g = optim.Adam(
        generator.parameters(), lr=pretraining_config.generator_lr, betas=(0.9, 0.999), foreach=True
    )

    opt_d = optim.Adam(
        discriminator.parameters(),
        lr=pretraining_config.discriminator_lr,
        betas=(0.9, 0.999),
        foreach=True,
    )

    # =========================
    # Losses
    # =========================

    criterion_l1 = nn.L1Loss()
    criterion_gan = nn.BCEWithLogitsLoss()

    # =========================
    # Training Loop
    # =========================

    # Create DataLoader (created here to avoid import-time path validation)
    pretrain_loader = create_pretrain_loader()

    for epoch in range(pretraining_config.num_epochs):
        generator.train()
        discriminator.train()

        tbar = tqdm(pretrain_loader, desc=f"SR Pretrain Epoch {epoch+1}")

        for lr_img, hr_img in tbar:
            lr_img = lr_img.to(device, non_blocking=True)
            hr_img = hr_img.to(device, non_blocking=True)

            # =====================================================
            # Train Discriminator
            # =====================================================
            opt_d.zero_grad()

            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                with torch.no_grad():
                    fake_sr = generator(lr_img)
            fake_sr = fake_sr.float()

            pred_real = discriminator(hr_img)
            pred_fake = discriminator(fake_sr.detach())

            loss_d_real = criterion_gan(pred_real, torch.ones_like(pred_real))
            loss_d_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))
            loss_d = loss_d_real + loss_d_fake

            log_d = loss_d.item()
            loss_d.backward()
            opt_d.step()

            # =====================================================
            # Train Generator
            # =====================================================
            opt_g.zero_grad()

            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                fake_sr = generator(lr_img)
            fake_sr = fake_sr.float()

            # 1. Pixel loss (MAE)
            loss_l1 = criterion_l1(fake_sr, hr_img)

            # 2. VGG perceptual loss
            fake_features = feature_extractor(fake_sr)
            real_features = feature_extractor(hr_img).detach()
            loss_vgg = criterion_l1(fake_features, real_features)

            # 3. Adversarial loss
            pred_fake = discriminator(fake_sr)
            loss_gan = criterion_gan(pred_fake, torch.ones_like(pred_fake))

            loss_g = (
                loss_l1
                + pretraining_config.vgg_weight * loss_vgg
                + pretraining_config.gan_weight * loss_gan
            )

            log_l1 = loss_l1.item()
            log_vgg = loss_vgg.item()
            log_gan = loss_gan.item()

            loss_g.backward()
            opt_g.step()

            tbar.set_postfix(
                {
                    "L_D": f"{log_d:.3f}",
                    "L_L1": f"{log_l1:.3f}",
                    "L_VGG": f"{log_vgg:.3f}",
                    "L_GAN": f"{log_gan:.3f}",
                }
            )

        # =========================
        # Checkpoint
        # =========================
        if (epoch + 1) % 10 == 0:
            ckpt_path = get_checkpoint_path(f"sr_generator_pretrain_ep{epoch+1}.pth")
            torch.save(generator.state_dict(), ckpt_path)
            print(f"[SR PRETRAIN] Checkpoint saved to: {ckpt_path}")

    # Save final pretrained weights
    final_gen_path = get_checkpoint_path(save_path)
    if os.path.exists(final_gen_path):
        print(f"[WARNING] Overwriting existing file: {final_gen_path}")
    torch.save(generator.state_dict(), final_gen_path)
    print(f"[SR PRETRAIN] Generator saved to: {final_gen_path}")
    
    # Save discriminator weights for Phase 2 joint training
    # Note: Phase 2 discriminator has different input channels (3+N),
    # but the loading function handles the channel mismatch automatically
    final_disc_path = get_checkpoint_path(save_disc_path)
    if os.path.exists(final_disc_path):
        print(f"[WARNING] Overwriting existing file: {final_disc_path}")
    torch.save(discriminator.state_dict(), final_disc_path)
    print(f"[SR PRETRAIN] Discriminator saved to: {final_disc_path}")
    
    print(f"[SR PRETRAIN] Finished successfully.")
    
    return final_gen_path, final_disc_path


if __name__ == "__main__":
    pretrain_sr()
