import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import pretraining_config
from esrgan import Generator, Discriminator, disc_config
from training.dataloder import pretrain_loader
from training.feature_extractor import VGG19FeatureExtractor  # standard VGG perceptual


def apply_spectral_norm(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.utils.spectral_norm(module)



def pretrain_sr():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SR PRETRAIN] Device: {device}")

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
        generator.parameters(), lr=pretraining_config.generator_lr, betas=(0.9, 0.999)
    )

    opt_d = optim.Adam(
        discriminator.parameters(),
        lr=pretraining_config.discriminator_lr,
        betas=(0.9, 0.999),
    )

    # =========================
    # Losses
    # =========================

    criterion_l1 = nn.L1Loss()
    criterion_gan = nn.BCEWithLogitsLoss()

    # =========================
    # Training Loop
    # =========================

    for epoch in range(pretraining_config.num_epochs):
        generator.train()
        discriminator.train()

        tbar = tqdm(pretrain_loader, desc=f"SR Pretrain Epoch {epoch+1}")

        for lr_img, hr_img in tbar:
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            # =====================================================
            # Train Discriminator
            # =====================================================
            opt_d.zero_grad()

            with torch.no_grad():
                fake_sr = generator(lr_img)

            pred_real = discriminator(hr_img)
            pred_fake = discriminator(fake_sr.detach())

            loss_d_real = criterion_gan(pred_real, torch.ones_like(pred_real))
            loss_d_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            opt_d.step()

            # =====================================================
            # Train Generator
            # =====================================================
            opt_g.zero_grad()

            fake_sr = generator(lr_img)

            # 1. Pixel loss (MAE)
            loss_l1 = criterion_l1(fake_sr, hr_img)

            # 2. VGG perceptual loss
            # Extract features
            fake_features = feature_extractor(fake_sr)

            # We detach real_features because we don't want to backpropagate
            # through the VGG model or the real image.
            real_features = feature_extractor(hr_img).detach()

            # Calculate L1 distance between the feature maps
            loss_vgg = criterion_l1(fake_features, real_features)

            # 3. Adversarial loss
            pred_fake = discriminator(fake_sr)
            loss_gan = criterion_gan(pred_fake, torch.ones_like(pred_fake))

            # Total Loss
            loss_g = (
                loss_l1
                + pretraining_config.vgg_weight * loss_vgg
                + pretraining_config.gan_weight * loss_gan
            )

            loss_g.backward()
            opt_g.step()

            tbar.set_postfix(
                {
                    "L_D": f"{loss_d.item():.3f}",
                    "L_L1": f"{loss_l1.item():.3f}",
                    "L_VGG": f"{loss_vgg.item():.3f}",
                    "L_GAN": f"{loss_gan.item():.3f}",
                }
            )

        # =========================
        # Checkpoint
        # =========================
        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), f"sr_generator_pretrain_ep{epoch+1}.pth")

    print("[SR PRETRAIN] Finished successfully")


if __name__ == "__main__":
    pretrain_sr()
