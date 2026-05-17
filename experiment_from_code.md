# ULR2SS Experiment Documentation

## Executive Summary

This experiment investigates whether incorporating **Active Boundary Loss (ABL)** into an ultra-low-resolution semantic segmentation pipeline (ULR2SS) improves boundary accuracy without degrading standard segmentation metrics. The approach jointly trains a Super-Resolution (SR) generator (ESRGAN-based) with a semantic segmentation model (DeepLabV3+) to perform segmentation on 16×16 ultra-low-resolution images that are first upscaled to 384×384.

---

## 1. Problem Statement

### 1.1 Background
Semantic segmentation at ultra-low resolution (ULR) faces significant challenges:
- **Boundary ambiguity**: Limited pixel density causes edges to be represented by only a few pixels
- **Mixed pixels**: Boundary pixels contain information from multiple semantic classes
- **Spatial aliasing**: Uncertainty in precise boundary positions

### 1.2 Research Question
*Can Active Boundary Loss (ABL) improve boundary accuracy in ultra-low-resolution semantic segmentation without compromising standard pixel-level metrics (mIoU, Pixel Accuracy)?*

### 1.3 Hypothesis
Incorporating ABL into the joint SR + segmentation training pipeline will:
1. Improve boundary-focused metrics (Boundary F1, Hausdorff Distance)
2. Maintain or improve standard metrics (mIoU, Pixel Accuracy)

---

## 2. Architecture Overview

### 2.1 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ULR2SS TRAINING PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Input: I_gt (384×384 HR)                                                  │
│            │                                                                │
│            ▼                                                                │
│   ┌─────────────────┐                                                       │
│   │  Downsample to  │ → I_ulr (16×16) → Upsample → I_lr (96×96)            │
│   │   Ultra-Low Res │                                                       │
│   └─────────────────┘                                                       │
│            │                                                                │
│            ▼                                                                │
│   ┌─────────────────┐                                                       │
│   │   ESRGAN        │ → I_sr (384×384 Super-Resolved)                       │
│   │   Generator     │                                                       │
│   └─────────────────┘                                                       │
│            │                                                                │
│    ┌───────┴───────┐                                                        │
│    │               │                                                        │
│    ▼               ▼                                                        │
│ ┌──────────┐  ┌──────────────┐                                              │
│ │DeepLabV3+│  │ RADIO        │                                              │
│ │Segmentor │  │ Feature Ext. │                                              │
│ └──────────┘  └──────────────┘                                              │
│    │               │                                                        │
│    ▼               ▼                                                        │
│ S_pred        L_fea (Feature Loss)                                          │
│    │                                                                        │
│    └────────────┬───────────────────────────────────────────────────────┐   │
│                 │                                                       │   │
│                 ▼                                                       │   │
│         ┌───────────────┐                                               │   │
│         │ Discriminator │ ← z_fake = concat(I_sr, S_pred)               │   │
│         │ (PatchGAN)    │ ← z_real = concat(I_gt, S_gt)                 │   │
│         └───────────────┘                                               │   │
│                 │                                                       │   │
│                 ▼                                                       │   │
│              L_adv                                                      │   │
│                                                                         │   │
│   LOSS COMPUTATION:                                                     │   │
│   L_tot = (1-α)(λ₁L₂ + λ₂L_fea + λ₃L_adv) + αL_ce + λ_abl·L_abl        │   │
│                                                                         │   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Resolution Pipeline

| Stage | Resolution | Description |
|-------|------------|-------------|
| Ground Truth (I_gt) | 384×384 | High-resolution input image |
| Ultra-Low Resolution (I_ulr) | 16×16 | Simulated ULR input via bicubic downsampling |
| Low Resolution (I_lr) | 96×96 | Intermediate upsampled resolution |
| Super-Resolved (I_sr) | 384×384 | Generator output (4× upscale from 96×96) |

---

## 3. Model Components

### 3.1 Generator: ESRGAN Architecture

The generator uses RRDB (Residual-in-Residual Dense Block) architecture from ESRGAN:

**Architecture Details:**
- **Input channels**: 3 (RGB)
- **Feature channels**: 64
- **Number of RRDB blocks**: 23
- **Upsampling**: 2× nearest-neighbor upsampling (applied twice for 4× total)
- **Residual scaling**: β = 0.2

**RRDB Block Structure:**
```
RRDB = 3 × DenseBlock with residual connection
DenseBlock = 5 × ConvBlock with dense connections
ConvBlock = Conv2d(3×3) + LeakyReLU(0.2)
```

### 3.2 Discriminator: PatchGAN with Spectral Normalization

**Input**: Concatenated image + segmentation mask
- Phase 1 (Pretraining): 3 channels (RGB only)
- Phase 2 (Joint training): 3 + N channels (RGB + one-hot encoded mask, where N = num_classes)

**Architecture:**
```
Configuration: [(kernel, out_channels, stride), ...]
disc_config = [
    (3, 64, 1), (3, 64, 2),    # 64 channels
    (3, 128, 1), (3, 128, 2),  # 128 channels
    (3, 256, 1), (3, 256, 2),  # 256 channels
    (3, 512, 1), (3, 512, 2),  # 512 channels
]
+ AdaptiveAvgPool(6×6) + FC(512×36 → 1024) + FC(1024 → 1)
```

**Stability Techniques:**
- Spectral Normalization on all Conv2d and Linear layers
- One-sided label smoothing (real labels: 0.9 instead of 1.0)
- Instance noise injection (decays from 0.1 to 0.02 over training)
- Discriminator update skipping when D loss < 0.1

### 3.3 Segmentor: DeepLabV3+

**Architecture:**
- **Backbone**: ResNet-101 (pretrained on ImageNet)
- **Output stride**: 16
- **ASPP (Atrous Spatial Pyramid Pooling)**: Multi-scale feature extraction
- **Decoder**: Combines low-level and high-level features
- **BatchNorm**: Standard BatchNorm2d (frozen during training)

**Learning Rate Strategy:**
- Backbone: 1× base LR
- ASPP + Decoder: 10× base LR

### 3.4 Feature Extractors

#### 3.4.1 NVIDIA RADIO (Joint Training)
- **Model**: RADIOv2.5-g (loaded via torch.hub)
- **Purpose**: Perceptual feature loss computation
- **Input requirement**: Images resized to 378×378 (divisible by 14)
- **Frozen**: All parameters frozen (fixed feature extractor)

#### 3.4.2 VGG19 (Pretraining)
- **Layers**: First 35 layers (before 5th pooling)
- **Purpose**: Standard perceptual loss for SR pretraining
- **Frozen**: All parameters frozen

---

## 4. Loss Functions

### 4.1 Total Loss Equation

$$
\mathcal{L}_{\text{tot}} = (1-\alpha)\left(\lambda_1 \mathcal{L}_2 + \lambda_2 \mathcal{L}_{\text{fea}} + \lambda_3 \mathcal{L}_{\text{adv}}\right) + \alpha \mathcal{L}_{\text{ce}} + \lambda_{\text{abl}} \mathcal{L}_{\text{abl}}
$$

### 4.2 Individual Loss Components

| Loss | Symbol | Equation | Purpose |
|------|--------|----------|---------|
| **Pixel Loss (L2/MSE)** | $\mathcal{L}_2$ | $\frac{1}{HWC}\sum_{h,w,c}(I_{gt} - I_{sr})^2$ | Image reconstruction quality |
| **Feature Loss** | $\mathcal{L}_{\text{fea}}$ | $\mathcal{L}_1 + \mathcal{L}_{\cos}$ | Perceptual similarity |
| **L1 Component** | $\mathcal{L}_1$ | $\|\hat{F}_{\text{real}} - \hat{F}_{\text{fake}}\|_1$ | Feature magnitude difference |
| **Cosine Component** | $\mathcal{L}_{\cos}$ | $1 - \cos(\hat{F}_{\text{real}}, \hat{F}_{\text{fake}})$ | Feature direction similarity |
| **Adversarial Loss** | $\mathcal{L}_{\text{adv}}$ | $\text{BCE}(D(z_{\text{fake}}), 1)$ | Generator fooling discriminator |
| **Discriminator Loss** | $\mathcal{L}_D$ | $\text{BCE}(D(z_{\text{real}}), 1) + \text{BCE}(D(z_{\text{fake}}), 0)$ | Real/fake discrimination |
| **Cross-Entropy** | $\mathcal{L}_{\text{ce}}$ | $-x_y + \log\sum_i e^{x_i}$ | Segmentation classification |
| **Active Boundary Loss** | $\mathcal{L}_{\text{abl}}$ | See Section 4.3 | Boundary-focused loss |

### 4.3 Active Boundary Loss (ABL) Details

ABL (Wang et al., 2022) focuses training on boundary regions by:

1. **Boundary Detection**: Using KL divergence between adjacent pixel predictions to identify uncertain boundaries
2. **Direction Prediction**: Learning to predict the direction toward the nearest ground truth boundary
3. **Distance Weighting**: Weighting loss by distance to boundary (closer = higher weight)

**Key ABL Parameters:**
```python
ABL(
    isdetach=True,           # Detach neighbor logits (stability)
    max_N_ratio=1/100,       # Max boundary pixel ratio
    ignore_label=255,        # Ignore label index
    label_smoothing=0.2,     # Conflict-suppression smoothing (paper recommendation)
    max_clip_dist=20.0       # Distance clipping threshold
)
```

**ABL Algorithm:**
1. Compute ground truth boundary from mask transitions
2. Compute distance transform from boundary pixels
3. Extract predicted boundaries via KL divergence thresholding
4. For each predicted boundary pixel, compute direction toward nearest GT boundary
5. Compute weighted cross-entropy over predicted directions

---

## 5. Training Configuration

### 5.1 Loss Weights

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Balancing parameter | α | 0.3 | Balance between generator (0.7) and segmentation (0.3) loss |
| Pixel loss weight | λ₁ | 0.5 | L2 reconstruction weight |
| Feature loss weight | λ₂ | 0.01 | RADIO perceptual loss weight |
| Adversarial loss weight | λ₃ | 0.005 | GAN loss weight (lowered for stability) |
| ABL loss weight | λ_abl | 0.02 | Active Boundary Loss weight |

### 5.2 Optimizer Settings

| Component | Optimizer | Learning Rate | Momentum/Betas | Weight Decay |
|-----------|-----------|---------------|----------------|--------------|
| Generator | Adam | 1×10⁻⁴ | β=(0.9, 0.999) | - |
| Discriminator | Adam | 5×10⁻⁶ | β=(0.9, 0.999) | - |
| Segmentor (backbone) | SGD (Nesterov) | 1×10⁻³ | momentum=0.9 | 5×10⁻⁴ |
| Segmentor (head) | SGD (Nesterov) | 1×10⁻² | momentum=0.9 | 5×10⁻⁴ |

### 5.3 Learning Rate Schedule

- **Scheduler**: Polynomial decay (`poly`)
- **Formula**: $lr = base\_lr \times (1 - \frac{iter}{max\_iter})^{power}$
- **Applied to**: Segmentor only

### 5.4 GAN Stability Measures

| Technique | Implementation | Purpose |
|-----------|----------------|---------|
| Spectral Normalization | Applied to all D weights | Lipschitz constraint |
| Label Smoothing | Real labels = 0.9 | Prevent D overconfidence |
| Instance Noise | Gaussian, σ decays 0.1→0.0 over training | Prevent mode collapse |
| D Update Frequency | Update D every `d_update_freq` steps (default: 2) | Prevent D dominance on small datasets |
| Gradient Clipping | max_norm = 1.0 | Prevent gradient explosion |

---

## 6. Training Pipeline

### 6.1 Phase 1: SR Pretraining

**Purpose**: Initialize generator with SR capability before joint training

**Loss Function:**
$$\mathcal{L}_G = \mathcal{L}_{L1} + 5 \times 10^{-3} \mathcal{L}_{VGG} + 10^{-2} \mathcal{L}_{GAN}$$

**Training Process:**
1. Sample HR images from dataset
2. Degrade to ULR (16×16) via bicubic downsampling
3. Upsample to LR (96×96) via bicubic
4. Generate SR image (384×384) via Generator
5. Compute losses against original HR
6. Update Generator and Discriminator

**Output Checkpoints:**
- `pretrained_generator.pth`
- `pretrained_discriminator.pth`

### 6.2 Phase 2: Joint Training

**Purpose**: Jointly optimize SR and segmentation with ABL

**Training Loop (per batch):**
```
1. Load (I_gt, S_gt, dist_maps)
2. Create ULR: I_ulr = downsample(I_gt, 16×16)
3. Create LR:  I_lr = upsample(I_ulr, 96×96)
4. Generate:   I_sr = Generator(I_lr)
5. Segment:    S_pred = Segmentor(I_sr)
6. Discriminate:
   - z_real = concat(I_gt, one_hot(S_gt))
   - z_fake = concat(I_sr, softmax(S_pred))
7. Compute losses (L2, L_fea, L_adv, L_ce, L_abl)
8. Update D (if L_D > 0.1)
9. Update G and Seg
```

**Checkpointing:**
- Every 5 epochs: `joint_checkpoint_ep{N}.pth`
- Final: `joint_checkpoint_final.pth`

### 6.3 Phase 3: Evaluation

**Metrics Computed:**

| Category | Metric | Formula Reference |
|----------|--------|-------------------|
| Pixel-Level | mIoU | Mean Intersection over Union |
| Pixel-Level | PA | Pixel Accuracy |
| Pixel-Level | PA Class | Per-class Pixel Accuracy |
| Pixel-Level | FWIoU | Frequency Weighted IoU |
| Boundary | Boundary F1 | F1 with tolerance τ=2 pixels |
| Boundary | Symmetric Boundary Dice | Dice after τ-dilation |
| Boundary | Hausdorff Distance | Max boundary mismatch |
| Boundary | Mean Hausdorff Distance | Mean HD across samples |
| Boundary | Average Surface Distance | Mean boundary distance |

---

## 7. Dataset

### 7.1 Source
- **Dataset**: SunRGBD (13 semantic classes + background)
- **Reference**: [sunrgbd-meta-data](https://github.com/ankurhanda/sunrgbd-meta-data)
- **Classes**: 14 total (including background)

### 7.2 Data Structure
```
datasets/
├── custom/
│   ├── rgb/         # Training RGB images
│   └── label/       # Training segmentation masks
└── custom_demo/
    ├── rgb/         # Evaluation RGB images
    └── label/       # Evaluation ground truth masks
```

### 7.3 Data Preprocessing

| Transform | Training | Evaluation |
|-----------|----------|------------|
| Center Crop | 384×384 | 384×384 |
| Downsample (ULR) | Bicubic to 16×16 | Bicubic to 16×16 |
| Upsample (LR) | Bicubic to 96×96 | Bicubic to 96×96 |
| Normalization | ToTensor [0,1] | ToTensor [0,1] |
| Mask Transform | LongTensor | LongTensor |

### 7.4 Distance Map Computation
When ABL is enabled, distance maps are precomputed during data loading:
1. Compute GT boundary using mask value transitions
2. Apply signed distance transform (negative inside, positive outside)
3. Store as additional tensor per sample

---

## 8. Experimental Results

*To be populated after running full experiment.*

**Pixel-Level Metrics:**

| Metric | Value |
|--------|-------|
| mIoU | - |
| PA | - |
| PA Class | - |
| FWIoU | - |

**Boundary Metrics (τ=2 pixels):**

| Metric | Value |
|--------|-------|
| Boundary F1 | - |
| Symmetric Boundary Dice | - |
| Hausdorff Distance | - |
| Mean Hausdorff Distance | - |
| Average Surface Distance | - |

---

## 9. Implementation Details

### 9.1 Key Files

| File | Purpose |
|------|---------|
| `config.py` | Global configuration and hyperparameters |
| `full_pipeline.py` | Orchestrates training phases |
| `pretraining.py` | Phase 1: SR pretraining |
| `training.py` | Phase 2: Joint training with ABL |
| `evaluation.py` | Phase 3: Metric computation |
| `validation.py` | Per-epoch boundary-aware validation (saves best checkpoint by BF₁) |
| `batch_inference.py` | Standalone batch inference (images only, no metrics) |
| `paths.py` | Checkpoint path helpers (dated folder resolution) |
| `esrgan.py` | Generator and Discriminator architectures |
| `modeling/deeplab.py` | DeepLabV3+ segmentation model |
| `abl/abl.py` | Active Boundary Loss implementation |
| `utils2/metrics.py` | Evaluation metrics (mIoU, boundary metrics) |

### 9.2 Checkpoints Directory Structure
```
checkpoints/
├── joint_checkpoint_best.pth
├── joint_checkpoint_ep30.pth
├── MM-DD/                           # Date-organized folders
│   ├── pretrained_generator.pth
│   ├── pretrained_discriminator.pth
│   └── joint_checkpoint_final.pth
```

### 9.3 Running the Experiment

```bash
# Full pipeline (pretrain → joint train)
python full_pipeline.py

# Full pipeline with evaluation
python full_pipeline.py --evaluate

# Skip pretraining (load existing weights)
python full_pipeline.py --skip-pretrain

# Evaluate only
python full_pipeline.py --eval-only --checkpoint path/to/checkpoint.pth

# Batch inference (no metrics)
python full_pipeline.py --batch-inference
```

---

## 10. Ablation Considerations

### 10.1 ABL Enable/Disable

The experiment can be run with or without ABL via `training_config.use_abl_loss`:

```python
training_config = SimpleNamespace(
    ...
    use_abl_loss = True,  # Set to False for baseline
    lambda_abl = 0.02,    # ABL weight when enabled
    ...
)
```

When ABL is disabled:
- Dataset returns only `(image, mask)` instead of `(image, mask, dist_maps)`
- ABL loss term is set to 0 in total loss computation
- Training loop skips ABL-related computations

### 10.2 Potential Ablations for Future Work

1. **ABL Weight Sensitivity**: Test λ_abl ∈ {0.01, 0.02, 0.05, 0.1}
2. **ABL vs. Baseline**: Compare with `use_abl_loss=False`
3. **Resolution Study**: Test at different ULR sizes (8×8, 16×16, 32×32)
4. **Feature Extractor**: Compare RADIO vs. VGG19 for joint training
5. **Loss Balance**: Vary α (segmentation vs. generation weight)

---

## 11. References

### 11.1 Papers
- **ULR2SS Base**: Huang et al., "Improved Semantic Segmentation for Ultra-Low-Resolution Images"
- **Active Boundary Loss**: Wang et al., "Active Boundary Loss for Semantic Segmentation" (2022)
- **ESRGAN**: Wang et al., "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks"
- **DeepLabV3+**: Chen et al., "Encoder-Decoder with Atrous Separable Convolution"
- **RADIO**: NVIDIA AM-RADIO visual foundation model

### 11.2 Implementation Sources
- SunRGBD dataset: https://github.com/ankurhanda/sunrgbd-meta-data
- Original ULR2SS: https://github.com/hxy-0818/ULR2SS

---

## 12. Conclusion

This experiment implements a joint SR + semantic segmentation pipeline with Active Boundary Loss for ultra-low-resolution images. The implementation provides:

1. **Complete ULR segmentation pipeline**: From 16×16 input through SR to segmentation at 384×384
2. **ABL integration**: Boundary-focused loss term added to joint training
3. **Comprehensive evaluation suite**: Both pixel-level and boundary-specific metrics
4. **Ablation capability**: Easy toggle between ABL-enabled and baseline training via config

*Final conclusions to be drawn after running full experiments with proper train/test splits.*
