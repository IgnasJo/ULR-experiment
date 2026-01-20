# Full Training Pipeline Documentation

Comprehensive command-line interface for SR (Super Resolution) + Semantic Segmentation joint training pipeline.

## Overview

The `full_pipeline.py` orchestrates three training phases:

1. **Phase 1: Pretraining** - Train the Generator (SR network) with VGG perceptual loss
2. **Phase 2: Joint Training** - Train Generator + Segmentor jointly using all loss functions
3. **Phase 3: Evaluation** - Evaluate the trained model on test set

## Prerequisites

```bash
pip install -r requirements.txt
```

## Usage

### 1. Full Pipeline (All Phases)

Run the complete training workflow from pretraining to joint training:

```bash
python full_pipeline.py
```

**Output:**
- Pretrained generator: `pretrained_generator.pth`
- Intermediate checkpoints: `joint_checkpoint_ep{N}.pth` (every 5 epochs)
- Final checkpoint: `joint_checkpoint_final.pth`

---

### 2. Full Pipeline + Evaluation

Run complete training and then evaluate the final model:

```bash
python full_pipeline.py --evaluate
```

**Output:**
- All training outputs (as above)
- Evaluation results in `evaluation_output_32res/` (configurable)
- Metrics file: `final_results.txt`
- Visualizations: `bleeding_edge_error.png`, `bf_score_curve.png`

---

### 3. Skip Pretraining

If you already have pretrained weights, skip Phase 1 and go directly to joint training:

```bash
python full_pipeline.py --skip-pretrain
```

**Uses default:** `pretrained_generator.pth`

Or specify custom pretrained weights:

```bash
python full_pipeline.py --skip-pretrain --pretrained-path path/to/my_pretrained.pth
```

**Output:**
- Loads existing pretrained weights
- Runs Phase 2 (joint training)
- Same checkpoint outputs as full pipeline

---

### 4. Pretrain Only

Run only the pretraining phase (useful for experimentation):

```bash
python full_pipeline.py --pretrain-only
```

**Output:**
- Intermediate checkpoints: `sr_generator_pretrain_ep{N}.pth` (every 10 epochs)
- Final weights: `pretrained_generator.pth`

Custom output path:

```bash
python full_pipeline.py --pretrain-only --pretrained-path my_sr_model.pth
```

---

### 5. Joint Training Only

Run only the joint training phase with existing pretrained weights:

```bash
python full_pipeline.py --joint-only path/to/pretrained_generator.pth
```

**Output:**
- Intermediate checkpoints: `joint_checkpoint_ep{N}.pth` (every 5 epochs)
- Final checkpoint: `joint_checkpoint_final.pth`

**Example:**
```bash
python full_pipeline.py --joint-only pretrained_generator.pth
```

---

### 6. Evaluation Only

Evaluate a trained model without any training:

```bash
python full_pipeline.py --eval-only
```

**Uses default checkpoint** from config: `checkpoints/joint_checkpoint_best.pth`

**Output:**
- Inference results in `evaluation_output_32res/`
- Segmentation masks, super-resolved images, visualizations
- Metrics in `final_results.txt`

---

### 7. Evaluate Specific Checkpoint

Evaluate a particular checkpoint:

```bash
python full_pipeline.py --eval-only --checkpoint path/to/joint_checkpoint_ep50.pth
```

With custom output directory:

```bash
python full_pipeline.py --eval-only --checkpoint joint_checkpoint_final.pth --eval-output results/final_evaluation/
```

**Example:**
```bash
python full_pipeline.py --eval-only --checkpoint checkpoints/my_model.pth --eval-output my_results/
```

---

## Command Reference

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--skip-pretrain` | flag | - | Skip Phase 1, load existing pretrained weights |
| `--pretrain-only` | flag | - | Run only Phase 1 pretraining |
| `--pretrained-path` | str | `pretrained_generator.pth` | Path to save/load pretrained weights |
| `--joint-only` | str | None | Run only joint training with specified weights |
| `--evaluate` | flag | - | Run evaluation after training (Phase 3) |
| `--eval-only` | flag | - | Run only evaluation, skip training |
| `--checkpoint` | str | None | Checkpoint path for evaluation (uses config default if None) |
| `--eval-output` | str | None | Output directory for evaluation results (uses config default if None) |

---

## Workflow Examples

### Example 1: First-time full training with evaluation

```bash
python full_pipeline.py --evaluate
```

Runs:
1. Pretraining → `pretrained_generator.pth`
2. Joint training → `joint_checkpoint_final.pth`
3. Evaluation on final model

### Example 2: Resume training from pretrained weights

```bash
python full_pipeline.py --skip-pretrain --pretrained-path pretrained_generator.pth --evaluate
```

Runs:
1. Skips pretraining (loads `pretrained_generator.pth`)
2. Joint training
3. Evaluation

### Example 3: Experiment with pretraining only

```bash
python full_pipeline.py --pretrain-only --pretrained-path experiment_sr_v1.pth
```

Saves SR model to `experiment_sr_v1.pth` for later use

### Example 4: Evaluate checkpoint at epoch 50

```bash
python full_pipeline.py --eval-only --checkpoint joint_checkpoint_ep50.pth --eval-output results_ep50/
```

Evaluates the 50-epoch checkpoint and saves results to `results_ep50/`

### Example 5: Train only joint model (with existing pretrained weights)

```bash
python full_pipeline.py --joint-only pretrained_generator.pth
```

Skips pretraining, trains only joint model from epoch 0

### Example 6: Quick evaluation without any training

```bash
python full_pipeline.py --eval-only
```

Uses checkpoint path from `config.py`: `checkpoints/joint_checkpoint_best.pth`

---

## Output Structure

### After Pretraining (`--pretrain-only`)

```
experiment/
├── sr_generator_pretrain_ep10.pth      # Checkpoint at epoch 10
├── sr_generator_pretrain_ep20.pth      # Checkpoint at epoch 20
├── ...
└── pretrained_generator.pth            # Final weights
```

### After Joint Training

```
experiment/
├── joint_checkpoint_ep5.pth            # Checkpoint at epoch 5
├── joint_checkpoint_ep10.pth           # Checkpoint at epoch 10
├── ...
└── joint_checkpoint_final.pth          # Final model (both gen + seg)
```

### After Evaluation

```
evaluation_output_32res/
├── image1.png                          # Super-resolved image
├── seg_image1.png                      # Raw segmentation mask
├── vis_image1.png                      # Colored segmentation visualization
├── image2.png
├── seg_image2.png
├── vis_image2.png
├── ...
├── final_results.txt                   # Metrics summary
├── bleeding_edge_error.png             # Boundary analysis
└── bf_score_curve.png                  # BF-score curve
```

---

## Configuration

Training parameters are defined in `config.py`:

- **pretraining_config**: Pretraining hyperparameters
- **training_config**: Joint training hyperparameters
- **evaluation_config**: Evaluation paths and settings

To modify defaults (learning rates, batch sizes, epochs, etc.), edit `config.py`:

```python
# Example: Change number of training epochs
training_config.num_epochs = 150

# Example: Change generator learning rate
training_config.generator_lr = 5e-5
```

---

## Model Checkpoint Format

Checkpoints are saved in a unified format compatible with `inference.py`:

```python
checkpoint = {
    'gen_state_dict': generator.state_dict(),
    'seg_state_dict': segmentor.state_dict(),
    'epoch': epoch_number
}
```

This format is automatically loaded by `load_models()` in `inference.py`.

---

## Evaluation Metrics

The evaluation phase computes:

- **mIoU** - Mean Intersection over Union (segmentation quality)
- **PA** - Pixel Accuracy
- **PA Class** - Per-class Pixel Accuracy
- **FWIoU** - Frequency Weighted IoU
- **Boundary metrics** - Bleeding edge error, BF-score

Results are saved to `final_results.txt` and plotted in PNG files.

---

## Tips & Troubleshooting

### Issue: "FileNotFoundError: Pretrained weights not found"

**Solution:** Make sure the pretrained file exists or run without `--skip-pretrain`:

```bash
# Wrong (file doesn't exist):
python full_pipeline.py --skip-pretrain --pretrained-path nonexistent.pth

# Correct (create pretrained first):
python full_pipeline.py --pretrain-only
python full_pipeline.py --skip-pretrain --pretrained-path pretrained_generator.pth
```

### Issue: Want to resume interrupted training

**Solution:** Use the last saved checkpoint:

```bash
python full_pipeline.py --joint-only pretrained_generator.pth
```

This will continue training from epoch 0 (create new checkpoint). To truly resume from epoch N, manually modify the training loop.

### Issue: Evaluation crashes with "shape mismatch"

**Solution:** Ensure test and ground truth images are the same size. Check `config.py`:

```python
format_config.high_resolution = 384  # Must match your data
```

### Issue: Out of memory (OOM) errors

**Solution:** Reduce batch size in `config.py`:

```python
training_config.batch_size = 1   # From default
pretraining_config.batch_size = 8  # From default 16
```

---

## Performance Monitoring

Monitor training progress:

- **Pretraining**: Losses displayed in progress bar
  - L_D (Discriminator Loss)
  - L_L1 (Pixel Loss)
  - L_VGG (VGG Perceptual Loss)
  - L_GAN (Adversarial Loss)

- **Joint Training**: Losses and metrics displayed per batch
  - L_D (Discriminator Loss)
  - L_2 (Pixel Loss)
  - L_CE (Cross-Entropy Loss)
  - L_Adv (Adversarial Loss)
  - L_ABL (Boundary Loss)

- **Evaluation**: Metrics printed and saved
  - mIoU, PA, PA_Class, FWIoU

---

## Next Steps

After training:

1. **Inference on new images**: Use `inference.py`
   ```bash
   python inference.py --input image.jpg --output output/ --checkpoint joint_checkpoint_final.pth
   ```

2. **Further fine-tuning**: Use trained checkpoint with modified config
   ```bash
   python full_pipeline.py --joint-only joint_checkpoint_final.pth
   ```

3. **Compare checkpoints**: Evaluate multiple checkpoints
   ```bash
   python full_pipeline.py --eval-only --checkpoint joint_checkpoint_ep50.pth --eval-output results_ep50/
   python full_pipeline.py --eval-only --checkpoint joint_checkpoint_final.pth --eval-output results_final/
   ```

---

## Additional Notes

- All checkpoints are saved in the working directory by default
- Evaluation resumes from checkpoint if interrupted (see `evaluation_checkpoint_path` in config)
- Feature extraction for losses uses RADIO model (pretrained, frozen)
- Segmentor backbone uses ResNet-50 with output stride 16
- Generator is ESRGAN-based for 4x super-resolution
