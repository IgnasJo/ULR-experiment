# Full Training Pipeline - Quick Reference

Complete SR + Semantic Segmentation joint training pipeline with one-line commands.

## First-Time Setup

Install dependencies from the experiment directory:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify CUDA execution (recommended before training):

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import torch; x=torch.randn(256,256,device='cuda'); y=x@x; torch.cuda.synchronize(); print(y.shape)"
```

If the second command fails with `no kernel image is available`, your PyTorch build does not match your GPU architecture. Reinstall using `requirements.txt` in this folder.

## Quick Commands

| Command | Purpose |
|---------|---------|
| `python full_pipeline.py` | Pretrain → Joint Train (saves to `checkpoints/MM-DD/`) |
| `python full_pipeline.py --evaluate` | Full training + evaluation |
| `python full_pipeline.py --pretrain-only` | Pretrain only, save to `checkpoints/MM-DD/pretrained_generator.pth` |
| `python full_pipeline.py --skip-pretrain` | Skip pretrain, load from config default |
| `python full_pipeline.py --skip-pretrain --pretrained-path path.pth` | Skip pretrain, load from custom path |
| `python full_pipeline.py --joint-only path.pth` | Joint training only with pretrained weights |
| `python full_pipeline.py --eval-only` | Evaluate using default checkpoint from config |
| `python full_pipeline.py --eval-only --checkpoint path.pth` | Evaluate specific checkpoint |
| `python full_pipeline.py --eval-only --checkpoint path.pth --eval-output dir/` | Evaluate with custom output directory |
| `python full_pipeline.py --batch-inference` | Run batch inference (save images only, no metrics) |
| `python full_pipeline.py --batch-inference --checkpoint path.pth` | Batch inference with specific checkpoint |
| `python full_pipeline.py --batch-inference --test-dir path` | Batch inference with custom test folder |

## Checkpoint Structure

All checkpoints auto-save to `checkpoints/MM-DD/` (dated folder, MM=month, DD=day):
- **Pretraining**: `sr_generator_pretrain_ep{N}.pth`, `pretrained_generator.pth`
- **Joint Training**: `joint_checkpoint_ep{N}.pth`, `joint_checkpoint_final.pth`
- **Evaluation**: `evaluation_checkpoint.pkl`

## Configuration

Edit `config.py` for:
- Learning rates, batch sizes, epochs in `training_config`, `pretraining_config`
- Dataset paths, train/eval split in `dataset_config`
- Evaluation settings in `evaluation_config`

## Key Features

- **Phase 1**: Pretraining (VGG perceptual + GAN loss)
- **Phase 2**: Joint training (Generator + Segmentor with all losses)
- **Phase 3**: Evaluation (mIoU, PA, boundary metrics, visualizations)
- **Batch Inference**: Standalone inference (saves images only, no metrics)
- **Resume**: Automatic checkpointing on interrupt (training and evaluation)
- **Dated Folders**: Checkpoints organized by date (MM-DD format)

## Output

**Training**: Intermediate + final checkpoints in `checkpoints/MM-DD/`
**Evaluation**: Results in `evaluation_output/` (segmentation masks, SR images, metrics)
**Batch Inference**: Results in `batch_inference_output/`

## Tips

- First run: `python full_pipeline.py --evaluate` (full pipeline + eval)
- Resume with pretrained: `python full_pipeline.py --skip-pretrain --evaluate`
- Experiment with SR only: `python full_pipeline.py --pretrain-only`
- Compare checkpoints: Run `--eval-only` with different `--checkpoint` paths
- Check losses in progress bar during training

## Next Steps

After training:
1. Use `python inference.py --input image.jpg --checkpoint path.pth --output output/` for inference
2. Fine-tune with `python full_pipeline.py --joint-only checkpoint.pth`
3. Evaluate different checkpoints with `--eval-only --checkpoint path.pth`

## Overfit Test Entry Point

To validate full pretrain + joint train + evaluation on the tiny local dataset:

```bash
python overfit_test_pipeline.py --pretrain-epochs 1 --train-epochs 1 --target-miou 0.0 --allow-gpu
```

