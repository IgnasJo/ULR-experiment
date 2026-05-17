## Setup

This directory contains the experiment code used in this workspace.

```bash
cd experiment
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### GPU compatibility note (new NVIDIA Blackwell GPUs)

`requirements.txt` pins a CUDA 12.8 nightly PyTorch stack that supports Blackwell (`sm_120`) GPUs.

Quick verification:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import torch; x=torch.randn(256,256,device='cuda'); y=x@x; torch.cuda.synchronize(); print(y.shape)"
```

### Datasets
- Paths are configured through environment variables in `config.py` (`ULR_TRAIN_RGB`, `ULR_TRAIN_LABEL`, `ULR_VAL_RGB`, `ULR_VAL_LABEL`, `ULR_TEST_RGB`, `ULR_TEST_LABEL`).
- Source of SunRGBD dataset: ["Training on RGB data for 13 classes"](https://github.com/ankurhanda/sunrgbd-meta-data?tab=readme-ov-file#training-on-rgb-data-for-13-classes)

## Weight
For convenience, our pre-trained ULR2SS model can be downloaded directly here:
[ULR2SS_Weight](https://drive.google.com/file/d/1QhA2XHYmiajAhTJt9WqJocHGk6vEq3Tj/view)

## Train pipeline usage
```bash
python full_pipeline.py
python full_pipeline.py --skip-pretrain
python full_pipeline.py --pretrain-only
python full_pipeline.py --joint-only weights.pth
```

## End-to-end overfit test

Run a minimal full pipeline (pretrain + joint train + eval) on `ULR_overfit_data`:

```bash
python full_pipeline.py --overfit --pretrain-epochs 1 --train-epochs 1 --target-miou 0.0 --allow-gpu
```

For an overfit-oriented sanity pass, increase epochs (for example `--train-epochs 20`).

## Demo Test
```bash
python inference.py \
  --input  /home/user/ULR2SS/images/rgb_demo1.png \      # path to image/folder
  --output /home/user/ULR2SS/images/output \  # path to save results
  --checkpoint /home/user/ULR2SS/joint_checkpoint_best.pth \ # ckpt path
```