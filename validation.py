"""
Per-epoch validation with boundary-aware metrics.

Runs a deterministic, augmentation-free evaluation on a fixed dataset subset
after each training epoch. Metrics are logged to the console and appended to
``val_history.json`` in the active checkpoint directory. The best checkpoint
(by BF₁) is saved as ``joint_checkpoint_best.pth``.

Environment variables (all optional):
    ULR_VAL_RGB          — validation image directory
    ULR_VAL_LABEL        — validation mask directory
    ULR_VAL_SUBSET_SIZE  — number of samples per validation run (0 = all)
    ULR_VAL_FREQ         — validate every N epochs (default: 1)
    ULR_VAL_TAU          — pixel tolerance τ for BF₁ (default: 2)
"""

import os
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import (
    checkpoint_config,
    model_config,
    perf_config,
    training_config,
    validation_config,
)
from paths import get_checkpoint_name, get_checkpoint_path
from training.dataloder import create_eval_loader
from utils2.metrics import Evaluator


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_state_dict(sd: dict) -> dict:
    """Remove ``_orig_mod.`` (torch.compile) and ``module.`` (DataParallel) prefixes."""
    return {
        k.replace("_orig_mod.", "").replace("module.", ""): v
        for k, v in sd.items()
    }


def _append_history(path: str, entry: dict) -> None:
    """Load existing JSON array from *path*, append *entry*, write back."""
    history = []
    if os.path.exists(path):
        try:
            with open(path) as fh:
                history = json.load(fh)
        except (json.JSONDecodeError, IOError):
            history = []
    history.append(entry)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(history, fh, indent=2)


def _save_best_checkpoint(
    generator,
    segmentor,
    epoch: int,
    metrics: dict,
) -> None:
    """Save the best-BF₁ checkpoint with stripped/unwrapped state dicts."""
    best_path = get_checkpoint_path(get_checkpoint_name("joint", is_best=True))
    torch.save(
        {
            "gen_state_dict": _strip_state_dict(generator.state_dict()),
            "seg_state_dict": _strip_state_dict(segmentor.state_dict()),
            "epoch": epoch,
            "bf1": metrics["Boundary_F1"],
            "miou": metrics["mIoU"],
        },
        best_path,
    )
    print(f"[Val] ★ New best BF₁={metrics['Boundary_F1']:.4f} — saved to {best_path}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_val_loader() -> DataLoader:
    """
    Build a deterministic, augmentation-free validation DataLoader.

    The dataset is loaded with the evaluation transform (centre-crop →
    bicubic downsample → tensor → normalise; no random augmentations).
    Files are iterated in sorted order so the subset is always the same.

    If ``ULR_VAL_RGB`` / ``ULR_VAL_LABEL`` are not set the loader falls back
    to the training paths and emits a visible warning — validation metrics
    will then reflect *training-set* performance rather than held-out data.
    """
    img_dir = validation_config.val_image_dir
    msk_dir = validation_config.val_mask_dir

    if img_dir is None or msk_dir is None:
        img_dir = training_config.image_dir
        msk_dir = training_config.mask_dir
        print(
            "\n[Validation] WARNING: ULR_VAL_RGB / ULR_VAL_LABEL are not set. "
            "Validation is using TRAINING data — metrics may overestimate "
            "generalisation performance.\n"
        )

    # create_eval_loader: shuffle=False, centre-crop only, no HR tensors
    loader = create_eval_loader(img_dir, msk_dir, batch_size=1, include_hr=False)

    n = validation_config.val_subset_size
    if n and 0 < n < len(loader.dataset):
        # Use the first N samples (sorted filenames → reproducible without a seed)
        subset = Subset(loader.dataset, list(range(n)))
        loader = DataLoader(
            subset,
            batch_size=1,
            shuffle=False,
            num_workers=perf_config.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=perf_config.num_workers > 0,
            prefetch_factor=2 if perf_config.num_workers > 0 else None,
        )

    return loader


def run_epoch_validation(
    generator,
    segmentor,
    epoch: int,
    best_bf1: float,
    device: torch.device,
    val_loader: DataLoader = None,
    history_path: str = None,
) -> tuple:
    """
    Run boundary-aware validation after a training epoch.

    The function is a no-op (returns ``({}, best_bf1)``) when the epoch does
    not satisfy the configured frequency (``ULR_VAL_FREQ``).

    Metrics computed
    ~~~~~~~~~~~~~~~~
    * ``Boundary_F1`` (BF₁, τ pixels) — **primary model-selection criterion**
    * ``mIoU``, ``mAcc``, ``Pixel_Accuracy``
    * ``Symmetric_Boundary_Dice``
    * ``ARI`` (Adjusted Rand Index)
    * ``Covering`` (Segmentation Covering)

    Expensive metrics (PSNR / SSIM / LPIPS / FID / Hausdorff distance) are
    intentionally omitted to keep per-epoch overhead low.

    Args:
        generator:    SR generator model (any train/eval state is accepted).
        segmentor:    Segmentor model (any train/eval state is accepted).
        epoch:        Current epoch, 1-indexed.
        best_bf1:     Best BF₁ seen so far; used for checkpoint selection.
        device:       Device on which to run inference.
        val_loader:   Pre-built DataLoader reused across epochs.  When *None*
                      a new loader is built (less efficient).
        history_path: Path for the JSON history file.  Defaults to
                      ``<checkpoint_dir>/val_history.json``.

    Returns:
        ``(metrics_dict, new_best_bf1)`` — metrics for *epoch* and the
        (possibly updated) best BF₁ value.
    """
    if epoch % validation_config.val_freq != 0:
        return {}, best_bf1

    # Preserve original training state so we can restore it after validation
    gen_training = generator.training
    seg_training = segmentor.training

    generator.eval()
    segmentor.eval()

    if val_loader is None:
        val_loader = create_val_loader()

    evaluator = Evaluator(num_class=model_config.num_classes)
    tau = validation_config.val_tau

    with torch.no_grad():
        for batch in tqdm(
            val_loader, desc=f"[Val ep{epoch:03d}]", leave=False, unit="img"
        ):
            # EvaluationDataset (include_hr=False) → (lr_img, gt_mask, filename)
            lr_img, gt_mask, _filename = batch

            lr_img = lr_img.to(device)
            sr_img = generator(lr_img).float()      # upcast from bfloat16 if needed
            seg_logits = segmentor(sr_img)
            seg_pred = torch.argmax(seg_logits, dim=1)

            gt_np = gt_mask.squeeze().cpu().numpy()
            pred_np = seg_pred.squeeze().cpu().numpy()
            evaluator.add_batch_with_boundaries(gt_np, pred_np)

    # ------------------------------------------------------------------
    # Boundary-aware metrics (Hausdorff / ASD skipped for speed)
    # ------------------------------------------------------------------
    bf1      = evaluator.Boundary_F1(tau=tau)
    sym_dice = evaluator.Symmetric_Boundary_Dice(tau=tau)
    miou     = evaluator.Mean_Intersection_over_Union()
    macc     = evaluator.Pixel_Accuracy_Class()
    pa       = evaluator.Pixel_Accuracy()
    ari      = evaluator.Adjusted_Rand_Index()
    covering = evaluator.Segmentation_Covering()

    metrics = {
        "epoch":                  epoch,
        "Boundary_F1":            float(bf1),
        "mIoU":                   float(miou),
        "mAcc":                   float(macc),
        "Pixel_Accuracy":         float(pa),
        "Symmetric_Boundary_Dice": float(sym_dice),
        "ARI":                    float(ari),
        "Covering":               float(covering),
        "val_subset_size":        len(val_loader.dataset),
        "tau":                    tau,
    }

    # Console summary
    print(
        f"\n[Val ep{epoch:03d}] "
        f"BF₁={bf1:.4f} | mIoU={miou:.4f} | mAcc={macc:.4f} | "
        f"ARI={ari:.4f} | Covering={covering:.4f} | SymDice={sym_dice:.4f} "
        f"[n={len(val_loader.dataset)}, τ={tau}px]"
    )

    # Persist to JSON history
    if history_path is None:
        history_path = os.path.join(checkpoint_config.base_dir, "val_history.json")
    _append_history(history_path, metrics)

    # Best-model selection: save checkpoint when BF₁ improves
    if bf1 > best_bf1:
        best_bf1 = bf1
        _save_best_checkpoint(generator, segmentor, epoch, metrics)

    # Restore original training modes
    generator.train(gen_training)
    segmentor.train(seg_training)

    return metrics, best_bf1
