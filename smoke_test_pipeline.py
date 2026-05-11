"""
Smoke test runner for full_pipeline.py.

Purpose:
- Run the full training + evaluation pipeline end to end
- Use a tiny local dataset (experiment/test_data) for train/val/test
- Encourage intentional overfitting to verify the stack works without runtime errors

Usage examples:
    python smoke_test_pipeline.py
    python smoke_test_pipeline.py --pretrain-epochs 3 --train-epochs 20
    python smoke_test_pipeline.py --target-miou 0.60
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full pipeline smoke test on tiny dataset to validate end-to-end execution."
    )
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=3,
        help="Number of SR pretraining epochs (default: 3)",
    )
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=25,
        help="Number of joint training epochs (default: 25)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Training batch size (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--target-miou",
        type=float,
        default=0.70,
        help="Soft target mIoU for overfit check on the same train/test set (default: 0.70)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Checkpoint directory override (default: checkpoints/smoke_overfit_<timestamp>)",
    )
    parser.add_argument(
        "--eval-output",
        type=str,
        default=None,
        help="Evaluation output directory override (default: evaluation_output/smoke_overfit_<timestamp>)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete checkpoint/eval directories before running.",
    )
    parser.add_argument(
        "--allow-gpu",
        action="store_true",
        help="Allow CUDA usage. By default this smoke runner forces CPU for compatibility.",
    )
    return parser.parse_args()


def require_dir(path: Path, label: str) -> None:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"{label} directory not found: {path}")


def run_command(command: list[str], env: dict[str, str], cwd: Path) -> None:
    print("\n[Smoke] Running command:")
    print(" ", " ".join(command))
    result = subprocess.run(command, cwd=str(cwd), env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}")


def main() -> int:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    full_pipeline = script_dir / "full_pipeline.py"

    # Reuse the same tiny split for train/val/test to intentionally make overfitting easy.
    test_images = script_dir / "test_data" / "test_images"
    test_labels = script_dir / "test_data" / "test_labels"
    require_dir(script_dir, "Experiment root")
    require_dir(test_images, "Test images")
    require_dir(test_labels, "Test labels")
    if not full_pipeline.exists():
        raise FileNotFoundError(f"full_pipeline.py not found at: {full_pipeline}")

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else script_dir / "checkpoints" / f"smoke_overfit_{run_stamp}"
    eval_output = Path(args.eval_output) if args.eval_output else script_dir / "evaluation_output" / f"smoke_overfit_{run_stamp}"

    if args.clean:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        shutil.rmtree(eval_output, ignore_errors=True)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    eval_output.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "ULR_TRAIN_RGB": str(test_images),
            "ULR_TRAIN_LABEL": str(test_labels),
            "ULR_VAL_RGB": str(test_images),
            "ULR_VAL_LABEL": str(test_labels),
            "ULR_TEST_RGB": str(test_images),
            "ULR_TEST_LABEL": str(test_labels),
            "ULR_PRETRAIN_EPOCHS": str(args.pretrain_epochs),
            "ULR_TRAIN_EPOCHS": str(args.train_epochs),
            "ULR_BATCH_SIZE": str(args.batch_size),
            "ULR_PRETRAIN_BATCH_SIZE": str(args.batch_size),
            "ULR_SEED": str(args.seed),
            # Disable ABL by default for faster smoke validation.
            "ULR_USE_ABL": "False",
            "ULR_CHECKPOINT_DIR": str(checkpoint_dir),
            "ULR_EVAL_OUTPUT_DIR": str(eval_output),
        }
    )

    if not args.allow_gpu:
        # Avoid CUDA architecture/runtime mismatches during smoke validation.
        env["CUDA_VISIBLE_DEVICES"] = ""

    print("=" * 72)
    print("Smoke Test Pipeline")
    print("=" * 72)
    print(f"Experiment dir : {script_dir}")
    print(f"Train images   : {test_images}")
    print(f"Train labels   : {test_labels}")
    print(f"Pretrain epochs: {args.pretrain_epochs}")
    print(f"Train epochs   : {args.train_epochs}")
    print(f"Checkpoint dir : {checkpoint_dir}")
    print(f"Eval output    : {eval_output}")
    print(f"GPU enabled    : {args.allow_gpu}")

    run_command(
        [sys.executable, str(full_pipeline), "--evaluate", "--eval-output", str(eval_output)],
        env=env,
        cwd=script_dir,
    )

    metrics_path = eval_output / "metrics.json"
    if not metrics_path.exists():
        print("[Smoke] WARNING: Pipeline finished but metrics.json was not found.")
        print(f"[Smoke] Expected: {metrics_path}")
        return 1

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    miou = float(metrics.get("mIoU", 0.0))
    print("\n" + "=" * 72)
    print("Smoke Test Summary")
    print("=" * 72)
    print(f"mIoU         : {miou:.4f}")
    print(f"Target mIoU  : {args.target_miou:.4f}")
    print(f"Metrics file : {metrics_path}")

    if miou >= args.target_miou:
        print("[Smoke] PASS: Overfit target reached and no runtime errors occurred.")
        return 0

    print("[Smoke] WARNING: Pipeline completed without runtime errors, but overfit target was not reached.")
    print("[Smoke] Try increasing --train-epochs or lowering --target-miou.")
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[Smoke] ERROR: {exc}")
        raise SystemExit(1)