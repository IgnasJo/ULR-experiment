"""
Full Training Pipeline
======================
Integrates pretraining, joint training, evaluation, and batch inference phases.

Usage:
    python full_pipeline.py                         # Run full pipeline (pretrain → joint)
    python full_pipeline.py --evaluate              # Run full pipeline + evaluation
    python full_pipeline.py --skip-pretrain         # Skip pretraining, load existing weights
    python full_pipeline.py --pretrain-only         # Only run pretraining
    python full_pipeline.py --joint-only path.pth   # Only run joint training with weights
    python full_pipeline.py --eval-only             # Only run evaluation
    python full_pipeline.py --eval-only --checkpoint path.pth  # Evaluate specific checkpoint
    python full_pipeline.py --batch-inference       # Run batch inference on separate test folder
    python full_pipeline.py --batch-inference --checkpoint path.pth  # Batch inference with specific checkpoint
    python full_pipeline.py --overfit               # Overfit validation on ULR_overfit_data (pretrain→joint→eval)
    python full_pipeline.py --overfit --train-epochs 100 --target-miou 0.70  # Overfit with custom settings
"""

import argparse
import os
import sys
import importlib.util

# Load training.py directly to avoid conflict with training/ package
spec = importlib.util.spec_from_file_location("training_module", os.path.join(os.path.dirname(__file__), "training.py"))
training_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training_module)
train_joint = training_module.train_joint

# Import pretraining and evaluation normally (no package conflict)
from pretraining import pretrain_sr
from evaluation import evaluate
from batch_inference import batch_inference
from config import evaluation_config, checkpoint_config
from paths import get_checkpoint_path


def run_evaluation(checkpoint_path=None, output_dir=None):
    """
    Run evaluation on trained model.
    
    Args:
        checkpoint_path: Path to model checkpoint (uses config default if None)
        output_dir: Output directory for results (uses config default if None)
    """
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    ckpt = checkpoint_path or get_checkpoint_path(checkpoint_config.joint_filename)
    out_dir = output_dir or evaluation_config.output_dir
    
    print(f"Checkpoint: {ckpt}")
    print(f"Output dir: {out_dir}")
    
    evaluate(
        evaluation_config.test_dir,
        out_dir,
        ckpt,
        evaluation_config.test_dir_gt,
    )
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETED")
    print("="*60)


def run_batch_inference(checkpoint_path=None, test_dir=None, output_dir=None):
    """
    Run batch inference on a separate test folder (not the train/eval split).
    
    Args:
        checkpoint_path: Path to model checkpoint (uses config default if None)
        test_dir: Input test images directory (uses config default if None)
        output_dir: Output directory for results (uses config default if None)
    """
    print("\n" + "="*60)
    print("BATCH INFERENCE")
    print("="*60)
    
    from config import batch_inference_config
    
    ckpt = checkpoint_path or get_checkpoint_path(checkpoint_config.joint_filename)
    # Use batch_inference_config.rgb_dir if set, otherwise fall back to test_dir
    test = test_dir or batch_inference_config.rgb_dir or evaluation_config.test_dir
    out_dir = output_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_inference_output")
    
    print(f"Checkpoint: {ckpt}")
    print(f"Test dir:   {test}")
    print(f"Output dir: {out_dir}")
    
    batch_inference(test, out_dir, ckpt)
    
    print("\n" + "="*60)
    print("BATCH INFERENCE COMPLETED")
    print("="*60)


def run_full_pipeline(skip_pretrain=False, pretrain_only=False, pretrained_gen_path="pretrained_generator.pth", pretrained_disc_path="pretrained_discriminator.pth", run_eval=False):
    """
    Run the full training pipeline.
    
    Args:
        skip_pretrain: Skip pretraining and load existing weights
        pretrain_only: Only run pretraining phase
        pretrained_gen_path: Path to save/load pretrained generator weights
        pretrained_disc_path: Path to pretrained discriminator weights (from Phase 1)
        run_eval: Run evaluation after training
    """
    print("\n" + "="*60)
    print("FULL TRAINING PIPELINE")
    print("="*60)
    
    # Phase 1: Pretraining
    if not skip_pretrain:
        print("\n[Pipeline] Starting Phase 1: Pretraining...")
        pretrained_gen_path, pretrained_disc_path = pretrain_sr(
            save_path=pretrained_gen_path, 
            save_disc_path=pretrained_disc_path
        )
    else:
        print(f"\n[Pipeline] Skipping pretraining, will load from: {pretrained_gen_path}")
        if not os.path.exists(pretrained_gen_path):
            print(f"[Error] Pretrained weights not found at: {pretrained_gen_path}")
            print("[Error] Please run pretraining first or provide valid path.")
            return
    
    # Phase 2: Joint Training
    joint_checkpoint = get_checkpoint_path("joint_checkpoint_final.pth")
    if not pretrain_only:
        print("\n[Pipeline] Starting Phase 2: Joint Training...")
        # Pass both generator and discriminator pretrained weights
        # Discriminator loading handles channel mismatch (3ch -> 3+N ch) automatically
        disc_path = pretrained_disc_path if os.path.exists(pretrained_disc_path) else None
        if disc_path:
            print(f"[Pipeline] Will load pretrained discriminator from: {disc_path}")
        train_joint(pretrained_generator_path=pretrained_gen_path, pretrained_discriminator_path=disc_path)
        print(f"[Pipeline] Joint training completed. Checkpoint: {joint_checkpoint}")
    else:
        print("\n[Pipeline] Pretraining only mode - skipping joint training.")
    
    # Phase 3: Evaluation (optional)
    if run_eval:
        if not os.path.exists(joint_checkpoint):
            print(f"\n[Warning] Final checkpoint not found: {joint_checkpoint}")
            print("[Warning] Skipping evaluation.")
        else:
            print("\n[Pipeline] Starting Phase 3: Evaluation...")
            run_evaluation(checkpoint_path=joint_checkpoint)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED")
    print("="*60)


def run_overfit(args) -> int:
    """
    Run overfit validation inline using ULR_overfit_data.

    Mutates config objects directly (same-process, no subprocess). Returns an
    exit code: 0 = PASS, 1 = metrics.json missing, 2 = target mIoU not reached.
    """
    import json
    import shutil
    from datetime import datetime
    import torch
    import numpy as np
    import config as _cfg

    script_dir = os.path.dirname(os.path.abspath(__file__))
    overfit_data = os.path.join(script_dir, "ULR_overfit_data")
    test_images = os.path.join(overfit_data, "test_images")
    test_labels = os.path.join(overfit_data, "test_labels")

    for path, label in [(test_images, "Test images"), (test_labels, "Test labels")]:
        if not os.path.isdir(path):
            raise FileNotFoundError(f"[Overfit] {label} directory not found: {path}")

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = args.checkpoint_dir or os.path.join(script_dir, "checkpoints", f"overfit_{run_stamp}")
    eval_dir = args.eval_output or os.path.join(script_dir, "evaluation_output", f"overfit_{run_stamp}")

    if args.clean:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        shutil.rmtree(eval_dir, ignore_errors=True)

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Mutate config objects directly — SimpleNamespace is mutable and the same
    # objects are shared across all downstream imports (training.py, evaluation.py…).
    _cfg.checkpoint_config.base_dir = ckpt_dir
    _cfg.training_config.num_epochs = args.train_epochs
    _cfg.training_config.batch_size = args.batch_size
    _cfg.training_config.image_dir = test_images
    _cfg.training_config.mask_dir = test_labels
    _cfg.training_config.alpha = args.alpha
    _cfg.training_config.lambda_2 = args.lambda_fea
    _cfg.training_config.lambda_3 = args.lambda_adv
    _cfg.training_config.use_abl_loss = False  # disabled for overfit speed
    _cfg.pretraining_config.num_epochs = args.pretrain_epochs
    _cfg.pretraining_config.batch_size = args.batch_size
    _cfg.pretraining_config.hr_image_dir = test_images
    _cfg.evaluation_config.test_dir = test_images
    _cfg.evaluation_config.test_dir_gt = test_labels
    _cfg.evaluation_config.output_dir = eval_dir

    print("=" * 72)
    print("Overfit Validation")
    print("=" * 72)
    print(f"Data dir       : {overfit_data}")
    print(f"Pretrain epochs: {args.pretrain_epochs}")
    print(f"Train epochs   : {args.train_epochs}")
    print(f"Batch size     : {args.batch_size}")
    print(f"Checkpoint dir : {ckpt_dir}")
    print(f"Eval output    : {eval_dir}")
    print(f"Alpha          : {args.alpha}")
    print(f"Lambda fea     : {args.lambda_fea}")
    print(f"Lambda adv     : {args.lambda_adv}")
    print(f"Target mIoU    : {args.target_miou}")

    pretrained_gen = get_checkpoint_path("pretrained_generator.pth")
    pretrained_disc = get_checkpoint_path("pretrained_discriminator.pth")
    run_full_pipeline(run_eval=True, pretrained_gen_path=pretrained_gen, pretrained_disc_path=pretrained_disc)

    metrics_path = os.path.join(eval_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        print("[Overfit] WARNING: Pipeline finished but metrics.json was not found.")
        print(f"[Overfit] Expected: {metrics_path}")
        return 1

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    miou = float(metrics.get("mIoU", 0.0))
    print("\n" + "=" * 72)
    print("Overfit Validation Summary")
    print("=" * 72)
    print(f"mIoU         : {miou:.4f}")
    print(f"Target mIoU  : {args.target_miou:.4f}")
    print(f"Metrics file : {metrics_path}")

    if miou >= args.target_miou:
        print("[Overfit] PASS: Overfit target reached.")
        return 0

    print("[Overfit] WARNING: Pipeline completed but overfit target not reached.")
    print("[Overfit] Try increasing --train-epochs or lowering --target-miou.")
    return 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Training Pipeline")
    parser.add_argument("--skip-pretrain", action="store_true", help="Skip pretraining, load existing weights")
    parser.add_argument("--pretrain-only", action="store_true", help="Only run pretraining")
    parser.add_argument("--pretrained-gen", type=str, default=get_checkpoint_path("pretrained_generator.pth"), help="Path for pretrained generator weights")
    parser.add_argument("--pretrained-disc", type=str, default=get_checkpoint_path("pretrained_discriminator.pth"), help="Path for pretrained discriminator weights")
    parser.add_argument("--finetune", type=str, default=None, help="Fine-tune from a full joint checkpoint (gen+seg loaded in-memory)")
    parser.add_argument("--joint-only", type=str, default=None, help="Run only joint training with specified generator weights")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation after training")
    parser.add_argument("--eval-only", action="store_true", help="Run only evaluation (skip training)")
    parser.add_argument("--batch-inference", action="store_true", help="Run batch inference on separate test folder")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for evaluation/inference")
    parser.add_argument("--eval-output", type=str, default=None, help="Output directory for evaluation results")
    parser.add_argument("--test-dir", type=str, default=None, help="Test images directory for batch inference")
    # Overfit validation mode
    parser.add_argument("--overfit", action="store_true", help="Run overfit validation on ULR_overfit_data (pretrain→joint→eval)")
    parser.add_argument("--pretrain-epochs", type=int, default=5, help="[--overfit] SR pretraining epochs (default: 5)")
    parser.add_argument("--train-epochs", type=int, default=60, help="[--overfit] Joint training epochs (default: 60)")
    parser.add_argument("--batch-size", type=int, default=1, help="[--overfit] Training batch size (default: 1)")
    parser.add_argument("--seed", type=int, default=42, help="[--overfit] Deterministic seed (default: 42)")
    parser.add_argument("--target-miou", type=float, default=0.35, help="[--overfit] mIoU pass/fail threshold (default: 0.35)")
    parser.add_argument("--alpha", type=float, default=0.9, help="[--overfit] Segmentation loss weight (default: 0.9)")
    parser.add_argument("--lambda-adv", type=float, default=0.001, help="[--overfit] Adversarial loss weight (default: 0.001)")
    parser.add_argument("--lambda-fea", type=float, default=0.005, help="[--overfit] Feature loss weight (default: 0.005)")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="[--overfit] Checkpoint directory (default: checkpoints/overfit_<timestamp>)")
    parser.add_argument("--clean", action="store_true", help="[--overfit] Delete checkpoint/eval dirs before running")
    parser.add_argument("--allow-gpu", action="store_true", help="[--overfit] GPU is used if available (default); set CUDA_VISIBLE_DEVICES= to force CPU")

    args = parser.parse_args()

    if args.overfit:
        sys.exit(run_overfit(args))
    elif args.finetune:
        train_joint(pretrained_checkpoint_path=args.finetune)
    elif args.batch_inference:
        run_batch_inference(
            checkpoint_path=args.checkpoint,
            test_dir=args.test_dir,
            output_dir=args.eval_output
        )
    elif args.eval_only:
        run_evaluation(checkpoint_path=args.checkpoint, output_dir=args.eval_output)
    elif args.joint_only:
        # When running joint-only, also check for discriminator weights
        disc_path = args.pretrained_disc if os.path.exists(args.pretrained_disc) else None
        train_joint(pretrained_generator_path=args.joint_only, pretrained_discriminator_path=disc_path)
    else:
        run_full_pipeline(
            skip_pretrain=args.skip_pretrain,
            pretrain_only=args.pretrain_only,
            pretrained_gen_path=args.pretrained_gen,
            pretrained_disc_path=args.pretrained_disc,
            run_eval=args.evaluate
        )
