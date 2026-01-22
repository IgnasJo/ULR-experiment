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
from config import evaluation_config, get_checkpoint_path


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
    
    ckpt = checkpoint_path or get_checkpoint_path(evaluation_config.checkpoint_path)
    out_dir = output_dir or evaluation_config.evaluation_dir
    eval_ckpt = get_checkpoint_path("evaluation_checkpoint.pkl")
    
    print(f"Checkpoint: {ckpt}")
    print(f"Output dir: {out_dir}")
    
    evaluate(
        evaluation_config.test_dir,
        out_dir,
        ckpt,
        eval_ckpt,
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
    
    ckpt = checkpoint_path or get_checkpoint_path(evaluation_config.checkpoint_path)
    test = test_dir or evaluation_config.test_dir
    out_dir = output_dir or "batch_inference_output"
    
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Training Pipeline")
    parser.add_argument("--skip-pretrain", action="store_true", help="Skip pretraining, load existing weights")
    parser.add_argument("--pretrain-only", action="store_true", help="Only run pretraining")
    parser.add_argument("--pretrained-gen", type=str, default="pretrained_generator.pth", help="Path for pretrained generator weights")
    parser.add_argument("--pretrained-disc", type=str, default="pretrained_discriminator.pth", help="Path for pretrained discriminator weights")
    parser.add_argument("--joint-only", type=str, default=None, help="Run only joint training with specified generator weights")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation after training")
    parser.add_argument("--eval-only", action="store_true", help="Run only evaluation (skip training)")
    parser.add_argument("--batch-inference", action="store_true", help="Run batch inference on separate test folder")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for evaluation/inference")
    parser.add_argument("--eval-output", type=str, default=None, help="Output directory for evaluation results")
    parser.add_argument("--test-dir", type=str, default=None, help="Test images directory for batch inference")
    
    args = parser.parse_args()
    
    if args.batch_inference:
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
