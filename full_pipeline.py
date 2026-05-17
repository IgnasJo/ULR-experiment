"""
Full Training Pipeline
======================
Integrates pretraining, joint training, evaluation, and batch inference phases.
All console output is tee'd to outputs/run_log.txt by default.

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
    python full_pipeline.py --finetune path.pth     # Fine-tune from a full joint checkpoint
"""

import argparse
import os
import sys
import importlib.util


# ---------------------------------------------------------------------------
# Output root
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_OUTPUTS_DIR = os.path.join(_SCRIPT_DIR, "outputs")


# ---------------------------------------------------------------------------
# Tee: write stdout/stderr to both console and a log file
# ---------------------------------------------------------------------------

class _Tee:
    """Duplicates writes to *stream* into *file_path*, creating dirs as needed."""

    def __init__(self, stream, file_path: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self._stream = stream
        self._file = open(file_path, "a", encoding="utf-8", buffering=1)

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def fileno(self):
        return self._stream.fileno()

    def isatty(self):
        return self._stream.isatty()

    def close(self):
        self._file.close()


def _setup_logging():
    log_path = os.path.join(_OUTPUTS_DIR, "run_log.txt")
    sys.stdout = _Tee(sys.stdout, log_path)
    sys.stderr = _Tee(sys.stderr, log_path)
    print(f"[Pipeline] Logging to: {log_path}")


# ---------------------------------------------------------------------------
# Lazy imports (after logging is set up)
# ---------------------------------------------------------------------------

# Load training.py directly to avoid conflict with training/ package
spec = importlib.util.spec_from_file_location("training_module", os.path.join(_SCRIPT_DIR, "training.py"))
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
    out_dir = output_dir or os.path.join(_OUTPUTS_DIR, "batch_inference")
    
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
    print("[Pipeline] Optimizations enabled: AMP, torch.compile, ABL loss")
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
    _setup_logging()

    parser = argparse.ArgumentParser(description="Full Training Pipeline")
    parser.add_argument("--skip-pretrain", action="store_true", help="Skip pretraining, load existing weights")
    parser.add_argument("--pretrain-only", action="store_true", help="Only run pretraining")
    parser.add_argument("--pretrained-gen", type=str, default=get_checkpoint_path("pretrained_generator.pth"), help="Path for pretrained generator weights")
    parser.add_argument("--pretrained-disc", type=str, default=get_checkpoint_path("pretrained_discriminator.pth"), help="Path for pretrained discriminator weights")
    parser.add_argument("--finetune", type=str, default=None, help="Fine-tune from a full joint checkpoint (gen+seg loaded in-memory)")
    parser.add_argument("--joint-only", type=str, default=None, help="Run only joint training with specified generator weights")
    parser.add_argument("--train-epochs", type=int, default=None, help="Override training epochs for --joint-only and --finetune")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation after training")
    parser.add_argument("--eval-only", action="store_true", help="Run only evaluation (skip training)")
    parser.add_argument("--batch-inference", action="store_true", help="Run batch inference on separate test folder")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for evaluation/inference")
    parser.add_argument("--eval-output", type=str, default=None, help="Output directory for evaluation results")
    parser.add_argument("--test-dir", type=str, default=None, help="Test images directory for batch inference")

    args = parser.parse_args()

    # Enable speed optimizations globally for all pipeline modes.
    # ABL is opt-out: enabled by default but can be suppressed via ULR_USE_ABL=0.
    import config as _cfg
    _cfg.training_config.use_amp = True
    _cfg.training_config.use_compile = True
    _abl_env = os.environ.get('ULR_USE_ABL', '').strip().lower()
    if _abl_env not in ('false', '0', 'no'):
        _cfg.training_config.use_abl_loss = True

    if args.finetune:
        if args.train_epochs is not None:
            _cfg.training_config.num_epochs = args.train_epochs
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
        if args.train_epochs is not None:
            _cfg.training_config.num_epochs = args.train_epochs
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
