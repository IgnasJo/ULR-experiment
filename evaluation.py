from inference import infer_and_save, load_models
from utils2.metrics import Evaluator
import os
import numpy as np
import pickle
from PIL import Image
from pathlib import Path
from config import evaluation_config

def preprocess_gt(img_path):
    img = Image.open(img_path).convert('L')
    if img.width > 384 or img.height > 384:
      left = (img.width -  384) // 2
      top  = (img.height - 384) // 2
      img = img.crop((left, top, left + 384, top + 384))
    return np.array(img).astype(np.int64)

def evaluate(test_folder, output_folder, checkpoint_path, evaluation_checkpoint_path, gt_folder):
    # 1. Setup Output and Checkpoint Paths
    os.makedirs(output_folder, exist_ok=True)
    resume_file = f"{evaluation_checkpoint_path}.pkl"
    
    # 2. Initialize or Resume State
    processed_files = set()
    
    if os.path.exists(resume_file):
        print(f"Found checkpoint! Resuming from {resume_file}...")
        try:
            with open(resume_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                evaluator = checkpoint_data['evaluator']
                processed_files = checkpoint_data['processed_files']
            print(f"   > Resuming with {len(processed_files)} images already processed.")
        except Exception as e:
            print(f"   [ERROR] Could not load checkpoint: {e}")
            print("   > Starting fresh instead.")
            evaluator = Evaluator(num_class=14)
    else:
        print("Starting fresh evaluation...")
        evaluator = Evaluator(num_class=14)

    # 3. Load Models (Always load fresh)
    print(f"   > Loading models from {checkpoint_path}...")
    gen_model, seg_model = load_models(checkpoint_path)

    gt_path_obj = Path(gt_folder)

    print(f"--- Starting Evaluation Loop ---")
    print(f"Test Folder: {os.path.abspath(test_folder)}")
    if gt_folder:
        print(f"GT Folder:   {os.path.abspath(gt_folder)}")
    print("-" * 30)

    # 4. Main Processing Loop
    file_list = sorted(os.listdir(test_folder))
    
    for i, fname in enumerate(file_list):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # SKIP logic: If we already did this file, skip it
        if fname in processed_files:
            continue

        img_path = os.path.join(test_folder, fname)

        # Run Inference
        # Assuming infer_and_save is defined in your environment
        infer_and_save(img_path, gen_model, seg_model, output_folder)

        # Evaluation Logic (Only if GT is provided)
        base_name = Path(fname).stem
        gt_matches = list(gt_path_obj.glob(f"{base_name}.*"))

        if gt_matches:
            actual_gt_path = gt_matches[0]
            print(f"\n[{i+1}/{len(file_list)}] Processing: {fname}")

            # Process Data
            # Assuming preprocess_gt is defined in your environment
            gt = preprocess_gt(actual_gt_path)

            # Find Prediction (Handling extension mismatch)
            pred_path = os.path.join(output_folder, f"seg_{fname}")
            if not os.path.exists(pred_path):
                    pred_path = os.path.join(output_folder, f"seg_{base_name}.png")

            seg_pred = np.array(Image.open(pred_path))

            # Safety Check
            if gt.shape != seg_pred.shape:
                print(f"  [ERROR] Shape mismatch! GT: {gt.shape}, Pred: {seg_pred.shape}")
                continue

            # Update Running Average (Main Evaluator)
            evaluator.add_batch(gt, seg_pred)

            # Single Image Metrics for logging
            single_eval = Evaluator(num_class=14)
            single_eval.add_batch(gt, seg_pred)
            print(f"  > Current Img : mIoU: {single_eval.Mean_Intersection_over_Union():.4f} | PA: {single_eval.Pixel_Accuracy():.4f}")
            print(f"  > Running Avg : mIoU: {evaluator.Mean_Intersection_over_Union():.4f} | PA: {evaluator.Pixel_Accuracy():.4f}")
        else:
            print(f"Warning: No ground truth found for {base_name}")

        # 5. Save Checkpoint (After every file)
        processed_files.add(fname)
        
        # We only need to pickle if we have an evaluator (meaning we have GT)
        # If no GT, we just track processed files to avoid re-inference
        checkpoint_data = {
            'evaluator': evaluator,
            'processed_files': processed_files
        }
        
        with open(resume_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)

    # 6. Final Metrics
    if evaluator:
        print("\n" + "=" * 30)
        print("FINAL METRICS")
        print("=" * 30)
        print(f"mIoU:      {evaluator.Mean_Intersection_over_Union():.4f}")
        print(f"PA:        {evaluator.Pixel_Accuracy():.4f}")
        print(f"PA Class:  {evaluator.Pixel_Accuracy_Class():.4f}")
        print(f"FWIoU:     {evaluator.Frequency_Weighted_Intersection_over_Union():.4f}")
        
        # Optional: Save final text report to Drive as well
        with open(os.path.join(output_folder, "final_results.txt"), "w") as f:
            f.write(f"mIoU: {evaluator.Mean_Intersection_over_Union():.4f}\n")
            f.write(f"PA:   {evaluator.Pixel_Accuracy():.4f}\n")

evaluate(
  evaluation_config.test_dir,
  evaluation_config.evaluation_dir,
  evaluation_config.checkpoint_path,
  evaluation_config.evaluation_checkpoint_path,
  evaluation_config.test_dir_gt,
)