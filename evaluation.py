"""
Evaluation script using DataLoader approach (consistent with training.py).
Uses the same dataset/dataloader pattern for reproducibility.

Computes all metrics from the ULR2SS paper (Table 4):
- RGB Reconstruction Fidelity: PSNR, SSIM, LPIPS, FID
- Segmentation Accuracy: ARI, Covering, BF
- Semantic Accuracy: mIoU, mAcc
"""
import torch
import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from esrgan import Generator
from modeling.deeplab import DeepLab
from utils2.metrics import Evaluator
from utils2.sr_metrics import SRMetricsAccumulator
from training.dataloder import create_eval_loader
from config import evaluation_config, format_config, model_config, checkpoint_config
from paths import get_checkpoint_path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def strip_module_state_dict(sd):
    from collections import OrderedDict
    new_sd = OrderedDict()
    for k, v in sd.items():
        # Strip 'module.' (DataParallel) and '_orig_mod.' (torch.compile) prefixes
        k = k.replace('_orig_mod.', '').replace('module.', '')
        new_sd[k] = v
    return new_sd


def load_models(checkpoint_path):
    """Load generator and segmentor from joint checkpoint."""
    gen = Generator(format_config.img_channels).to(device)
    seg = DeepLab(num_classes=model_config.num_classes, backbone='resnet', output_stride=16,
                  sync_bn=None, freeze_bn=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    gen.load_state_dict(strip_module_state_dict(ckpt['gen_state_dict']))
    seg.load_state_dict(strip_module_state_dict(ckpt['seg_state_dict']))
    gen.eval()
    seg.eval()
    return gen, seg


def save_outputs(sr_tensor, seg_pred, filename, output_folder):
    """Save super-resolved image and segmentation mask (same as inference.py)."""
    os.makedirs(output_folder, exist_ok=True)
    
    # Save SR image (same postprocess as inference.py)
    sr_img = sr_tensor.squeeze(0).cpu().detach()
    sr_pil = transforms.ToPILImage()(sr_img)
    sr_pil.save(os.path.join(output_folder, filename))
    
    # Save raw segmentation mask (class indices 0-13 as uint8)
    seg_np = seg_pred.squeeze().cpu().numpy().astype(np.uint8)
    Image.fromarray(seg_np).save(os.path.join(output_folder, f"seg_{filename}"))


def evaluate(test_folder, output_folder, checkpoint_path, gt_folder):
    """
    Evaluate model using DataLoader approach (consistent with training/pretraining).
    Computes all metrics from the ULR2SS paper Table 4.
    """
    # 1. Setup
    os.makedirs(output_folder, exist_ok=True)
    
    # 2. Initialize State (always fresh)
    evaluator = Evaluator(num_class=model_config.num_classes)
    sr_accumulator = SRMetricsAccumulator(device=str(device), compute_lpips=True)
    print("Starting fresh evaluation...")

    # 3. Load Models
    print(f"   > Loading models from {checkpoint_path}...")
    gen_model, seg_model = load_models(checkpoint_path)

    # 4. Create DataLoader (with HR images for SR metrics)
    print(f"   > Creating evaluation DataLoader...")
    eval_loader = create_eval_loader(test_folder, gt_folder, batch_size=1, include_hr=True)
    
    print(f"--- Starting Evaluation Loop ---")
    print(f"Test Folder: {os.path.abspath(test_folder)}")
    print(f"GT Folder:   {os.path.abspath(gt_folder)}")
    print(f"Total samples: {len(eval_loader.dataset)}")
    print("-" * 30)

    # 5. Evaluation Loop (DataLoader approach - same pattern as training.py)
    tbar = tqdm(enumerate(eval_loader), total=len(eval_loader), desc="Evaluating")
    
    for i, (lr_img, gt_mask, hr_img, filenames) in tbar:
        filename = filenames[0]  # batch_size=1
        
        # Move to device
        lr_img = lr_img.to(device)
        gt_mask = gt_mask.to(device)
        
        # Inference
        with torch.no_grad():
            sr_img = gen_model(lr_img)
            seg_logits = seg_model(sr_img)
            seg_pred = torch.argmax(seg_logits, dim=1)
        
        # Save outputs (SR image + raw segmentation mask)
        save_outputs(sr_img, seg_pred, filename, output_folder)
        
        # SR quality metrics (PSNR, SSIM, LPIPS, FID)
        sr_accumulator.update(sr_img.cpu(), hr_img)
        
        # Convert to numpy for segmentation metrics
        gt_np = gt_mask.squeeze().cpu().numpy()
        pred_np = seg_pred.squeeze().cpu().numpy()
        
        # Update segmentation metrics (boundary + region + ARI/Covering)
        evaluator.add_batch_with_boundaries(gt_np, pred_np)
        
        # Update progress bar
        current_miou = evaluator.Mean_Intersection_over_Union()
        current_macc = evaluator.Pixel_Accuracy_Class()
        tbar.set_postfix(mIoU=f"{current_miou:.4f}", mAcc=f"{current_macc:.4f}")

    # 6. Final Metrics
    print("\n" + "=" * 50)
    print("FINAL METRICS (ULR2SS Paper Table 4)")
    print("=" * 50)
    
    # Get all segmentation metrics at once
    seg_metrics = evaluator.get_all_metrics(tau=2, alpha=1.0)
    
    # Get SR quality metrics
    sr_metrics = sr_accumulator.summary(compute_fid_score=True)
    
    # Merge into single dict
    all_metrics = {**seg_metrics, **sr_metrics}
    
    # Print RGB Reconstruction Fidelity
    print("\n--- RGB Reconstruction Fidelity ---")
    print(f"PSNR:   {all_metrics.get('PSNR', float('nan')):.4f}")
    print(f"SSIM:   {all_metrics.get('SSIM', float('nan')):.4f}")
    print(f"LPIPS:  {all_metrics.get('LPIPS', float('nan')):.4f}")
    print(f"FID:    {all_metrics.get('FID', float('nan')):.4f}")
    
    # Print Segmentation Accuracy
    print("\n--- Segmentation Accuracy ---")
    print(f"ARI:      {all_metrics['ARI']:.4f}")
    print(f"Covering: {all_metrics['Covering']:.4f}")
    print(f"BF:       {all_metrics['Boundary_F1']:.4f}")
    
    # Print Semantic Accuracy
    print("\n--- Semantic Accuracy ---")
    print(f"mIoU:  {all_metrics['mIoU']:.4f}")
    print(f"mAcc:  {all_metrics['mAcc']:.4f}")
    
    # Print additional metrics
    print("\n--- Additional Metrics ---")
    print(f"PA:        {all_metrics['Pixel_Accuracy']:.4f}")
    print(f"FWIoU:     {all_metrics['FWIoU']:.4f}")
    print(f"Symmetric Boundary Dice:  {all_metrics['Symmetric_Boundary_Dice']:.4f}")
    print(f"Hausdorff Distance:       {all_metrics['Hausdorff_Distance']:.4f}")
    print(f"Mean Hausdorff Distance:  {all_metrics['Mean_Hausdorff_Distance']:.4f}")
    print(f"Average Surface Distance: {all_metrics['Average_Surface_Distance']:.4f}")
    
    # Save final text report
    with open(os.path.join(output_folder, "final_results.txt"), "w") as f:
        f.write("=== RGB Reconstruction Fidelity ===\n")
        f.write(f"PSNR:  {all_metrics.get('PSNR', float('nan')):.4f}\n")
        f.write(f"SSIM:  {all_metrics.get('SSIM', float('nan')):.4f}\n")
        f.write(f"LPIPS: {all_metrics.get('LPIPS', float('nan')):.4f}\n")
        f.write(f"FID:   {all_metrics.get('FID', float('nan')):.4f}\n")
        f.write("\n=== Segmentation Accuracy ===\n")
        f.write(f"ARI:      {all_metrics['ARI']:.4f}\n")
        f.write(f"Covering: {all_metrics['Covering']:.4f}\n")
        f.write(f"BF:       {all_metrics['Boundary_F1']:.4f}\n")
        f.write("\n=== Semantic Accuracy ===\n")
        f.write(f"mIoU: {all_metrics['mIoU']:.4f}\n")
        f.write(f"mAcc: {all_metrics['mAcc']:.4f}\n")
        f.write("\n=== Additional Metrics ===\n")
        f.write(f"PA:   {all_metrics['Pixel_Accuracy']:.4f}\n")
        f.write(f"FWIoU: {all_metrics['FWIoU']:.4f}\n")
        f.write(f"Symmetric Boundary Dice: {all_metrics['Symmetric_Boundary_Dice']:.4f}\n")
        f.write(f"Hausdorff Distance: {all_metrics['Hausdorff_Distance']:.4f}\n")
        f.write(f"Mean Hausdorff Distance: {all_metrics['Mean_Hausdorff_Distance']:.4f}\n")
        f.write(f"Average Surface Distance: {all_metrics['Average_Surface_Distance']:.4f}\n")

    # Save JSON for programmatic consumption (e.g. overfit_test_pipeline.py)
    with open(os.path.join(output_folder, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)


if __name__ == "__main__":
    evaluate(
        evaluation_config.test_dir,
        evaluation_config.output_dir,
        get_checkpoint_path(checkpoint_config.joint_filename),
        evaluation_config.test_dir_gt,
    )
