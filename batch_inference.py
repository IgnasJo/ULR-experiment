from inference import infer_and_save, load_models
import os
from config import evaluation_config, get_checkpoint_path


def batch_inference(test_folder, output_folder, checkpoint_path):
    """
    Run batch inference on a folder of images (save SR images and segmentation masks only).
    
    Args:
        test_folder: Path to input images
        output_folder: Path to save outputs
        checkpoint_path: Path to model checkpoint
    """
    # Setup Output
    os.makedirs(output_folder, exist_ok=True)

    # Load Models
    print(f"Loading models from {checkpoint_path}...")
    gen_model, seg_model = load_models(checkpoint_path)

    print(f"--- Starting Batch Inference ---")
    print(f"Input Folder:  {os.path.abspath(test_folder)}")
    print(f"Output Folder: {os.path.abspath(output_folder)}")
    print("-" * 30)

    # Process all images
    file_list = sorted(os.listdir(test_folder))
    image_count = 0
    
    for i, fname in enumerate(file_list):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(test_folder, fname)
        
        print(f"[{i+1}/{len(file_list)}] Processing: {fname}")
        
        # Run Inference (saves SR image and segmentation mask)
        infer_and_save(img_path, gen_model, seg_model, output_folder)
        image_count += 1

    print("\n" + "=" * 30)
    print(f"BATCH INFERENCE COMPLETE")
    print(f"Processed {image_count} images")
    print(f"Results saved to: {os.path.abspath(output_folder)}")
    print("=" * 30)


if __name__ == "__main__":
    batch_inference(
        evaluation_config.test_dir,
        "batch_inference_output",
        get_checkpoint_path(evaluation_config.checkpoint_path),
    )