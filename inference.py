#!/usr/bin/env python3
"""
Inference Script for Vessel Segmentation
Run trained nnU-Net v2 model on new fluorescent microscopy images

Based on the methodology from:
"Automated diameter assessment in pial arteries in acute ischemic stroke"
Jills Jose, Amsterdam UMC, 2025
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from tqdm import tqdm
import json


def check_nnunet_environment():
    """Check if nnU-Net environment variables are set"""
    required_vars = ['nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results']
    
    for var in required_vars:
        if not os.getenv(var):
            print(f"Error: {var} environment variable not set")
            print(f"Please set: export {var}=/path/to/{var.lower()}")
            return False
    
    return True


def run_nnunet_predict(input_dir, output_dir, dataset_id=1, config='2d', fold=0, 
                      save_probabilities=False, disable_tta=False):
    """
    Run nnU-Net prediction using command line interface
    
    Args:
        input_dir: Directory containing test images
        output_dir: Directory for prediction outputs
        dataset_id: Dataset identifier
        config: Model configuration (2d, 3d_fullres, etc.)
        fold: Training fold to use
        save_probabilities: Save probability maps
        disable_tta: Disable test-time augmentation
    """
    
    cmd = [
        'nnUNetv2_predict',
        '-i', str(input_dir),
        '-o', str(output_dir),
        '-d', str(dataset_id),
        '-c', config,
        '-f', str(fold)
    ]
    
    if save_probabilities:
        cmd.append('--save_probabilities')
    
    if disable_tta:
        cmd.append('--disable_tta')
    
    print("Running nnU-Net prediction...")
    print("Command:", ' '.join(cmd))
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Prediction completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Prediction failed with return code {e.returncode}")
        print("Error output:", e.stderr)
        return False


def visualize_predictions(image_path, prediction_path, output_path=None):
    """
    Create visualization overlay of original image and prediction
    
    Args:
        image_path: Path to original NIfTI image
        prediction_path: Path to predicted segmentation mask
        output_path: Optional path to save visualization
    """
    
    # Load images
    img_nii = nib.load(image_path)
    pred_nii = nib.load(prediction_path)
    
    img_data = img_nii.get_fdata()
    pred_data = pred_nii.get_fdata()
    
    # Handle 3D images by taking middle slice or MIP
    if len(img_data.shape) == 3:
        img_slice = img_data[:, :, img_data.shape[2] // 2]
        pred_slice = pred_data[:, :, pred_data.shape[2] // 2]
    else:
        img_slice = img_data
        pred_slice = pred_data
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img_slice, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Prediction mask
    axes[1].imshow(pred_slice, cmap='jet')
    axes[1].set_title('Predicted Segmentation')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(img_slice, cmap='gray', alpha=0.7)
    axes[2].imshow(pred_slice, cmap='jet', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    
    plt.show()


def convert_predictions_to_binary(prediction_dir, threshold=0.5):
    """
    Convert probability maps to binary masks if needed
    
    Args:
        prediction_dir: Directory containing predictions
        threshold: Threshold for binarization
    """
    
    prediction_files = list(Path(prediction_dir).glob("*.nii.gz"))
    
    for pred_file in tqdm(prediction_files, desc="Converting to binary"):
        nii_img = nib.load(pred_file)
        data = nii_img.get_fdata()
        
        # Check if already binary
        unique_vals = np.unique(data)
        if len(unique_vals) <= 2 and np.all(np.isin(unique_vals, [0, 1])):
            continue
        
        # Binarize
        binary_data = (data > threshold).astype(np.uint8)
        
        # Save binary version
        binary_nii = nib.Nifti1Image(binary_data, nii_img.affine, nii_img.header)
        binary_path = pred_file.parent / f"{pred_file.stem}_binary.nii.gz"
        nib.save(binary_nii, binary_path)
        
        print(f"Binary mask saved: {binary_path}")


def generate_prediction_report(prediction_dir, original_dir=None):
    """
    Generate a summary report of predictions
    
    Args:
        prediction_dir: Directory containing predictions
        original_dir: Directory containing original images (optional)
    """
    
    prediction_files = sorted(list(Path(prediction_dir).glob("*.nii.gz")))
    
    if not prediction_files:
        print("No prediction files found!")
        return
    
    report = {
        "total_predictions": len(prediction_files),
        "predictions": []
    }
    
    print(f"\nPrediction Summary:")
    print(f"Total predictions: {len(prediction_files)}")
    print("-" * 50)
    
    for pred_file in prediction_files:
        nii_img = nib.load(pred_file)
        data = nii_img.get_fdata()
        
        # Calculate basic statistics
        vessel_pixels = np.sum(data > 0)
        total_pixels = data.size
        vessel_fraction = vessel_pixels / total_pixels
        
        pred_info = {
            "filename": pred_file.name,
            "shape": data.shape,
            "vessel_pixels": int(vessel_pixels),
            "vessel_fraction": float(vessel_fraction),
            "data_range": [float(data.min()), float(data.max())]
        }
        
        report["predictions"].append(pred_info)
        
        print(f"File: {pred_file.name}")
        print(f"  Shape: {data.shape}")
        print(f"  Vessel pixels: {vessel_pixels} ({vessel_fraction:.3%})")
        print(f"  Value range: [{data.min():.3f}, {data.max():.3f}]")
        print()
    
    # Save report
    report_path = Path(prediction_dir) / "prediction_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Report saved to: {report_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Run vessel segmentation inference")
    parser.add_argument("--input_dir", "-i", type=str, required=True,
                       help="Directory containing test images (NIfTI format)")
    parser.add_argument("--output_dir", "-o", type=str, required=True,
                       help="Directory for prediction outputs")
    parser.add_argument("--dataset_id", "-d", type=int, default=1,
                       help="Dataset ID used for training")
    parser.add_argument("--config", "-c", type=str, default="2d",
                       help="Model configuration")
    parser.add_argument("--fold", "-f", type=int, default=0,
                       help="Training fold to use")
    parser.add_argument("--save_probabilities", action="store_true",
                       help="Save probability maps")
    parser.add_argument("--disable_tta", action="store_true",
                       help="Disable test-time augmentation")
    parser.add_argument("--visualize", action="store_true",
                       help="Create visualization overlays")
    parser.add_argument("--binary_threshold", type=float, default=0.5,
                       help="Threshold for binary conversion")
    
    args = parser.parse_args()
    
    # Check environment
    if not check_nnunet_environment():
        return 1
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("Starting vessel segmentation inference...")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Dataset ID: {args.dataset_id}")
    print(f"Configuration: {args.config}")
    print(f"Fold: {args.fold}")
    
    # Run prediction
    success = run_nnunet_predict(
        args.input_dir, 
        args.output_dir,
        args.dataset_id,
        args.config,
        args.fold,
        args.save_probabilities,
        args.disable_tta
    )
    
    if not success:
        print("Inference failed!")
        return 1
    
    # Post-processing
    print("\nPost-processing predictions...")
    
    # Convert to binary if needed
    convert_predictions_to_binary(args.output_dir, args.binary_threshold)
    
    # Generate report
    generate_prediction_report(args.output_dir, args.input_dir)
    
    # Create visualizations
    if args.visualize:
        print("\nCreating visualizations...")
        input_files = sorted(list(Path(args.input_dir).glob("*.nii.gz")))
        pred_files = sorted(list(Path(args.output_dir).glob("*.nii.gz")))
        
        viz_dir = Path(args.output_dir) / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        for img_file, pred_file in zip(input_files, pred_files):
            viz_path = viz_dir / f"{pred_file.stem}_overlay.png"
            try:
                visualize_predictions(img_file, pred_file, viz_path)
            except Exception as e:
                print(f"Visualization failed for {pred_file.name}: {e}")
    
    print("\nInference completed successfully!")
    print(f"Results saved to: {args.output_dir}")
    print("\nNext steps:")
    print("1. Import binary masks into ImageJ/Fiji")
    print("2. Apply VasoMetrics plugin for diameter quantification")
    print("3. Use interactive correction tool for quality control")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
