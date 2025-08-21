#!/usr/bin/env python3
"""
Data Conversion Script for Vessel Segmentation
Converts fluorescent microscopy z-stacks to NIfTI format for nnU-Net v2 training

Based on the methodology from:
"Automated diameter assessment in pial arteries in acute ischemic stroke"
Jills Jose, Amsterdam UMC, 2025
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path
import argparse
import tifffile
from tqdm import tqdm
import json


def create_mip_from_stack(image_stack):
    """
    Create Maximum Intensity Projection from z-stack
    
    Args:
        image_stack: 3D numpy array (z, y, x)
    
    Returns:
        2D numpy array representing MIP
    """
    return np.max(image_stack, axis=0)


def convert_tiff_to_nifti(input_path, output_path, create_mip=True):
    """
    Convert TIFF z-stack to NIfTI format
    
    Args:
        input_path: Path to input TIFF file
        output_path: Path for output NIfTI file
        create_mip: Whether to create maximum intensity projection
    """
    try:
        # Load TIFF stack
        image_stack = tifffile.imread(input_path)
        
        if create_mip and len(image_stack.shape) == 3:
            # Create MIP for 3D stacks (>300 z-planes as mentioned in thesis)
            image_data = create_mip_from_stack(image_stack)
        else:
            image_data = image_stack
        
        # Ensure proper data type
        if image_data.dtype != np.float32:
            image_data = image_data.astype(np.float32)
        
        # Create NIfTI image
        nii_img = nib.Nifti1Image(image_data, affine=np.eye(4))
        
        # Save NIfTI file
        nib.save(nii_img, output_path)
        
        return True
        
    except Exception as e:
        print(f"Error converting {input_path}: {str(e)}")
        return False


def prepare_nnunet_dataset(input_dir, output_dir, dataset_name="Dataset001_VesselSeg"):
    """
    Prepare dataset in nnU-Net format
    
    Args:
        input_dir: Directory containing raw images and masks
        output_dir: nnU-Net raw data directory
        dataset_name: Name for the dataset
    """
    
    dataset_path = Path(output_dir) / dataset_name
    images_tr_path = dataset_path / "imagesTr"
    labels_tr_path = dataset_path / "labelsTr"
    
    # Create directories
    images_tr_path.mkdir(parents=True, exist_ok=True)
    labels_tr_path.mkdir(parents=True, exist_ok=True)
    
    input_path = Path(input_dir)
    
    # Find image and mask files
    image_files = sorted(list(input_path.glob("*_image.tif*")))
    mask_files = sorted(list(input_path.glob("*_mask.tif*")))
    
    if len(image_files) != len(mask_files):
        print(f"Warning: Number of images ({len(image_files)}) != number of masks ({len(mask_files)})")
    
    successful_conversions = 0
    
    for i, (img_file, mask_file) in enumerate(zip(image_files, mask_files), 1):
        # nnU-Net naming convention
        case_id = f"case_{i:03d}"
        
        img_output = images_tr_path / f"{case_id}_0000.nii.gz"
        mask_output = labels_tr_path / f"{case_id}.nii.gz"
        
        print(f"Converting {img_file.name} -> {img_output.name}")
        
        # Convert image
        if convert_tiff_to_nifti(img_file, img_output):
            # Convert mask
            if convert_tiff_to_nifti(mask_file, mask_output, create_mip=True):
                successful_conversions += 1
            else:
                print(f"Failed to convert mask: {mask_file}")
        else:
            print(f"Failed to convert image: {img_file}")
    
    # Create dataset.json
    dataset_json = {
        "channel_names": {
            "0": "fluorescent"
        },
        "labels": {
            "background": 0,
            "vessel": 1
        },
        "numTraining": successful_conversions,
        "file_ending": ".nii.gz",
        "dataset_name": dataset_name,
        "reference": "Jose et al. Automated diameter assessment in pial arteries in acute ischemic stroke. Amsterdam UMC, 2025",
        "description": "Fluorescent microscopy vessel segmentation dataset"
    }
    
    with open(dataset_path / "dataset.json", 'w') as f:
        json.dump(dataset_json, f, indent=4)
    
    print(f"\nDataset preparation complete!")
    print(f"Successfully converted: {successful_conversions} cases")
    print(f"Dataset location: {dataset_path}")
    print(f"Ready for nnU-Net preprocessing")


def main():
    parser = argparse.ArgumentParser(description="Convert fluorescent microscopy data for nnU-Net training")
    parser.add_argument("--input_dir", "-i", type=str, required=True,
                       help="Directory containing TIFF images and masks")
    parser.add_argument("--output_dir", "-o", type=str, required=True,
                       help="nnU-Net raw data directory")
    parser.add_argument("--dataset_name", "-n", type=str, default="Dataset001_VesselSeg",
                       help="Dataset name for nnU-Net")
    parser.add_argument("--no_mip", action="store_true",
                       help="Skip maximum intensity projection for z-stacks")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return
    
    print("Starting data conversion for vessel segmentation...")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Dataset name: {args.dataset_name}")
    
    prepare_nnunet_dataset(args.input_dir, args.output_dir, args.dataset_name)


if __name__ == "__main__":
    main()
