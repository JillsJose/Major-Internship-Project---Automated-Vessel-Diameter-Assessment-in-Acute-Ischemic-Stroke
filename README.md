# Automated Vessel Diameter Assessment in Acute Ischemic Stroke

This repository contains the implementation and documentation for automated cerebral vessel diameter quantification using nnU-Net v2, developed as part of a Master's thesis at Amsterdam UMC.

## Project Overview

This project addresses the challenge of manually measuring vessel diameters in fluorescent microscopy images of cerebral vasculature during acute ischemic stroke. The automated pipeline achieves:
- **Dice similarity coefficient**: >0.92 
- **Correlation with manual measurements**: r = 0.926
- **Processing time reduction**: ~75% compared to manual methods

## Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- 32GB+ system RAM
- Dataset used can be found internally for AMC users on: L:\Basic\divh\BMEPH\Ed\STUDENTS\Jills Jose\Dataset used - nnUNetv2
### 1. Environment Setup

```bash
# Create conda environment
conda create -n vessel_seg python=3.10
conda activate vessel_seg

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install nnU-Net v2
pip install nnunetv2
```

### 2. nnU-Net Configuration

```bash
# Set environment variables (add to ~/.bashrc for persistence)
export nnUNet_raw="/path/to/your/nnUNet_raw"
export nnUNet_preprocessed="/path/to/your/nnUNet_preprocessed"  
export nnUNet_results="/path/to/your/nnUNet_results"

# Create directories
mkdir -p $nnUNet_raw $nnUNet_preprocessed $nnUNet_results
```

### 3. Data Preparation

Your data should follow nnU-Net naming conventions:

```
nnUNet_raw/Dataset001_VesselSeg/
├── dataset.json
├── imagesTr/
│   ├── case_001_0000.nii.gz
│   ├── case_002_0000.nii.gz
│   └── ...
└── labelsTr/
    ├── case_001.nii.gz
    ├── case_002.nii.gz
    └── ...
```

Create `dataset.json`:
```json
{
    "channel_names": {
        "0": "fluorescent"
    },
    "labels": {
        "background": 0,
        "vessel": 1
    },
    "numTraining": 42,
    "file_ending": ".nii.gz"
}
```

### 4. Training Pipeline

```bash
# Dataset preprocessing
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity

# Training (2D configuration)
nnUNetv2_train 1 2d 0 --npz

# Training typically takes 12-24 hours on Tesla V100
```

### 5. Inference

```bash
# Run prediction on test images
nnUNetv2_predict -i /path/to/test/images -o /path/to/output -d 1 -c 2d -f 0
```

## Data Format and Preprocessing

### Image Preparation
1. **Fluorescent microscopy z-stacks** → Maximum Intensity Projection (MIP)
2. **Convert to NIfTI format** using provided conversion script
3. **Manual annotation** in ImageJ/Fiji focusing on MCA branches
4. **Time investment**: ~4-5 hours per image for annotation

### Key Parameters Used
- **Patch size**: 2048×2048 pixels
- **Batch size**: 2 images  
- **Learning rate**: 0.01 (adaptive)
- **Loss function**: Combined Dice + Cross-Entropy
- **Training duration**: ~1000 epochs

## Diameter Quantification

After obtaining segmentation masks, vessel diameters are measured using VasoMetrics:

### VasoMetrics Integration
1. Import binary mask into ImageJ
2. Apply VasoMetrics plugin for FWHM-based diameter calculation
3. Use interactive post-processing tool for manual corrections

### Interactive Correction Tool
- **Purpose**: Fix diameter measurement errors in complex regions
- **Usage**: Adjust FWHM boundaries using slider interface
- **Critical for**: Bifurcations and low-contrast regions

## Results Summary

| Metric | Value |
|--------|-------|
| Median Dice Score | 0.92 |
| Sensitivity | 0.94 |
| Specificity | 0.997 |
| Diameter Correlation | r = 0.926 |
| Mean Bias | -0.67 μm |
| Limits of Agreement | -8.27 to 6.92 μm |

## Current Limitations and Future Work

### Known Issues
- **Bifurcation accuracy**: Limited by 2D MIP approach
- **Complex vessel overlap**: Challenging for automated segmentation  
- **Manual annotation bottleneck**: 4-5 hours per image

### Recommended Next Steps
1. **3D Implementation**: Process full volumetric data instead of MIP
2. **Dataset expansion**: Target 200+ annotated volumes
3. **VasoMetrics improvements**: Better bifurcation handling
4. **Multi-modal training**: Include 2-photon microscopy data

## Hardware Requirements

### Training
- **GPU**: NVIDIA Tesla V100 (32GB) or equivalent
- **RAM**: 96GB (used in study)
- **Storage**: ~500GB for full dataset
- **Training time**: 24 hours with HPC resources

### Inference
- **GPU**: 16GB+ VRAM recommended
- **Processing time**: ~5-10 minutes per image

## Dataset Access

The training dataset (42 fluorescent microscopy volumes) is available upon request:
- **Contact**: j.jose@amsterdamumc.nl , m.r.khokhar@amsterdamumc.nl  
- **Internal Users**: L:\Basic\divh\BMEPH\Ed\STUDENTS\Jills Jose\Dataset used - nnUNetv2
- **Format**: NIfTI files with binary masks
- **Ethics**: Amsterdam UMC approved protocols

## File Structure
```
vessel-diameter-assessment/
├── README.md
├── requirements.txt
├── scripts/
│   ├── data_conversion.py
│   ├── training_pipeline.sh
│   └── inference.py
├── tools/
│   └── interactive_correction.py
└── docs/
    ├── installation_guide.md
    └── troubleshooting.md
```

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch size or patch size
2. **Preprocessing fails**: Check NIfTI file integrity
3. **Poor segmentation**: Verify manual annotations quality
4. **Training crashes**: Monitor system resources and logs

### Performance Optimization
- Use mixed precision training: `--enable_amp`
- Monitor GPU utilization during training
- Ensure sufficient system RAM for data loading


## License

This project is released under MIT License. See LICENSE file for details.

## Acknowledgments

- **Supervisor**: Dr. Ed Van Bavel (Professor of Vascular Biophysics)
- **Daily Supervisor**: Moeed Khokhar (PhD Candidate)  
- **Examiner**: Dr. Anton Feenstra (Associate Professor, Bioinformatics)
- **Institution**: Amsterdam UMC, Vrije Universiteit Amsterdam

## Contact

For questions or collaboration:
- **Email**: j.jose@amsterdamumc.nl
- **Institution**: Amsterdam UMC
- **Program**: MSc Bioinformatics and Systems Biology (UvA-VU)

---

*This repository provides a foundation for automated vessel diameter assessment. Future development should focus on 3D implementation and improved handling of complex vessel morphologies.*
