#!/bin/bash

# Training Pipeline for Vessel Segmentation using nnU-Net v2
# Based on methodology from Jills Jose, Amsterdam UMC, 2025
# 
# Usage: ./training_pipeline.sh [DATASET_ID] [FOLD]
# Example: ./training_pipeline.sh 1 0

set -e  # Exit on any error

# Default parameters
DATASET_ID=${1:-1}
FOLD=${2:-0}
CONFIG="2d"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== nnU-Net v2 Training Pipeline for Vessel Segmentation ===${NC}"
echo "Dataset ID: $DATASET_ID"
echo "Configuration: $CONFIG"
echo "Fold: $FOLD"
echo ""

# Check environment variables
if [[ -z "${nnUNet_raw}" ]]; then
    echo -e "${RED}Error: nnUNet_raw environment variable not set${NC}"
    echo "Please set: export nnUNet_raw=/path/to/nnUNet_raw"
    exit 1
fi

if [[ -z "${nnUNet_preprocessed}" ]]; then
    echo -e "${RED}Error: nnUNet_preprocessed environment variable not set${NC}"
    echo "Please set: export nnUNet_preprocessed=/path/to/nnUNet_preprocessed"
    exit 1
fi

if [[ -z "${nnUNet_results}" ]]; then
    echo -e "${RED}Error: nnUNet_results environment variable not set${NC}"
    echo "Please set: export nnUNet_results=/path/to/nnUNet_results"
    exit 1
fi

echo -e "${GREEN}Environment variables check: PASSED${NC}"
echo "nnUNet_raw: $nnUNet_raw"
echo "nnUNet_preprocessed: $nnUNet_preprocessed"
echo "nnUNet_results: $nnUNet_results"
echo ""

# Check if dataset exists
DATASET_PATH="${nnUNet_raw}/Dataset$(printf '%03d' $DATASET_ID)_VesselSeg"
if [[ ! -d "$DATASET_PATH" ]]; then
    echo -e "${RED}Error: Dataset not found at $DATASET_PATH${NC}"
    echo "Please run data_conversion.py first to prepare your dataset"
    exit 1
fi

echo -e "${GREEN}Dataset found: $DATASET_PATH${NC}"

# Check for GPU
if ! nvidia-smi > /dev/null 2>&1; then
    echo -e "${YELLOW}Warning: nvidia-smi not found. Training may be slow without GPU${NC}"
else
    echo -e "${GREEN}GPU check: PASSED${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi
echo ""

# Step 1: Dataset integrity verification and preprocessing
echo -e "${YELLOW}Step 1: Dataset preprocessing and planning...${NC}"
echo "This may take several minutes for large datasets"
nnUNetv2_plan_and_preprocess -d $DATASET_ID --verify_dataset_integrity

if [[ $? -eq 0 ]]; then
    echo -e "${GREEN}Preprocessing completed successfully${NC}"
else
    echo -e "${RED}Preprocessing failed${NC}"
    exit 1
fi
echo ""

# Step 2: Training
echo -e "${YELLOW}Step 2: Starting model training...${NC}"
echo "Configuration: $CONFIG"
echo "Fold: $FOLD"
echo "Expected training time: 12-24 hours on Tesla V100"
echo ""

# Training parameters used in the study:
# - Patch size: 2048×2048 pixels
# - Batch size: 2 images  
# - Learning rate: 0.01 (adaptive)
# - Loss function: Combined Dice + Cross-Entropy
# - Training duration: ~1000 epochs

START_TIME=$(date +%s)

nnUNetv2_train $DATASET_ID $CONFIG $FOLD --npz

TRAINING_EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

if [[ $TRAINING_EXIT_CODE -eq 0 ]]; then
    echo ""
    echo -e "${GREEN}Training completed successfully!${NC}"
    echo "Training time: ${HOURS}h ${MINUTES}m"
    
    # Show training results location
    RESULTS_PATH="${nnUNet_results}/Dataset$(printf '%03d' $DATASET_ID)_VesselSeg/nnUNetTrainer__nnUNetPlans__${CONFIG}/fold_${FOLD}"
    echo "Results saved to: $RESULTS_PATH"
    
    # Check if model files exist
    if [[ -f "${RESULTS_PATH}/checkpoint_final.pth" ]]; then
        echo -e "${GREEN}Final checkpoint found: checkpoint_final.pth${NC}"
    fi
    
    # Display validation metrics if available
    if [[ -f "${RESULTS_PATH}/training_log_*.txt" ]]; then
        echo ""
        echo "Final validation metrics:"
        tail -n 5 "${RESULTS_PATH}"/training_log_*.txt | grep -E "(Dice|loss)" || true
    fi
    
else
    echo -e "${RED}Training failed with exit code: $TRAINING_EXIT_CODE${NC}"
    echo "Check the logs for details"
    exit 1
fi

echo ""
echo -e "${GREEN}=== Training Pipeline Complete ===${NC}"
echo ""
echo "Next steps:"
echo "1. Run inference on test images using:"
echo "   nnUNetv2_predict -i /path/to/test/images -o /path/to/output -d $DATASET_ID -c $CONFIG -f $FOLD"
echo ""
echo "2. Evaluate segmentation quality and extract vessel diameters using VasoMetrics"
echo ""
echo "For questions or issues, contact: j.jose@amsterdamumc.nl"
