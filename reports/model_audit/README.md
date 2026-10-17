# FarmWatch AI — Model Audit Package

## Overview
This bundle contains the complete, evidence-based audit package for FarmWatch AI's visible crop condition classification models, developed for **GenHack V1.0 Problem Statement 12 (Predicting Crop Stress From Field Images)**.

This package is prepared for external technical review, independent model analysis, and release verification.

## Hardware & Training Environment
* **Platform:** ASUS Zenbook 14 OLED
* **Processor:** Intel Core Ultra 5 125H (14 cores / 18 threads)
* **Graphics:** Intel Arc Integrated Graphics (Probe verified; CPU execution fallback used)
* **RAM:** 16 GB LPDDR5X
* **Operating System:** Windows 11 Home 64-bit
* **Python Version:** 3.13.14
* **PyTorch Version:** 2.14.1+cpu
* **Random Seed:** 42

## Model Checkpoint Lineage & Verified Performance
1. **Selected Production Model: MobileNetV3-Small Transfer Learning**
   - Checkpoint: `models/mobilenetv3_small_transfer.pth` (16.21 MB, SHA-256: `3872120a313f7067fe6bdf968f3326afc6ff017337b399c21e8a988709f49a0c`)
   - Pretrained: ImageNet-1K backbone, customized 11-class head (`Linear(1024, 11)`)
   - **Test Accuracy:** **67.29%** (144 / 214 correct)
   - **Macro F1:** **66.52%** (0 zero-F1 classes)
   - **Average CPU Latency:** **11.09 ms**

2. **Baseline Experiment 2: Enhanced 11-Class Deep CNN (Scratch Baseline)**
   - Checkpoint: `models/enhanced_cnn.pth` (13.87 MB, SHA-256: `5a87f6091d4ddb15cf1135a2dab26d71d193ccaee11ce227ddd10455a4a3fb15`)
   - Trained from Scratch: 100% Kaiming Normal initialization, zero ImageNet weights
   - **Test Accuracy:** **27.10%** (58 / 214 correct)
   - **Macro F1:** **21.74%** (3 zero-F1 classes)

3. **Baseline Experiment 1: Custom 28-Class CNN (Initial Baseline)**
   - Checkpoint: `models/cnn_from_scratch.pth` (1.13 MB, SHA-256: `cf552c770636a1f1d5cd83019aae5ee55144605c8e07a9dece5ce3c449844b57`)
   - Trained from Scratch on raw 28 classes; 19.62% validation accuracy.

## Directory Structure
```
FarmWatch_Model_Audit_Bundle/
├── models/
│   ├── mobilenetv3_small_transfer.pth # Primary model (11 classes, 1.53M params, 16.2 MB)
│   ├── enhanced_cnn.pth             # Scratch baseline (11 classes, 1.2M params, 14.5 MB)
│   ├── cnn_from_scratch.pth         # Initial baseline (28 classes, 97K params, 1.18 MB)
│   ├── mobilenetv3_preprocess_config.json # 224x224 resolution, ImageNet normalization
│   ├── mobilenetv3_class_to_idx.json
│   ├── focused_class_to_idx.json    # 11-class mapping
│   └── focused_preprocess_config.json # 128x128 resolution
├── data_metadata/
│   ├── class_distribution.csv       # Split distribution per class (951 train, 200 val, 214 test)
│   ├── focused_train_manifest.json  # 951 train samples (SHA-256 verified)
│   ├── focused_val_manifest.json    # 200 validation samples
│   ├── focused_test_manifest.json   # 214 held-out test samples
│   └── focused_dataset_audit.json   # 11-class subset audit (1,365 images)
├── evaluation_evidence/
│   ├── mobilenetv3_test_metrics.json # Full test metrics for MobileNetV3 (67.29% acc, 66.52% F1)
│   ├── mobilenetv3_test_predictions.csv # 214 sample predictions with top-3 rankings
│   ├── mobilenetv3_test_class_probabilities.csv # Full 11-class probability vectors
│   ├── mobilenetv3_incorrect_predictions.csv # 70 error analysis rows
│   ├── validation_uncertainty_metrics.json # Validation calibration & 0.60 threshold metrics
│   ├── recalculated_test_metrics.json # Scratch CNN test metrics (27.10% acc)
│   ├── test_predictions.csv         # Scratch CNN sample predictions
│   ├── test_class_probabilities.csv # Scratch CNN probability vectors
│   ├── incorrect_predictions.csv    # Scratch CNN 156 errors
│   ├── checkpoint_analysis.json     # Technical analysis of all checkpoints
│   └── consistency_audit.json       # Audit reconciliation of stored vs recalculated vs UI
├── visual_error_analysis/
│   ├── visual_error_catalog.csv     # Metadata for error analysis samples
│   └── *.png                        # Grad-CAM diagnostic figures
├── source_code/
│   ├── train_mobilenetv3.py         # MobileNetV3 transfer learning training script
│   ├── evaluate_mobilenetv3.py      # MobileNetV3 test evaluation script
│   ├── custom_cnn.py                # Scratch CNN model definitions
│   ├── gradcam.py                   # PyTorch Grad-CAM hook implementation
│   ├── model_service.py             # Production inference engine wrapper
│   └── recommendations.py           # Safe non-chemical agronomic recommendations
├── requirements.txt                 # Exact python dependencies
└── FarmWatch_Model_Audit_Report.md  # Comprehensive technical audit report
```

## How to Verify
To verify the primary MobileNetV3-Small test evaluation:
```powershell
python -m src.training.evaluate_mobilenetv3
```
Verified results: 214 test images evaluated, 67.29% test accuracy (144 / 214), 66.52% macro F1.
