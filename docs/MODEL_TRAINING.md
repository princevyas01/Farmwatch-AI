# FarmWatch AI: Model Training, Audit & Transfer-Learning Benchmark Guide

**Problem Statement:** GenHack V1.0 — PS 12: Predicting Crop Stress From Field Images  
**Deliverable Owner:** Model Training & Machine Learning Engineering Lead  
**Host Target:** ASUS Zenbook 14 OLED | Intel Core Ultra 5 125H (14 cores) | 16 GB RAM | Windows 11  

---

## 1. Executive Summary & Honest Scope

FarmWatch AI provides an automated visual estimation system for **visible crop and leaf conditions** across natural field-collected imagery.

### Strict Scope Boundaries:
- **What it detects:** Visible foliar symptoms (leaf spots, blights, rusts, powdery mildew) and healthy foliage states across 11 field-represented classes.
- **What it does NOT detect:** The model does not detect unannotated subterranean stress such as drought, nitrogen/nutrient deficiency, heat stress, or root rot.
- **Agronomic Disclaimer:** Predictions are uncalibrated empirical statistical estimates. The system never prescribes chemical pesticide dosages and directs high-impact field management decisions to certified extension personnel.
- **Pretrained vs. Scratch Honesty:** 
  - The custom baseline (`CustomCropCNN`) and enhanced architecture (`EnhancedCropCNN`) were trained **100% from random scratch initialization** (Kaiming Normal).
  - The selected production model (`MobileNetV3-Small`) was trained using **transfer learning from ImageNet-1K pretrained weights** (`MobileNet_V3_Small_Weights.DEFAULT`) with a two-stage fine-tuning schedule.

---

## 2. Dataset Acquisition, Provenance & Reproducibility

### Benchmark Dataset
- **Dataset:** PlantDoc: A Dataset for Visual Plant Disease Detection in Natural Field Images
- **Citations:** Kayal et al., *IEEE Access* (2023) / *ACM CoDS-COMAD* (2020), DOI: `10.1109/ACCESS.2023.3263042`.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **How to Obtain:** Available on GitHub at `https://github.com/pratikkayal/PlantDoc-Dataset` or via the official Zenodo archive. Clone or extract the repository so that the `train/` and `test/` folders are placed at `data/raw/plantdoc/`.
- **Configurable Dataset Root:** For portability across Windows/Linux or external storage drives, the environment variable `FARMWATCH_DATASET_ROOT` is supported. For example:
  ```powershell
  $env:FARMWATCH_DATASET_ROOT = "D:\datasets\PlantDoc"
  ```
  All dataset manifests use cross-platform repository-relative POSIX paths (`data/raw/plantdoc/train/...` and `train/...`) and resolve automatically through [`src/utils/dataset_paths.py`](file:///c:/Users/Prince/Documents/antigravity/dazzling-bohr/src/utils/dataset_paths.py).

### Split Provenance Reconciliation:
The evaluation dataset is a **custom held-out stratified partition** of the combined PlantDoc benchmark images, constructed to ensure well-represented classes ($\ge 100$ images) have adequate sample support across splits:
- **Train Split:** **951** unique images (889 from PlantDoc `train/`, 62 from PlantDoc `test/`)
- **Validation Split:** **200** unique images (182 from PlantDoc `train/`, 18 from PlantDoc `test/`)
- **Held-Out Test Split:** **214** unique images (202 from PlantDoc `train/`, 12 from PlantDoc `test/`)
- **Total:** **1,365** unique field images ($951 + 200 + 214 = 1,365$)
- **Important Provenance Truth:** Because images were pooled and stratified across classes, **this is not the official PlantDoc test directory split**. It is an independent custom held-out test split.
- **Zero Data Leakage:** Grouped SHA-256 hashes confirm 0 overlapping image hashes across train, validation, and test sets.

---

## 3. Systematic Model Progression

### Experiment 1: Initial Scratch Baseline (`CustomCropCNN`)
- **Classes:** All 28 raw PlantDoc classes.
- **Splits:** Train = 1,778, Val = 372, Test = 410.
- **Architecture:** 3-block Custom CNN (97,084 parameters), trained from random scratch initialization for 8 epochs.
- **Results:** 18.29% test accuracy, 11.17% macro F1. Suffered severe class starvation (13 out of 28 classes had 0.00% F1 due to classes having as few as 2 images).

### Experiment 2: Enhanced Scratch Baseline (`EnhancedCropCNN`)
- **Classes:** 11 well-represented classes ($\ge 100$ images each in PlantDoc: 1,365 field images total).
- **Splits:** Train = 951, Val = 200, Test = 214 (grouped by image SHA-256, 0 leakage).
- **Architecture:** 4-block deep CNN with progressive double convolutions (Conv-BN-ReLU-Conv-BN-ReLU-MaxPool), 32 to 256 channels, dual dropout (1,207,787 parameters).
- **Training Strategy:** 100% Random initialization (Kaiming Normal), class-weighted CrossEntropyLoss, 0.05 label smoothing, cosine annealing learning rate schedule, data augmentation. Zero ImageNet weights.
- **Results:** 27.10% test accuracy (58 / 214), 21.74% macro F1. Reduced zero-F1 classes from 13 down to 3.

### Experiment 3: Primary Selected Model (`MobileNetV3-Small Transfer Learning`)
- **Classes & Splits:** Exact identical 11-class subset and identical 951 train / 200 val / 214 test partitions.
- **Architecture:** Torchvision `mobilenet_v3_small` with ImageNet-1K pretrained weights. Classification head replaced with:
  ```python
  nn.Sequential(
      nn.Linear(576, 1024),
      nn.Hardswish(),
      nn.Dropout(p=0.2),
      nn.Linear(1024, 11)
  )
  ```
- **Total Parameters:** **1,529,131** (100% of model)
- **Two-Stage Fine-Tuning Strategy & Parameter Breakdown:**
  1. **Stage 1 (Classifier Head Warm-up, Epochs 1–4):**
     - Backbone features frozen (`model.features.requires_grad = False`).
     - **Trainable parameters:** **602,123** (39.38% of total) in the classifier head (`classifier[0]` Linear(576, 1024): 590,848; `classifier[3]` Linear(1024, 11): 11,275).
     - **Frozen parameters:** **927,008** (60.62% of total) across all 13 backbone feature blocks (`features[0]` through `features[12]`).
     - Optimizer: AdamW ($lr = 10^{-3}$, weight decay $= 10^{-4}$), CosineAnnealingLR.
  2. **Stage 2 (Top Backbone Fine-Tuning, Epochs 5–14):**
     - Top feature blocks (`features[9:]`: blocks 9, 10, 11, 12 with 736,488 parameters) unfrozen.
     - Early feature blocks (`features[0:9]` with 190,520 parameters) **remained frozen throughout all of training** to preserve low-level generic edge/texture filters.
     - **Trainable parameters:** **1,338,611** (87.54% of total).
     - **Frozen parameters:** **190,520** (12.46% of total).
     - Differential learning rates: $lr = 10^{-4}$ for top backbone blocks, $lr = 5 \times 10^{-4}$ for classifier head, CosineAnnealingLR.
- **Checkpoint Selection Rule:** Highest Validation Macro F1 on the 200-image validation partition. Checkpoint selected at Epoch 11 (Val Acc = 69.00%, Val Macro F1 = 67.80%).
- **Evaluation Disclosure:** The 214-image test set was evaluated during development as a held-out test partition; it was not used for hyperparameter or epoch selection.

---

## 4. Head-to-Head Comparative Benchmark

| Metric | Phase 1 (28-Class Baseline) | Phase 2 (11-Class Scratch CNN) | Phase 3 (MobileNetV3 Transfer Learning) | Delta (pp) vs Scratch | Relative Gain (%) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Architecture** | `CustomCropCNN` (3 blocks) | `EnhancedCropCNN` (4 blocks) | `MobileNetV3-Small` | — | — |
| **Weights** | Random (Kaiming Normal) | Random (Kaiming Normal) | Pretrained (ImageNet-1K) | Transfer Learning | — |
| **Total Parameters** | 97,084 | 1,207,787 | **1,529,131** | +321,344 | +26.6% |
| **Held-Out Test Accuracy** | 18.29% | 27.10% (58 / 214) | **67.29% (144 / 214)** | **+40.19 pp** | **+148.3%** |
| **Balanced Accuracy** | 14.19% | 27.10% | **65.98%** | **+38.88 pp** | **+143.5%** |
| **Macro Precision** | 13.14% | 24.03% | **69.10%** | **+45.07 pp** | **+187.6%** |
| **Macro Recall** | 14.19% | 27.10% | **65.98%** | **+38.88 pp** | **+143.5%** |
| **Macro F1-Score** | 11.17% | 21.74% | **66.52%** | **+44.78 pp** | **+206.0%** |
| **Weighted F1-Score** | 17.51% | 22.02% | **67.57%** | **+45.55 pp** | **+206.9%** |
| **Zero-F1 Classes** | 13 of 28 (46.4%) | 3 of 11 (27.3%) | **0 of 11 (0.0%)** | **-3 classes** | **100% eliminated** |
| **Model Forward Latency (CPU)** | 4.90 ms | 9.80 ms | **15.34 ms** | +5.54 ms | — |
| **Full API Latency (no CAM)** | 45.20 ms | 52.10 ms | **63.02 ms** | +10.92 ms | — |
| **Full API Latency (with CAM)** | 145.00 ms | 172.50 ms | **198.59 ms** | +26.09 ms | — |
| **Test Image Count** | 410 images | 214 images | **214 images** | Identical test set | — |

---

## 5. Detailed MobileNetV3 Per-Class Breakdown (214 Test Samples)

| Class Name | Test Support | Precision | Recall | F1-Score | Scratch F1 | F1 Delta (pp) | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Squash Powdery mildew leaf** | 20 | 90.00% | 90.00% | **90.00%** | 54.24% | +35.76 pp | High Accuracy |
| **Peach leaf** | 18 | 93.33% | 77.78% | **84.85%** | 45.61% | +39.24 pp | Distinct Folio Contours |
| **Corn leaf blight** | 29 | 83.33% | 86.21% | **84.75%** | 21.82% | +62.93 pp | High Sensitivity |
| **Corn rust leaf** | 18 | 77.78% | 77.78% | **77.78%** | 40.00% | +37.78 pp | Pustule Discrimination |
| **Raspberry leaf** | 19 | 91.67% | 57.89% | **70.97%** | 31.82% | +39.15 pp | Serrated Margins |
| **Blueberry leaf** | 18 | 60.00% | 83.33% | **69.77%** | 0.00% | **+69.77 pp** | Starved Class Revived |
| **Tomato leaf late blight** | 18 | 75.00% | 50.00% | **60.00%** | 30.77% | +29.23 pp | Moderate |
| **Potato leaf late blight** | 16 | 50.00% | 56.25% | **52.94%** | 6.90% | +46.04 pp | Moderate |
| **Tomato Septoria leaf spot** | 23 | 45.16% | 60.87% | **51.85%** | 8.00% | +43.85 pp | Small Spot Recognition |
| **Tomato leaf bacterial spot** | 17 | 53.85% | 41.18% | **46.67%** | 0.00% | **+46.67 pp** | Starved Class Revived |
| **Potato leaf early blight** | 18 | 40.00% | 44.44% | **42.11%** | 0.00% | **+42.11 pp** | Starved Class Revived |
| **Macro Average** | **214** | **69.10%** | **65.98%** | **66.52%** | **21.74%** | **+44.78 pp** | All 11 Classes Positive |

---

## 6. Uncertainty Handling & Heuristic Threshold Performance

- **Softmax Nature:** Softmax probabilities are uncalibrated empirical scores rather than true Bayesian posterior probabilities. No temperature scaling or Platt scaling was applied.
- **Validation Set (200 images) Calibration Audit:**
  - Overall accuracy on validation set: **69.00%** (138 / 200 correct)
  - Mean confidence: **0.7050**, Median: **0.7153**, Std: **0.2332**
  - **Heuristic 0.60 Cutoff Coverage:** **67.00%** (134 of 200 samples)
  - **Accuracy when confidence $\ge 0.60$:** **80.60%** (108 / 134 correct)
  - **Accuracy when confidence $< 0.60$:** **45.45%** (30 / 66 correct)
  - **Monotonic Calibration Bins:**
    - $[0.0, 0.4)$: 28 samples (14.0%), **32.14% accuracy**
    - $[0.4, 0.6)$: 38 samples (19.0%), **55.26% accuracy**
    - $[0.6, 0.8)$: 52 samples (26.0%), **61.54% accuracy**
    - $[0.8, 1.0]$: 82 samples (41.0%), **92.68% accuracy**

---

## 7. Judge Demonstration Walkthrough

When presenting to hackathon evaluators, follow this verified demonstration sequence:

### Step 1: Show the Executed Training Notebooks
Open either in VS Code or Jupyter:
- Primary transfer learning notebook: [`notebooks/FarmWatch_MobileNetV3_Transfer_Learning.ipynb`](file:///c:/Users/Prince/Documents/antigravity/dazzling-bohr/notebooks/FarmWatch_MobileNetV3_Transfer_Learning.ipynb) (2.16 MB)  
  *Walk judges through Cell 3 (training table showing 14 epochs), Cell 4 (loss & accuracy curves), Cell 6 (test set evaluation output), and Cell 7 (confusion matrix plot).*
- Baseline scratch CNN notebook: [`notebooks/FarmWatch_CNN_Training.ipynb`](file:///c:/Users/Prince/Documents/antigravity/dazzling-bohr/notebooks/FarmWatch_CNN_Training.ipynb) (1.60 MB)  
  *Show the progression from random scratch initialization to transfer learning.*

### Step 2: Run Live Automated Verification
Execute live in terminal to prove all 28 tests pass:
```powershell
.\.venv\Scripts\pytest -v
```

### Step 3: Run Standalone Evaluation Script
Show the test evaluation script executing on held-out samples:
```powershell
.\.venv\Scripts\python -m src.training.evaluate_mobilenetv3
```
Output confirms: 214 test images evaluated, 67.29% test accuracy, 66.52% macro F1, 11.09 ms inference latency.

### Step 4: Demonstrate the Web Application
Launch backend and frontend:
```powershell
# Terminal 1: Backend
.\.venv\Scripts\uvicorn backend.app.main:app --port 8000

# Terminal 2: Frontend
cd frontend
npm run dev
```
Navigate to `http://localhost:5173`.
1. Upload an image from `data/raw/plantdoc/test/Squash Powdery mildew leaf/`.
2. Observe instant prediction (Squash Powdery mildew leaf), high confidence score, and clear Grad-CAM focal heatmap over white powdery mildew patches.
3. Test a low-score or ambiguous leaf to demonstrate the **Heuristic Low-Score Advisory (<60%)** banner and safe cultural recommendations.
4. Click the **Model Metrics & Audit** tab to show live confusion matrix and per-class F1 distribution.
