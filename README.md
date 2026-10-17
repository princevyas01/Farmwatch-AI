# FarmWatch AI — Ground-Truth Field Crop Condition Classification

**GenHack V1.0 | Problem Statement 12: Predicting Crop Stress From Field Images**  
**Host Target:** ASUS Zenbook 14 OLED | Intel Core Ultra 5 125H | 16 GB RAM | Windows 11  
**Architecture:** PyTorch MobileNetV3-Small Transfer Learning + Scratch Baselines + FastAPI Backend + React 19 / TypeScript / Tailwind CSS Frontend  
**Integration Status:** 100% Tested & Verified End-to-End (28/28 Automated Tests Passing)

---

## 1. Executive Summary

FarmWatch AI provides an automated, locally executed visual estimation system for **visible crop and leaf conditions** using unconstrained field images. Unlike laboratory benchmarks where isolated leaves are photographed flat against uniform monochrome backgrounds, FarmWatch AI trains and evaluates strictly on the natural, noisy **PlantDoc benchmark** (variable illumination, wind flutter, overlapping leaves, and complex soil clutter).

### Non-Negotiable Scientific Integrity
1. **Explicit Lineage & Honesty:** Primary production model uses ImageNet-1K MobileNetV3-Small transfer learning. Baseline models were trained from scratch with Kaiming Normal initialization. No hidden weights or fabricated metrics.
2. **Explicit Feature Scope:** Strictly named *"Visible Crop/Leaf Condition Classification"*. Zero claims of detecting invisible subterranean stress (drought, nitrogen deficiency, salinity) without ground-truth physical labels.
3. **Zero Data Leakage:** Grouped train/val/test splits partitioned strictly by unique SHA-256 image content hashes (zero hash overlap across splits).
4. **Responsible Agronomy:** Provides only safe, low-risk cultural guidance (tool disinfection, pruning, aeration) with a strict prohibition against chemical pesticide dosage prescriptions.

---

## 2. Verified Benchmark Performance

All metrics below are generated directly from executed evaluation scripts on the held-out test partition (214 untouched field images):

| Metric | Phase 1 (28-Class Baseline) | Phase 2 (11-Class Scratch CNN) | Phase 3 (MobileNetV3 Transfer Learning) | Notes |
|---|---|---|---|---|
| **Model Architecture** | `CustomCropCNN` (3 blocks) | `EnhancedCropCNN` (4 blocks) | `MobileNetV3-Small` | Selected production model |
| **Total Parameters** | 97,084 | 1,207,787 | **1,529,131** | 100% of architecture |
| **Stage 1 (Head Warmup)** | — | — | **602,123 params** (39.4%) | 927,008 frozen backbone params |
| **Stage 2 (Fine-Tuning)** | — | — | **1,338,611 params** (87.5%) | 190,520 permanently frozen params |
| **Initialization** | Kaiming Normal (Scratch) | Kaiming Normal (Scratch) | ImageNet-1K Pretrained Backbone | Custom 11-class head |
| **Held-Out Test Accuracy** | 18.29% | 27.10% (58 / 214) | **67.29%** (144 / 214) | **+40.19 pp** (+148.3% rel) |
| **Macro F1-Score** | 11.17% | 21.74% | **66.52%** | **+44.78 pp** (+206.0% rel) |
| **Zero-F1 Starved Classes** | 13 of 28 classes (46%) | 3 of 11 classes (27%) | **0 of 11 classes (0%)** | Zero-F1 classes eliminated |
| **Model Forward Latency** | 4.90 ms | 9.80 ms | **15.34 ms** (P95: 21.58 ms) | Tensor forward pass on CPU |
| **Full API Latency (no CAM)** | 45.20 ms | 52.10 ms | **63.02 ms** | End-to-end request on CPU |
| **Full API Latency (with CAM)**| 145.00 ms | 172.50 ms | **198.59 ms** | End-to-end + Grad-CAM heatmap |
| **Heuristic Low-Score Advisory** | `< 0.60` threshold | `< 0.60` threshold | `< 0.60` threshold | 80.60% acc when $\ge 0.60$ |

---

## 3. Repository Architecture

```
dazzling-bohr/
├── backend/
│   ├── app/
│   │   ├── api/endpoints.py       # FastAPI router (/health, /model-info, /metrics, /predict, /samples)
│   │   ├── schemas/predict.py     # Pydantic schemas for request/response validation
│   │   ├── services/inference.py  # Model inference adapter
│   │   ├── services/recommendations.py # Low-risk non-chemical agronomic guidance
│   │   └── main.py                # FastAPI entrypoint with lifespan model pre-warming & CORS
├── frontend/
│   ├── src/
│   │   ├── components/            # Header, CropSelector (ROI), ImageUploader, PredictionResult, GradCamViewer, MetricsDashboard, DatasetLimitations
│   │   ├── lib/api.ts             # Typed REST API client
│   │   ├── types/api.ts           # TypeScript interfaces matching backend models
│   │   ├── App.tsx                # Main single-page application with tab switcher
│   │   └── index.css              # Tailwind CSS styling
│   ├── package.json               # React 19 + Vite 8 + Tailwind CSS + Lucide Icons
│   └── vite.config.ts             # Vite configuration with /api proxy
├── models/
│   ├── mobilenetv3_small_transfer.pth # Primary model checkpoint (16.21 MB, SHA-256: 3872120a...)
│   ├── mobilenetv3_preprocess_config.json # 224x224 resolution & ImageNet stats
│   ├── mobilenetv3_class_to_idx.json  # 11-class mapping
│   ├── enhanced_cnn.pth           # Scratch 11-class CNN checkpoint (13.87 MB)
│   ├── focused_class_to_idx.json  # 11-class label mapping
│   ├── focused_preprocess_config.json # 128x128 resolution
│   ├── cnn_from_scratch.pth       # 28-class baseline checkpoint (1.18 MB)
│   └── class_to_idx.json          # 28-class mapping
├── notebooks/
│   └── FarmWatch_CNN_Training.ipynb # Executed notebook with unbroken curves & logs
├── reports/
│   ├── mobilenetv3_test_metrics.json # Genuine MobileNetV3 held-out test evaluation metrics
│   ├── mobilenetv3_test_predictions.csv # Sample-level test predictions
│   ├── validation_uncertainty_metrics.json # Validation set calibration and threshold metrics
│   ├── focused_test_metrics.json  # Genuine 11-class scratch CNN held-out test metrics
│   ├── focused_confusion_matrix.png
│   └── model_audit/               # Comprehensive technical audit package & ZIP
├── src/
│   ├── explainability/gradcam.py  # PyTorch Grad-CAM hook
│   ├── inference/model_service.py # Singleton thread-safe model engine (supports MobileNetV3 & scratch CNNs)
│   ├── models/custom_cnn.py       # PyTorch architectures (EnhancedCropCNN, CustomCropCNN)
│   ├── training/train_mobilenetv3.py # Transfer learning training pipeline
│   └── utils/device.py            # Intel Arc XPU probe with CPU fallback
├── tests/
│   ├── test_api.py                # FastAPI integration tests
│   ├── test_e2e_integration.py    # Full end-to-end test on real test-set image
│   ├── test_mobilenetv3_inference.py # Checkpoint loading, finite logits & 224x224 inference
│   ├── test_model_and_inference.py# Checkpoint integrity & inference verification
│   └── test_smoke_training.py     # Forward/backward training smoke tests & Grad-CAM hooks
├── FarmWatch_Model_Audit_Report.md # Full technical model audit & release verification report
└── FarmWatch_Model_Audit_Bundle.zip # Complete audit archive for external reviewers
```

---

## 4. Quickstart & Startup Commands (Windows 11)

### Prerequisites
* Windows 11 (or any x86_64 PC)
* Python 3.13 (or 3.10+) in virtual environment (`.\.venv`)
* Node.js v24+ and npm v11+

### Step 1: Run Automated Verification Tests (28 Tests)
```powershell
.\.venv\Scripts\pytest -v
```
*Expected: 28 passed.*

### Step 2: Build React Frontend
```powershell
cd frontend
npm run build
cd ..
```
*Expected: `✓ built in ~750ms` with zero errors.*

### Step 3: Launch FastAPI Backend Server
In Terminal 1:
```powershell
.\.venv\Scripts\uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```
Interactive API documentation available at: `http://localhost:8000/docs`

> **Model Selection via Environment Variable:**
> - Default: MobileNetV3-Small (`FARMWATCH_MODEL=mobilenetv3`)
> - Enhanced Scratch CNN: `FARMWATCH_MODEL=enhanced_cnn`
> - Initial Scratch CNN: `FARMWATCH_MODEL=custom_cnn`

### Step 4: Launch React Frontend Application
In Terminal 2:
```powershell
cd frontend
npm run dev
```
Open browser at: **`http://localhost:5173`**

---

## 5. Dataset Provenance & Attribution

* **Dataset:** PlantDoc: A Dataset for Visual Plant Disease Detection in Field Images
* **Authors:** Pratik Kayal, Bhuvanesh S., et al. (ACM CoDS-COMAD 2020 / IEEE Access 2023)
* **DOI:** `10.1109/ACCESS.2023.3263042`
* **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
* **Dataset Acquisition:** The licensed PlantDoc dataset is open-access and available from the official repository at `https://github.com/pratikkayal/PlantDoc-Dataset`. Clone or download the dataset and place the `train/` and `test/` image directories in `data/raw/plantdoc/`.
* **Configurable Dataset Root:** To point to an external or alternate dataset location on Windows or Linux, set the `FARMWATCH_DATASET_ROOT` environment variable:
  ```powershell
  $env:FARMWATCH_DATASET_ROOT = "D:\datasets\PlantDoc"
  ```
  Manifests use cross-platform repository-relative POSIX paths (`data/raw/plantdoc/train/...`) and resolve automatically.
* **Evaluation Partition:** Custom held-out stratified split of 214 images from the combined PlantDoc benchmark (951 train, 200 val, 214 test, with zero SHA-256 hash overlap across splits).
* **11 Evaluated Classes:**
  1. `Blueberry leaf`
  2. `Corn leaf blight`
  3. `Corn rust leaf`
  4. `Peach leaf`
  5. `Potato leaf early blight`
  6. `Potato leaf late blight`
  7. `Raspberry leaf`
  8. `Squash Powdery mildew leaf`
  9. `Tomato Septoria leaf spot`
  10. `Tomato leaf bacterial spot`
  11. `Tomato leaf late blight`
