# FarmWatch AI: Integration Status, Component Inventory & Release Matrix

**Date:** 2026-10-17  
**Problem Statement:** GenHack V1.0 PS 12 — Predicting Crop Stress From Field Images  
**Target Hardware:** ASUS Zenbook 14 OLED | Intel Core Ultra 5 125H (14 Cores) | 16 GB RAM | Windows 11  
**Overall Integration Status:** 100% Verified & Tested End-to-End (28/28 Automated Tests Passing)

---

## 1. Verified Model Checkpoint Inventory

All checkpoints have been verified for integrity, load strictly into their respective architectures, and produce finite outputs.

| Checkpoint File | Relative Path | Architecture | Classes | File Size | SHA-256 Digest | Status in App |
|---|---|---|---|---|---|---|
| **`mobilenetv3_small_transfer.pth`** | `models/mobilenetv3_small_transfer.pth` | `MobileNetV3-Small` | 11 | 17,002,255 bytes (16.21 MB) | `3872120a313f7067fe6bdf968f3326afc6ff017337b399c21e8a988709f49a0c` | **Primary Production Model** |
| **`enhanced_cnn.pth`** | `models/enhanced_cnn.pth` | `EnhancedCropCNN` | 11 | 14,546,838 bytes (13.87 MB) | `5a87f6091d4ddb15cf1135a2dab26d71d193ccaee11ce227ddd10455a4a3fb15` | Preserved Scratch Baseline |
| **`cnn_from_scratch.pth`** | `models/cnn_from_scratch.pth` | `CustomCropCNN` | 28 | 1,184,727 bytes (1.13 MB) | `cf552c770636a1f1d5cd83019aae5ee55144605c8e07a9dece5ce3c449844b57` | Preserved 28-Class Baseline |

> **Checkpoint Hash Invariance:** Checked before and after all audit operations. Exactly identical SHA-256 hashes confirm zero parameter tampering or retraining.

---

## 2. Component Verification Matrix

| Component | Files Available | Status | Tests Executed | Verification Details |
|---|---|---|---|---|
| **Data Pipeline & Manifests** | `data/manifests/*.json`, `src/utils/dataset_paths.py` | Verified | `test_smoke_training.py`, `test_mobilenetv3_inference.py` | All 6 manifests converted to repository-relative POSIX paths. `FARMWATCH_DATASET_ROOT` supported. Zero hash collisions across splits. |
| **Primary Model Architecture** | `src/training/train_mobilenetv3.py` (`build_mobilenetv3_model`) | Verified | `test_mobilenetv3_inference.py` (5 tests) | Torchvision `mobilenet_v3_small` with custom 11-class head. Offline inference guarantee via `pretrained=False`. |
| **Baseline Architectures** | `src/models/custom_cnn.py` (`EnhancedCropCNN`, `CustomCropCNN`) | Verified | `test_model_and_inference.py` (4 tests) | Pure scratch initialization (Kaiming Normal). Fully preserved for comparative benchmarking. |
| **Hardware Abstraction** | `src/utils/device.py` | Verified | Device probe test | Intel Arc XPU probe with graceful CPU fallback (14 cores, Windows 11). |
| **Explainability (Grad-CAM)** | `src/explainability/gradcam.py` | Verified | Grad-CAM lifecycle test | Target conv layer hook (`model.features[-1]` for MobileNet, `block4[3]` for EnhancedCropCNN), overlay generation, PNG base64 export. |
| **Evaluation Metrics** | `reports/mobilenetv3_test_metrics.json`, `reports/focused_test_metrics.json` | Verified | Independent test export script | MobileNetV3: **67.29% accuracy, 66.52% macro F1**. Enhanced CNN: **27.10% accuracy, 21.74% macro F1**. |
| **Jupyter Notebooks** | `notebooks/FarmWatch_MobileNetV3_Transfer_Learning.ipynb`, `notebooks/FarmWatch_CNN_Training.ipynb` | Verified | Full execution & inspection | Unbroken cell outputs, training curves, confusion matrix, and markdown documentation. |
| **Inference Engine** | `src/inference/model_service.py` | Verified | API & inference unit tests | Model switching via `FARMWATCH_MODEL` env var, 224x224 & 128x128 transforms, ROI cropping, uncalibrated softmax, heuristic 0.60 threshold. |
| **FastAPI Backend** | `backend/app/main.py`, `backend/app/api/endpoints.py`, `backend/app/schemas/predict.py` | Verified | 15 API tests in `test_api.py` | `/health`, `/model-info`, `/metrics`, `/predict`, `/samples`, CORS allowlist, error concealment with UUID correlation ID. |
| **React Frontend** | `frontend/` (Vite + React 19 + TypeScript + Tailwind CSS) | Verified | `npm run build` | Built in 585 ms; zero compile errors; dynamic zero-F1 badge; uncalibrated advisory banner. |
| **End-to-End Flow** | `tests/test_e2e_integration.py` | Verified | Full pytest execution (28 passed) | Real held-out test sample processed through complete FastAPI + PyTorch CNN + Grad-CAM pipeline. |
| **Release Bundles** | `FarmWatch_Model_Audit_Bundle.zip` (root & reports), `dist/farmwatch_model_bundle.zip` | Verified | Zip integrity test | Clean ZIP archives containing code, models, sample predictions, class probabilities, manifests, and documentation. |

---

## 3. Parameter & Latency Specifications

### MobileNetV3-Small Parameter Breakdown
- **Total Parameters:** **1,529,131** (100%)
- **Stage 1 (Head Warm-up, Epochs 1–4):**
  - Trainable parameters: **602,123** (39.38%) in the classifier head
  - Frozen parameters: **927,008** (60.62%) across all feature extractor blocks
- **Stage 2 (Fine-Tuning, Epochs 5–14):**
  - Trainable parameters: **1,338,611** (87.54%) across `features[9:]` + classifier head
  - Frozen parameters: **190,520** (12.46%) in early feature blocks `features[0:9]` (permanently frozen to preserve low-level generic filters)

### Empirical Latency Benchmark (Intel Core Ultra 5 125H, CPU)
- **Pure Model-Forward Latency (tensor -> logits -> softmax):** Mean = **15.34 ms**, P95 = **21.58 ms**
- **Full API Latency (without Grad-CAM):** Mean = **63.02 ms** (includes image decoding, PIL transform, model forward, top-3 candidates, recommendation lookup, JSON serialization)
- **Full API Latency (with Grad-CAM):** Mean = **198.59 ms** (includes all above plus PyTorch backward hook gradient computation, heatmap generation, PIL image blending, and PNG base64 encoding)

---

## 4. Uncertainty & Validation Calibration Audit

- **Calibration Nature:** Softmax outputs are uncalibrated empirical model scores; no Platt or temperature scaling was applied.
- **Validation Split (200 images) Calibration Results:**
  - Overall accuracy on validation set: **69.00%** (138 / 200 correct)
  - Mean confidence: **0.7050**, Median: **0.7153**
  - **Heuristic 0.60 Threshold Coverage:** **67.00%** (134 / 200 samples)
  - **Accuracy when confidence $\ge 0.60$:** **80.60%** (108 / 134 correct)
  - **Accuracy when confidence $< 0.60$:** **45.45%** (30 / 66 correct)
  - **Monotonic Calibration Bins:**
    - $[0.0, 0.4)$: 28 samples (14.0%), **32.14% accuracy**
    - $[0.4, 0.6)$: 38 samples (19.0%), **55.26% accuracy**
    - $[0.6, 0.8)$: 52 samples (26.0%), **61.54% accuracy**
    - $[0.8, 1.0]$: 82 samples (41.0%), **92.68% accuracy**

---

## 5. Verified Execution Commands

### 1. Run Automated Test Suite (28 Tests)
```powershell
.\.venv\Scripts\pytest -v
```
*Expected: 28 passed in ~9s.*

### 2. Build Frontend
```powershell
cd frontend
npm run build
cd ..
```
*Expected: Built in ~600ms with zero errors.*

### 3. Launch Backend Server
```powershell
.\.venv\Scripts\uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

### 4. Launch Frontend Server
```powershell
cd frontend
npm run dev
```
Navigate to: `http://localhost:5173`
