# FarmWatch AI Implementation Plan

> **For Agent:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and verify FarmWatch AI—a local-first field-image crop condition classification system (Hackathon PS 12) featuring a custom PyTorch CNN trained from random initialization, an optional transfer-learning comparison, Grad-CAM visual explainability, a robust FastAPI backend, and a modern React + TypeScript + Tailwind CSS web interface with region-of-interest selection and honest uncertainty calibration.

**Architecture:** A clean three-tier local architecture:
1. **Data & ML Pipeline (`src/`):** Dataset audit, grouped 70/15/15 train/val/test splitting by original field image to strictly prevent crop leakage, bounding-box leaf cropping with provenance tracking, custom PyTorch CNN training from scratch, test-set evaluation with multi-metric reporting (accuracy, macro F1, confusion matrix), and Grad-CAM activation mapping.
2. **Inference Backend (`backend/`):** FastAPI application with Pydantic request/response schemas, image validation and safe decoding (PIL), singleton model loader with Intel Arc XPU support and CPU fallback, validation-calibrated low-confidence/uncertainty thresholding, Grad-CAM generation, and low-risk agronomist recommendations.
3. **Web Application (`frontend/`):** React 19 + Vite + TypeScript + Tailwind CSS interface featuring an agricultural tech aesthetic (forest green, warm off-white, charcoal), drag-and-drop image upload, interactive leaf region-of-interest (ROI) selector, real-time prediction results with raw model score interpretation, Grad-CAM overlay viewer, real test evaluation metrics viewer, and dataset limitations disclosure.

**Tech Stack:**
- **Languages & Frameworks:** Python 3.13, PyTorch 2.x, FastAPI, Pydantic, Pillow, NumPy, Scikit-learn
- **Frontend:** React, TypeScript, Vite, Tailwind CSS, Lucide Icons, Canvas API
- **Target Hardware:** ASUS Zenbook 14 OLED (Intel Core Ultra 5 125H, 16 GB RAM, Intel Arc Graphics, Windows 11) with Intel XPU probe and robust CPU fallback

---

## Non-Negotiable Honesty & Scope Rules
1. **Explicit Feature Scope:** Name the feature *"Visible Crop/Leaf Condition Classification"*—do not claim detection of drought, nutrient deficiencies, or heat stress unless verified labels exist in the dataset.
2. **Genuine Training from Scratch:** The core custom CNN must be trained from random initialization on downloaded dataset leaf crops. Pretrained models (e.g., MobileNetV3-Small) are strictly optional, must use the exact same splits, and must be clearly labeled as fine-tuning.
3. **No Fabricated Data or Numbers:** All metrics, confusion matrices, logs, and loss curves must be generated directly by executed scripts and saved in `artifacts/reports/`.
4. **No Pesticide Prescriptions:** Provide only general, low-risk cultural/monitoring practices and clearly instruct users to consult certified agronomists or plant pathologists for chemical interventions.
5. **Data Leakage Prevention:** Split strictly by original field image before generating leaf crops. Never allow crops from the same source image to appear across train, validation, or test sets.

---

## Implementation Tasks

### Task 1: Environment Setup & Hardware Acceleration Strategy
**Files:**
- Create: `backend/requirements.txt`
- Create: `src/__init__.py`
- Create: `src/utils/device.py`
- Test: `tests/test_device.py`

**Step 1: Write device detection & hardware abstraction**
Detect whether `torch.xpu.is_available()` is functional on the Intel Core Ultra 5 125H / Intel Arc GPU; if not, safely fall back to CPU without crashing or assuming CUDA.
```python
# src/utils/device.py
import torch

def get_torch_device() -> torch.device:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        try:
            # Test simple tensor allocation to ensure driver stability
            _ = torch.zeros(1, device="xpu")
            return torch.device("xpu")
        except Exception:
            return torch.device("cpu")
    return torch.device("cpu")
```

**Step 2: Create project virtual environment and install dependencies**
Install: `torch`, `torchvision`, `fastapi`, `uvicorn`, `pydantic`, `pillow`, `scikit-learn`, `numpy`, `matplotlib`, `pytest`.

**Step 3: Run test**
Run: `pytest tests/test_device.py -v`
Expected: PASS (reports either `xpu` or `cpu`, strictly logs actual device used).

---

### Task 2: Dataset Acquisition, Audit & Grouped Split Preprocessing
**Files:**
- Create: `src/data/download_dataset.py`
- Create: `src/data/inspect_dataset.py`
- Create: `src/data/preprocess_crops.py`
- Output: `artifacts/reports/dataset_audit.json`, `artifacts/reports/dataset_audit.md`
- Output: `data/processed/train_manifest.json`, `data/processed/val_manifest.json`, `data/processed/test_manifest.json`
- Test: `tests/test_data_pipeline.py`

**Step 1: Dataset acquisition & fallback check**
- Primary: Check `data/raw/fieldplant/` for Roboflow v11 export (COCO JSON or YOLO TXT). Provide precise manual download steps if Roboflow authentication is required.
- Fallback: PlantDoc dataset (`pratikkayal/PlantDoc-Dataset` on GitHub, CC BY 4.0, 2,598 images across 13 species and up to 17 disease classes).
- If fallback is used, document the exact reason and update all app wording accordingly.

**Step 2: Dataset audit script (`inspect_dataset.py`)**
Analyze and record:
- Total original image count and format distributions.
- Total bounding box annotations and class label frequency.
- Out-of-bounds coordinates, corrupt images, missing labels.
- Save structured audit to `artifacts/reports/dataset_audit.json` and human-readable `artifacts/reports/dataset_audit.md`.

**Step 3: Grouped 70/15/15 split by original field image**
- Group images by original image ID using a deterministic random seed (`seed=42`).
- Verify zero image overlap between train, val, and test partitions.
- Extract leaf crops from bounding boxes with safe coordinate clamping and a conservative 5% context margin.
- Save crops to `data/interim/crops/` with a provenance manifest mapping each crop to original image ID, source box coordinates, and label.

**Step 4: Run test**
Run: `pytest tests/test_data_pipeline.py -v`
Expected: PASS (verifies zero leakage between splits, valid crop tensor dimensions).

---

### Task 3: Custom CNN Architecture & Random Initialization Training
**Files:**
- Create: `src/models/custom_cnn.py`
- Create: `src/training/dataset.py`
- Create: `src/training/train_cnn.py`
- Output: `artifacts/models/best_custom_cnn.pt`, `artifacts/models/class_to_idx.json`, `artifacts/models/preprocessing_config.json`, `artifacts/reports/training_history.csv`
- Test: `tests/test_model_training.py`

**Step 1: Implement Custom CNN architecture**
- Readable PyTorch CNN:
  - Block 1: Conv2d(3, 32, 3, padding=1) + BatchNorm2d + ReLU + MaxPool2d(2)
  - Block 2: Conv2d(32, 64, 3, padding=1) + BatchNorm2d + ReLU + MaxPool2d(2)
  - Block 3: Conv2d(64, 128, 3, padding=1) + BatchNorm2d + ReLU + MaxPool2d(2)
  - GlobalAveragePooling + Dropout(0.3) + Linear(128, num_classes)
- Input resolution: 128x128 RGB (or 160x160 based on memory test).
- Default batch size: 8, `num_workers=0` (safe Windows defaults).

**Step 2: Smoke test first**
Run a 1-batch forward/backward pass, loss calculation, optimizer step, and checkpoint save/load to verify pipeline integrity before running full training.

**Step 3: Train model with early stopping & metrics tracking**
- Train using CrossEntropyLoss with class weighting if class imbalance is significant.
- Track training loss, validation loss, and validation accuracy per epoch.
- Save the checkpoint with the highest validation accuracy to `artifacts/models/best_custom_cnn.pt`.
- Export `training_history.csv` and generate loss/accuracy curves.

**Step 4: Optional transfer learning model (secondary)**
- If time and rules permit, fine-tune MobileNetV3-Small on the exact same train/val/test splits.
- Disclose clearly as transfer-learning fine-tuning.

---

### Task 4: Held-Out Test Evaluation & Grad-CAM Explainability
**Files:**
- Create: `src/evaluation/evaluate.py`
- Create: `src/explainability/gradcam.py`
- Output: `artifacts/reports/evaluation_metrics.json`, `artifacts/reports/confusion_matrix.png`, `artifacts/reports/gradcam_sample.png`
- Test: `tests/test_evaluation_and_gradcam.py`

**Step 1: Test set evaluation**
- Run inference strictly on the untouched test split (`data/processed/test_manifest.json`).
- Calculate Accuracy, Macro Precision, Macro Recall, Macro F1, and per-class metrics.
- Generate and save the Confusion Matrix plot to `artifacts/reports/confusion_matrix.png`.
- Measure average inference latency per sample.
- Save all metrics to `artifacts/reports/evaluation_metrics.json`.

**Step 2: Implement Grad-CAM**
- Register forward and backward hooks on the last convolutional layer of `custom_cnn.py`.
- Compute gradients of the predicted class score with respect to feature maps.
- Weight feature maps by global average pooled gradients, apply ReLU, resize to input resolution, and normalize to [0, 1].
- Overlay heatmap onto the RGB crop using a colormap (e.g., JET or VIRIDIS).
- Explicitly output disclaimer: *"Visual explanation shows model feature attention; does not constitute causal biological diagnosis."*

**Step 3: Run test**
Run: `pytest tests/test_evaluation_and_gradcam.py -v`
Expected: PASS (verifies metric schema, confusion matrix generation, Grad-CAM output shape matching input dimensions).

---

### Task 5: FastAPI Production Backend
**Files:**
- Create: `backend/app/main.py`
- Create: `backend/app/schemas/predict.py`
- Create: `backend/app/services/inference.py`
- Create: `backend/app/services/recommendations.py`
- Create: `backend/app/api/endpoints.py`
- Test: `tests/test_api.py`

**Step 1: Endpoints definition**
- `GET /api/health`: reports backend status, PyTorch version, active compute device (`xpu` or `cpu`), and model loaded status.
- `GET /api/model-info`: returns model name, architecture details, class index map, input resolution, normalization parameters, and training dataset source.
- `GET /api/metrics`: returns real test set evaluation metrics loaded directly from `artifacts/reports/evaluation_metrics.json`.
- `POST /api/predict`:
  - Accepts image file (`image/jpeg`, `image/png`, `image/webp`), max 15MB.
  - Accepts optional bounding box crop coordinates `[x_min, y_min, x_max, y_max]`.
  - Rejects corrupted or non-image files with HTTP 400 and actionable error details.
  - Loads model once as a singleton at startup (no reloading per request).
  - Performs preprocessing (resize, RGB conversion, normalize).
  - Generates raw model logits and softmax probabilities.
  - Computes Grad-CAM overlay and encodes as base64 PNG.
  - Applies validation-calibrated uncertainty threshold: if top score < 0.60, mark as `uncertain: true` with a clear advisory.
  - Returns structured JSON response with predicted class, model score, score interpretation, Grad-CAM base64, uncertainty flag, and low-risk cultural recommendations.

**Step 2: Low-risk agricultural recommendations engine**
- Maps predicted classes to non-prescriptive, low-risk actions:
  - Isolation / leaf pruning
  - Sanitation of tools
  - Moisture / humidity control
  - Referral to extension service or agronomist
  - Strict prohibition against chemical pesticide dosage instructions.

**Step 3: Run API tests**
Run: `pytest tests/test_api.py -v`
Expected: PASS (tests health check, valid image prediction, corrupt image rejection, oversized payload rejection, missing checkpoint response).

---

### Task 6: Modern React + TypeScript + Tailwind CSS Frontend
**Files:**
- Create: `frontend/` (scaffolded with Vite + React 19 + TypeScript)
- Create: `frontend/src/types/api.ts`
- Create: `frontend/src/lib/api.ts`
- Create: `frontend/src/components/Header.tsx`
- Create: `frontend/src/components/ImageUploader.tsx`
- Create: `frontend/src/components/CropSelector.tsx` (interactive ROI canvas)
- Create: `frontend/src/components/PredictionResult.tsx`
- Create: `frontend/src/components/GradCamViewer.tsx`
- Create: `frontend/src/components/MetricsDashboard.tsx`
- Create: `frontend/src/components/DatasetLimitations.tsx`
- Create: `frontend/src/App.tsx`
- Test: `npm run build` inside `frontend/`

**Step 1: Design system & styling**
- Palette: Forest green (`#1b4332`, `#2d6a4f`), warm off-white (`#f8f9fa`), slate/charcoal text (`#0f172a`, `#334155`), amber/emerald accents.
- Responsive, clean layout without cluttered analytics or fake marketing elements.

**Step 2: Core workflow implementation**
- **Upload & ROI Selection:** Allows uploading field photos or close-up leaf photos. Includes an interactive bounding box selector so users can target a specific leaf before inference.
- **Inference & Explainability:** Displays the top predicted class, raw model score with an explanation of what the score represents, confidence pill, and interactive toggle for the Grad-CAM heatmap overlay.
- **Uncertainty Alert:** Prominent banner for low-confidence predictions recommending physical inspection by an agronomist.
- **Low-Risk Recommendations:** Curated next steps for field monitoring and tool sanitation.
- **Real Metrics & Limitations:** Dedicated tab rendering real JSON evaluation metrics (test accuracy, macro F1, confusion matrix) and explicit dataset citations (FieldPlant IEEE Access, CC BY 4.0 license).

**Step 3: Verify build**
Run: `npm run build`
Expected: Zero build errors.

---

### Task 7: Comprehensive End-to-End Verification & Documentation
**Files:**
- Create: `README.md`
- Create: `AGENTS.md`
- Create: `.gitignore`
- Create: `.env.example`
- Output: `artifacts/reports/final_execution_report.md`
- Test: End-to-end integration run on a held-out test image.

**Step 1: Run end-to-end inference verification**
Verify that the full cycle works seamlessly: backend launched, model loaded, sample test image sent via API, Grad-CAM generated, correct response rendered in UI.

**Step 2: Write documentation**
- Complete `README.md`: Environment setup, dataset citation and license, training command, evaluation command, backend & frontend startup commands, known limitations, and 90-second demo script.
- `AGENTS.md`: Truthfulness constraints, architectural rules, and reproduction steps for future agent sessions.
- `.gitignore`: Exclude raw datasets, weights, virtual environments, node modules, and local cache.

---

## 10-Hour Execution Schedule

| Window | Focus | Key Deliverables & Evidence |
|---|---|---|
| **P0 (0–1h)** | Environment & Data Audit | Python 3.13 venv, PyTorch install, device check (Arc GPU / CPU), raw dataset audit, `dataset_audit.json/md`. |
| **P1 (1–3h)** | Grouped Split & Custom CNN Training | Grouped 70/15/15 split, bounding-box leaf crops, train custom CNN from random initialization, save best checkpoint & training logs. |
| **P2 (2–6h)** | API & UI Development | FastAPI endpoints (`/health`, `/model-info`, `/predict`, `/metrics`), React + Tailwind UI with ROI crop selector. |
| **P3 (5–7h)** | Grad-CAM & Calibration | Integrate model checkpoint with API, compute Grad-CAM heatmaps, tune uncertainty threshold on validation data. |
| **P4 (7–9h)** | Held-out Evaluation & Testing | Evaluate test split, generate confusion matrix & macro metrics, execute all backend and frontend unit/integration tests. |
| **P5 (Final h)** | Final Polish & Verification | Run `npm run build`, test end-to-end flow, verify `README.md`, `AGENTS.md`, and complete final report. |

---

## Final Acceptance Checklist
- [ ] Dataset source, license (CC BY 4.0), and actual counts recorded from files.
- [ ] Grouped train/val/test split strictly by original field image (zero image leakage).
- [ ] Custom PyTorch CNN trained from scratch (random initialization) with reloadable checkpoint.
- [ ] Real test set metrics (accuracy, macro F1, confusion matrix) produced by evaluation script.
- [ ] FastAPI loads trained model and performs local inference.
- [ ] React UI provides image upload, interactive ROI crop selection, Grad-CAM toggle, and uncertainty state.
- [ ] Clear feature scope: visible leaf condition classification, not universal crop stress.
- [ ] Zero build errors on `npm run build` and all backend tests pass.
