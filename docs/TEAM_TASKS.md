# FarmWatch AI: Six-Person Team Task Assignments & Integration Status

**Hackathon Problem:** GenHack V1.0 — PS 12: Predicting Crop Stress From Field Images  
**Core Goal:** Build and demonstrate a local-first field-image crop condition classification system with a genuinely trained CNN, honest scoping, explainability via Grad-CAM, and clean architecture.

---

## Team Division & File Ownership

### Member 1: Model Training Lead (Me + Antigravity)
**Status:** **Completed & Verified**  
**Deliverables:**
- Trained custom PyTorch CNN from random initialization: `models/enhanced_cnn.pth` (1,207,787 parameters).
- Tested Intel Arc XPU acceleration on ASUS Zenbook 14 OLED, retaining robust CPU fallback (14.05 ms latency).
- Executed Jupyter notebook: `notebooks/FarmWatch_CNN_Training.ipynb` (1.60 MB) with genuine training curves and confusion matrix.
- Evaluation metrics: `reports/focused_test_metrics.json` (27.10% test accuracy, 21.74% macro F1 on 11 classes).
- Artifacts & mappings: `models/focused_class_to_idx.json`, `models/focused_preprocess_config.json`.
- Packaged release bundle: `dist/farmwatch_model_bundle.zip` (12.69 MB).

---

### Member 2: Dataset Verification & Agricultural Guidance
**Status:** **Completed & Verified**  
**Deliverables:**
- Verified PlantDoc dataset licensing (CC BY 4.0) and citation (IEEE Access 2023, DOI: 10.1109/ACCESS.2023.3263042).
- Zero data leakage: 21 exact SHA-256 duplicate files removed; 70/15/15 splits grouped by unique content hash (`data/manifests/focused_train_manifest.json`, etc.).
- Feature strictly named: **"Visible Crop/Leaf Condition Classification"**. Zero claims of predicting invisible stress (drought, nitrogen deficiency, heat stress).
- Low-risk non-chemical recommendations service: `backend/app/services/recommendations.py` (isolation, sanitation, moisture management; strict prohibition against chemical pesticide dosages).

---

### Member 3: FastAPI Backend Lead
**Status:** **Completed & Verified**  
**Deliverables:**
- Production FastAPI service: `backend/app/main.py` with lifespan model pre-warming and CORS middleware.
- Inference singleton service: `src/inference/model_service.py` and `backend/app/services/inference.py`.
- Endpoints: `GET /api/health`, `GET /api/model-info`, `GET /api/metrics`, `POST /api/predict`, `GET /api/samples`.
- Input validation: rejects corrupted files (HTTP 400), unsupported MIME types (HTTP 400), empty files (HTTP 400), and oversized payloads > 15MB (HTTP 413).
- 10 automated backend tests: `tests/test_api.py`.

---

### Member 4: React Frontend Lead
**Status:** **Completed & Verified**  
**Deliverables:**
- React 19 + Vite + TypeScript + Tailwind CSS application (`frontend/`).
- Design system: Forest green (`#1b4332`, `#2d6a4f`), warm slate background, responsive two-column layout.
- Components:
  - `Header.tsx`: System badges, Intel Core Ultra 5 CPU status, tab navigation.
  - `ImageUploader.tsx`: Drag & drop, file picker, held-out benchmark sample gallery.
  - `CropSelector.tsx`: Interactive ROI canvas for isolating individual leaves in cluttered field photos.
  - `PredictionResult.tsx`: Condition title, confidence pill, uncertainty alert (<60%), top-3 hypotheses, low-risk cultural guidance.
  - `GradCamViewer.tsx`: Overlay & side-by-side modes with blend intensity slider.
  - `MetricsDashboard.tsx`: Authentic 27.10% accuracy, 21.74% macro F1, latency, per-class F1 bars, confusion matrix.
  - `DatasetLimitations.tsx`: Research citation, scope boundaries, baseline vs enhanced comparison.
- Production build: `npm run build` succeeds with 0 errors (1,929 modules bundled in 848 ms).

---

### Member 5: Explainability & Model Presentation
**Status:** **Completed & Verified**  
**Deliverables:**
- PyTorch Grad-CAM engine: `src/explainability/gradcam.py`.
- Final convolutional layer hooks (`block4[3]` with 256 channels).
- Automatic hook cleanup and memory leak prevention.
- Seamless base64 data URI export for browser rendering.
- Scientific disclaimer in UI: *"Visual explanation shows neural network feature attention; does not constitute biological or causal diagnosis."*

---

### Member 6: Integration, QA & Demo Lead
**Status:** **Completed & Verified**  
**Deliverables:**
- Full test suite: 18 automated tests passing in `tests/` (`.\.venv\Scripts\pytest -v`).
- End-to-end integration test: `tests/test_e2e_integration.py` verifies full pipeline on held-out field images.
- Rehearsed 90-second pitch script: `docs/DEMO_SCRIPT.md`.
- Hackathon slide outline: `presentation/FarmWatch_Presentation_Outline.md`.
- Comprehensive setup guide and truthful limitations: `README.md`.
