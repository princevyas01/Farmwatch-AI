# AGENTS.md — FarmWatch AI Project Guidelines

## Project Identity & Architecture
- **Problem Statement:** GenHack V1.0 PS 12 — Predicting Crop Stress From Field Images
- **Model Training:** PyTorch Custom CNN (`src/models/custom_cnn.py`), scratch-trained baseline (Kaiming Normal initialization).
- **Compute Strategy:** Probe Intel Arc XPU via `src/utils/device.py`; fall back gracefully to CPU. Never hard-code CUDA.
- **Dataset:** PlantDoc field-image benchmark (CC BY 4.0). Split manifests in `data/manifests/`.

## Non-Negotiable Honesty Rules
1. Never fabricate training logs, accuracy numbers, confusion matrices, or test metrics. Every number must come from an executed script.
2. The core model must remain trained from random initialization. If transfer learning is added, it must be explicitly labeled as fine-tuning.
3. Explicit feature naming: *"Visible Crop/Leaf Condition Classification"*. Never claim detection of unannotated stress types (drought, nitrogen deficiency, heat stress).
4. No chemical pesticide dosage prescriptions; only low-risk cultural guidance and referral to agronomists.
5. Zero data leakage: Splits must remain grouped by unique image content hash.

## Standard Verification Commands
- Unit & Integration Tests: `.\.venv\Scripts\pytest -v`
- Dataset Audit & Manifest Generation: `.\.venv\Scripts\python src/training/inspect_dataset.py`
- Scratch Training: `.\.venv\Scripts\python src/training/train_cnn.py`
- Test Evaluation: `.\.venv\Scripts\python src/training/evaluate_model.py`
- Standalone Inference: `.\.venv\Scripts\python src/inference_example.py`
