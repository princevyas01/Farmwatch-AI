# FarmWatch AI: Hackathon Presentation Slide Outline

**Hackathon:** GenHack V1.0  
**Problem Statement:** PS 12 — Predicting Crop Stress From Field Images  
**Project:** FarmWatch AI  
**Repository:** `dazzling-bohr`  

---

## Slide 1: Title & Team
* **Heading:** FarmWatch AI
* **Subheading:** Ground-Truth Visible Crop Condition Classification with Scratch-Trained CNN & Explainable Feature Attention
* **Track:** GenHack V1.0 · Problem Statement 12
* **Team Members (6):**
  - Member 1: Model Training & Architecture Lead
  - Member 2: Dataset Verification & Agronomy Lead
  - Member 3: FastAPI Backend Lead
  - Member 4: React & Design Systems Lead
  - Member 5: Grad-CAM Explainability Lead
  - Member 6: Integration, QA & Demonstration Lead
* **Hardware:** ASUS Zenbook 14 OLED (Intel Core Ultra 5 125H, 16 GB RAM, Windows 11)

---

## Slide 2: The Core Problem — Field Images vs Lab Benchmarks
* **The Problem:** 80% of crop disease papers use laboratory datasets (e.g., PlantVillage) with clean leaves centered on uniform black or grey backgrounds. These models collapse in real fields with harsh sunlight, shadows, overlapping plants, and soil background.
* **Our Target:** PlantDoc Field-Image Benchmark (CC BY 4.0, IEEE Access 2023).
* **Scope Discipline:** Strictly named *"Visible Crop/Leaf Condition Classification"*. We refuse to claim detection of invisible subterranean stressors (nitrogen deficiency, drought, salinity) without empirical ground truth.

---

## Slide 3: Dataset Audit & Zero-Leakage Architecture
* **Total Audited Images:** 2,560 unique field images across 28 classes.
* **Leakage Prevention:** 21 duplicate files identified and purged via SHA-256 content hashing. Splits grouped strictly by unique image content to ensure zero test contamination.
* **Iterative Engineering:**
  - *Baseline (28 classes):* Severe class imbalance (classes with 2–5 samples); 13 starved classes had 0.0% F1.
  - *Enhanced Subset (11 classes):* Filtered to well-represented conditions ($\ge 100$ samples each, 951 train / 200 val / 214 held-out test).

---

## Slide 4: Neural Architecture & Scratch Training
* **Architecture:** `EnhancedCropCNN`
  - 4 progressive convolutional blocks with double-convolutions (32 $\rightarrow$ 64 $\rightarrow$ 128 $\rightarrow$ 256 channels).
  - Batch normalization & ReLU activations for stable gradient flow.
  - Adaptive Average Pooling $\rightarrow$ Dropout (0.3) $\rightarrow$ Dense classification head.
* **Initialization:** Kaiming Normal (Trained 100% from scratch; zero ImageNet pretrained weights).
* **Optimization:** AdamW optimizer + Cosine Annealing learning rate schedule + class-weighted CrossEntropyLoss with 0.05 label smoothing.
* **Parameters:** 1,207,787 parameters.

---

## Slide 5: Empirical Results & Complete Honesty
* **Honesty Policy:** No fabricated logs or simulated accuracy numbers.
* **Held-Out Test Results (214 untouched samples):**
  - **Accuracy:** 27.10% (vs 9.09% random guess baseline, 3x improvement).
  - **Macro F1-Score:** 21.74% (vs 11.17% in baseline).
  - **Zero-F1 Classes:** 0 / 11 classes (completely eliminated class starvation).
  - **Top Performing Classes:** Squash Powdery Mildew (54.2% F1), Peach Leaf (45.6% F1), Corn Rust (40.0% F1).
  - **Inference Latency:** 14.05 ms / sample on Intel CPU.

---

## Slide 6: Explainability via Grad-CAM & Safety
* **Mechanism:** Backpropagates class gradients to the final convolutional layer `block4[3]` (256 channels, $8 \times 8$ spatial grid).
* **Farmer Insight:** Interactive blend overlay highlights exactly which leaf lesions and fungal patches influenced the classification.
* **Validation-Calibrated Uncertainty:** When model confidence $< 0.60$, the system automatically issues an **Uncertainty Alert**, advising manual inspection.
* **Ethical Guidance:** Zero chemical pesticide prescriptions; only low-risk cultural sanitation, airflow improvement, and agronomist referral.

---

## Slide 7: Production Architecture & Stack
* **Frontend:** React 19, Vite 8, TypeScript, Tailwind CSS v4, Lucide icons.
* **Backend:** FastAPI, Python 3.13, Pydantic schemas, CORS middleware, lifespan singleton model caching.
* **Compute:** Hardware abstraction layer with Intel Arc XPU detection and high-efficiency CPU fallback.
* **Packaging:** Single 12.69 MB distribution bundle (`dist/farmwatch_model_bundle.zip`) containing weights, mappings, configs, and standalone inference snippet.

---

## Slide 8: Live Demonstration & Next Steps
* **Live System Demo:** React UI $\rightarrow$ FastAPI Backend $\rightarrow$ PyTorch CNN $\rightarrow$ Grad-CAM visualization.
* **Future Work:**
  - Multi-spectral aerial imaging integration for macro field stress.
  - Mobile edge deployment via TorchScript (`models/enhanced_cnn_traced.pt`).
  - Active learning pipeline to crowd-source annotations from field agronomists.
