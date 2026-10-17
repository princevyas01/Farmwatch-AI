# FarmWatch AI: 90-Second Hackathon Demo Script

**Hackathon:** GenHack V1.0  
**Problem Statement:** PS 12 — Predicting Crop Stress From Field Images  
**Presenter:** Model Training & Full-Stack Integration Lead  
**Hardware:** ASUS Zenbook 14 OLED (Intel Core Ultra 5 125H, 16 GB RAM, Windows 11)  
**Total Target Time:** 90 Seconds  

---

## Preparation (Before Taking the Stage)

1. Open Terminal 1 (Backend):
   ```powershell
   .\.venv\Scripts\uvicorn backend.app.main:app --port 8000
   ```
2. Open Terminal 2 (Frontend):
   ```powershell
   cd frontend
   npm run dev
   ```
3. Open Browser to: `http://localhost:5173`
4. Have `notebooks/FarmWatch_CNN_Training.ipynb` open in VS Code / Jupyter in the background.

---

## 90-Second Presentation Sequence

### [0:00 - 0:15] Slide & Problem Statement: The Reality of Field Images
* *"Good afternoon, judges. We are presenting FarmWatch AI for PS 12. Unlike benchmark lab datasets like PlantVillage with clean grey backgrounds, real farmers photograph crops in the wild with bright sunlight, wind flutter, overlapping leaves, and complex soil clutter."*
* *"We made a strict scientific integrity decision: we trained a custom 4-block deep PyTorch CNN **100% from scratch** without pretrained ImageNet shortcuts, and we strictly scope our system as **'Visible Crop/Leaf Condition Classification'**."*

### [0:15 - 0:30] The Training Journey & Notebook Evidence
* *(Switch to VS Code / Jupyter showing `FarmWatch_CNN_Training.ipynb`)*
* *"In our notebook, you can see our empirical journey on the PlantDoc benchmark (CC BY 4.0, IEEE Access 2023). Our initial 28-class baseline suffered severe imbalance with 13 starved classes scoring 0% F1."*
* *"We audited the distribution, grouped splits by SHA-256 image hashes to ensure zero leakage, and trained `EnhancedCropCNN` on 11 well-represented classes using Kaiming Normal initialization, class-weighted loss, and cosine scheduling."*

### [0:30 - 0:50] Real Metrics & Zero-Fabrication Honesty
* *(Switch to browser, click 'Benchmark Metrics & Model Transparency' tab)*
* *"We report 100% real numbers: on 214 held-out test images, our scratch-trained model achieved **27.10% accuracy** (vs 9.09% random chance) and **21.74% Macro F1**, with zero dead classes and an average inference latency of **14.05 ms** on our Intel Core Ultra 5 CPU."*
* *"Notice the confusion matrix: high diagonal concentration on Squash Powdery Mildew, Peach Leaf, and Corn Rust proves genuine visual feature learning from scratch."*

### [0:50 - 1:15] Live Inference & Grad-CAM Visual Attention
* *(Click 'Leaf Diagnosis & Grad-CAM' tab. Click the sample chip for 'Squash Powdery mildew leaf' or 'Corn rust leaf')*
* *"Now let's test a real held-out field image live. Our frontend features an interactive leaf ROI canvas allowing farmers to isolate a specific diseased leaf in a busy bush."*
* *(Click 'Run Leaf Diagnosis')*
* *"In under 15 milliseconds, the model returns the top predicted condition, model confidence, and top-3 ranked hypotheses."*
* *"Notice the Grad-CAM visualization generated from the final convolutional layer `block4[3]`. The warm red/yellow heatmap highlights the exact fungal patches on the leaf surface that drove the CNN's classification."*

### [1:15 - 1:30] Calibrated Uncertainty & Safe Agronomic Guidance
* *(Click a low-confidence sample or select another leaf)*
* *"Crucially, when model confidence falls below our calibrated 60% threshold, FarmWatch AI automatically displays an **Uncertainty Alert**, warning the farmer that visual features are ambiguous and advising physical inspection."*
* *"Furthermore, adhering to responsible agronomy, our engine provides **zero chemical pesticide dosages**—only safe cultural sanitation, tool disinfection, and referral to certified extension officers."*
* *"Thank you, and we welcome your questions!"*

---

## Anticipated Judge Q&A Cheat Sheet

1. **Q: Why is accuracy 27.10% instead of 95%+?**  
   * **A:** Most 95%+ papers use PlantVillage (single leaf placed flat on black cardboard in a studio) or transfer learning with 25M pretrained ImageNet weights. PlantDoc is wild unconstrained field images. Training an 11-class CNN from pure random weights on 951 noisy training images is a severe statistical challenge; 27.10% represents a 3x lift over random guessing (9.09%) and our macro F1 improved from 11.17% to 21.74% with zero dead classes.

2. **Q: Why don't you detect nitrogen deficiency or drought?**  
   * **A:** Scientific honesty: the PlantDoc dataset contains only visible fungal, bacterial, and viral foliar symptoms. Claiming drought or nutrient detection without certified field ground-truth would be synthetic and irresponsible.

3. **Q: How does Intel Arc compute fit in?**  
   * **A:** Our hardware probe `src/utils/device.py` dynamically queries for Intel Arc XPU support. In environments with Intel oneAPI client runtimes, it leverages XPU acceleration; on standard CPU setups, it falls back gracefully with multi-threaded Intel CPU optimizations achieving 14.05 ms per inference.
