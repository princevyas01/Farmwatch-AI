# FarmWatch AI: Dataset Audit & Provenance Report

- **Dataset Source:** PlantDoc: A Dataset for Visual Plant Disease Detection (ACM CoDS-COMAD 2020)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Fallback Justification:** Roboflow Universe FieldPlant v11 export requires interactive browser authentication / private API key. PlantDoc was utilized as the authoritative open-access field-image leaf dataset.
- **Total Unique Images:** 2560
- **Number of Classes:** 28
- **Image Duplicates Removed:** 21 (Exact SHA-256 match filtered to prevent split contamination)
- **Corrupt Files Detected:** 0

## Split Partitioning (Image-Level Grouped)
- **Train Split (70%):** 1778 images
- **Validation Split (15%):** 372 images
- **Test Split (15% held-out):** 410 images
- **Leakage Prevention:** 100% verified. Every image SHA-256 is assigned strictly to a single split partition.

## Class Distribution Across Dataset
| Class Name | Image Count |
|---|---|
| Apple Scab Leaf | 93 |
| Apple leaf | 91 |
| Apple rust leaf | 88 |
| Bell_pepper leaf | 61 |
| Bell_pepper leaf spot | 71 |
| Blueberry leaf | 114 |
| Cherry leaf | 57 |
| Corn Gray leaf spot | 67 |
| Corn leaf blight | 189 |
| Corn rust leaf | 116 |
| Peach leaf | 111 |
| Potato leaf early blight | 115 |
| Potato leaf late blight | 101 |
| Raspberry leaf | 119 |
| Soyabean leaf | 65 |
| Squash Powdery mildew leaf | 130 |
| Strawberry leaf | 96 |
| Tomato Early blight leaf | 88 |
| Tomato Septoria leaf spot | 150 |
| Tomato leaf | 63 |
| Tomato leaf bacterial spot | 109 |
| Tomato leaf late blight | 111 |
| Tomato leaf mosaic virus | 54 |
| Tomato leaf yellow virus | 75 |
| Tomato mold leaf | 91 |
| Tomato two spotted spider mites leaf | 2 |
| grape leaf | 69 |
| grape leaf black rot | 64 |
