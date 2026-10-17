"""Packaging script to create a standalone, shareable bundle of the FarmWatch AI models.

Generates:
`dist/farmwatch_model_bundle.zip` containing:
- mobilenetv3_small_transfer.pth (Primary Transfer-Learning Model, 67.29% Test Acc)
- enhanced_cnn.pth (Preserved Scratch Baseline, 27.10% Test Acc)
- mobilenetv3_class_to_idx.json & focused_class_to_idx.json
- mobilenetv3_preprocess_config.json & focused_preprocess_config.json
- minimal_inference.py (Self-contained script supporting both models)
- MODEL_CARD.md (Full architectural comparison, metrics, and citations)
"""

import os
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = PROJECT_ROOT / "dist"
DIST_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_ZIP = DIST_DIR / "farmwatch_model_bundle.zip"

MINIMAL_INFERENCE_CODE = '''"""FarmWatch AI - Minimal Standalone Inference Snippet.

Supports both MobileNetV3-Small (Transfer Learning) and EnhancedCropCNN (Scratch Baseline).
Requires: torch, torchvision, Pillow
"""
import json
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torchvision.models as models
from torchvision.models import MobileNet_V3_Small_Weights


# 1. Custom CNN Architecture
class EnhancedCropCNN(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.block4(self.block3(self.block2(self.block1(x))))
        return self.classifier(self.global_pool(x))


def build_mobilenetv3(num_classes=11):
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(1024, num_classes)
    return model


def run_inference(image_path: str, model_type: str = "mobilenetv3"):
    """Run inference using either 'mobilenetv3' or 'enhanced_cnn'."""
    if model_type == "mobilenetv3":
        checkpoint_path = "mobilenetv3_small_transfer.pth"
        class_map_path = "mobilenetv3_class_to_idx.json"
        img_size = (224, 224)
        with open(class_map_path, "r", encoding="utf-8") as f:
            class_to_idx = json.load(f)
        model = build_mobilenetv3(num_classes=len(class_to_idx))
    else:
        checkpoint_path = "enhanced_cnn.pth"
        class_map_path = "focused_class_to_idx.json"
        img_size = (128, 128)
        with open(class_map_path, "r", encoding="utf-8") as f:
            class_to_idx = json.load(f)
        model = EnhancedCropCNN(num_classes=len(class_to_idx))

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    score, idx = torch.max(probs, dim=0)
    top_class = idx_to_class[int(idx.item())]
    confidence = float(score.item())

    return {
        "model_type": model_type,
        "predicted_class": top_class,
        "model_score": round(confidence, 4),
        "is_uncertain": confidence < 0.60,
    }


if __name__ == "__main__":
    print("Example usage:")
    print("  result = run_inference('test_leaf.jpg', model_type='mobilenetv3')")
    print("  print(result)")
'''

MODEL_CARD_MD = """# FarmWatch AI — Model Card

## Summary
- **Hackathon:** GenHack V1.0 PS 12 (Predicting Crop Stress From Field Images)
- **Primary Model:** MobileNetV3-Small (Transfer Learning from ImageNet-1K)
  - Test Accuracy: **67.29%** | Macro F1: **66.52%** | Latency: **11.09 ms** (CPU)
- **Baseline Model:** EnhancedCropCNN (Scratch-Trained Baseline, Kaiming Normal)
  - Test Accuracy: **27.10%** | Macro F1: **21.74%** | Latency: **14.05 ms** (CPU)
- **Task:** Visible Crop/Leaf Condition Classification
- **Dataset:** PlantDoc (CC BY 4.0, 11 well-represented classes, zero-leakage splits)

## 11 Classes
1. Blueberry leaf
2. Corn leaf blight
3. Corn rust leaf
4. Peach leaf
5. Potato leaf early blight
6. Potato leaf late blight
7. Raspberry leaf
8. Squash Powdery mildew leaf
9. Tomato Septoria leaf spot
10. Tomato leaf bacterial spot
11. Tomato leaf late blight

## Usage
Extract the zip and run `python minimal_inference.py`.
"""


def package_bundle():
    print(f"Creating model release bundle at: {OUTPUT_ZIP}")
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        # Add weights and configs for both models
        z.write(PROJECT_ROOT / "models" / "mobilenetv3_small_transfer.pth", arcname="mobilenetv3_small_transfer.pth")
        z.write(PROJECT_ROOT / "models" / "enhanced_cnn.pth", arcname="enhanced_cnn.pth")
        z.write(PROJECT_ROOT / "models" / "mobilenetv3_class_to_idx.json", arcname="mobilenetv3_class_to_idx.json")
        z.write(PROJECT_ROOT / "models" / "focused_class_to_idx.json", arcname="focused_class_to_idx.json")
        z.write(PROJECT_ROOT / "models" / "mobilenetv3_preprocess_config.json", arcname="mobilenetv3_preprocess_config.json")
        z.write(PROJECT_ROOT / "models" / "focused_preprocess_config.json", arcname="focused_preprocess_config.json")

        # Add minimal inference script and model card
        z.writestr("minimal_inference.py", MINIMAL_INFERENCE_CODE.strip())
        z.writestr("MODEL_CARD.md", MODEL_CARD_MD.strip())

    file_size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"Successfully packaged {OUTPUT_ZIP.name} ({file_size_mb:.2f} MB)")


if __name__ == "__main__":
    package_bundle()
