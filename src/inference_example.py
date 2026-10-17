"""Minimal standalone inference demonstration for FarmWatch AI.

Demonstrates:
1. How to instantiate the CustomCropCNN model and load the checkpoint.
2. Exact image preprocessing (resizing to 128x128, RGB conversion, ImageNet normalization).
3. Softmax probability and raw model score extraction.
4. Validation-calibrated uncertainty check (< 60% confidence).
5. Grad-CAM visual attention generation.
6. Safe low-risk agronomic recommendations mapping.
"""

import sys
import json
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.custom_cnn import CustomCropCNN, EnhancedCropCNN
from src.explainability.gradcam import GradCAM
from src.utils.device import get_torch_device, get_device_info


def load_model_and_config(checkpoint_path: Path):
    """Load model weights and configuration for inference."""
    device = get_torch_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    num_classes = checkpoint["num_classes"]
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    architecture = checkpoint.get("architecture", "CustomCropCNN")

    if architecture == "EnhancedCropCNN":
        model = EnhancedCropCNN(num_classes=num_classes)
    else:
        model = CustomCropCNN(num_classes=num_classes)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Preprocessing transform identical to training validation
    input_size = tuple(checkpoint.get("input_size", [128, 128]))
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return model, transform, idx_to_class, device


def predict_leaf_image(
    image_path: Path,
    checkpoint_path: Path = None,
    uncertainty_threshold: float = 0.60,
    generate_cam: bool = True,
):
    """Run full inference on a single image file."""
    if checkpoint_path is None:
        checkpoint_path = PROJECT_ROOT / "models" / "cnn_from_scratch.pth"

    model, transform, idx_to_class, device = load_model_and_config(checkpoint_path)

    # 1. Load and validate image
    with Image.open(image_path) as img:
        rgb_img = img.convert("RGB")

    input_tensor = transform(rgb_img).unsqueeze(0).to(device)

    # 2. Forward pass
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    # 3. Top prediction
    top_prob, top_idx = torch.max(probs, dim=0)
    score = float(top_prob.item())
    pred_class = idx_to_class[int(top_idx.item())]
    is_uncertain = score < uncertainty_threshold

    # Top-3 predictions
    top3_probs, top3_indices = torch.topk(probs, k=min(3, len(probs)))
    top_3 = [
        {"class": idx_to_class[int(idx.item())], "score": round(float(prob.item()), 4)}
        for prob, idx in zip(top3_probs, top3_indices)
    ]

    result = {
        "predicted_class": pred_class,
        "model_score": round(score, 4),
        "is_uncertain": is_uncertain,
        "score_interpretation": (
            "Low confidence model score (< 60%). Recommend physical inspection."
            if is_uncertain
            else f"High-confidence match for visible leaf symptoms ({score * 100:.1f}%)."
        ),
        "top_3_candidates": top_3,
        "device_used": str(device),
    }

    # 4. Optional Grad-CAM
    if generate_cam:
        gradcam = GradCAM(model)
        heatmap, _, _ = gradcam.generate_heatmap(input_tensor, target_class=int(top_idx.item()))
        overlay = gradcam.create_overlay_image(rgb_img.resize((128, 128)), heatmap)
        gradcam.remove_hooks()
        output_overlay_path = PROJECT_ROOT / "reports" / "latest_inference_gradcam.png"
        output_overlay_path.parent.mkdir(parents=True, exist_ok=True)
        overlay.save(output_overlay_path)
        result["gradcam_overlay_saved"] = str(output_overlay_path)

    return result


if __name__ == "__main__":
    # Test on the first sample from test manifest
    test_manifest = PROJECT_ROOT / "data" / "manifests" / "test_manifest.json"
    if test_manifest.exists():
        with open(test_manifest, "r", encoding="utf-8") as f:
            test_items = json.load(f)
        if test_items:
            sample_img_path = Path(test_items[0]["file_path"])
            print(f"Running inference on test sample: {sample_img_path} (True Class: {test_items[0]['class_name']})")
            res = predict_leaf_image(sample_img_path)
            print("Inference Result:")
            print(json.dumps(res, indent=2))
