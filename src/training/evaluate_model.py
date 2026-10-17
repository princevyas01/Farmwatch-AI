"""Held-out test set evaluation script for FarmWatch AI trained CNN.

Calculates:
- Actual accuracy, macro precision, macro recall, and macro F1
- Per-class classification metrics
- Confusion matrix plot saved to reports/confusion_matrix.png
- Average inference latency (ms/sample)
- Sample prediction visualizations with Grad-CAM saved to reports/sample_predictions.png
- Machine-readable test metrics saved to reports/test_metrics.json
"""

import os
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
from PIL import Image

from src.models.custom_cnn import CustomCropCNN
from src.training.train_cnn import PlantDocDataset, get_transforms
from src.explainability.gradcam import GradCAM
from src.utils.device import get_torch_device


def evaluate_test_set(project_root: Path, checkpoint_path: Path = None) -> Dict[str, Any]:
    """Execute evaluation on the held-out test manifest."""
    device = get_torch_device()
    if checkpoint_path is None:
        checkpoint_path = project_root / "models" / "cnn_from_scratch.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    class_to_idx = checkpoint["class_to_idx"]
    num_classes = checkpoint["num_classes"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(num_classes)]

    model = CustomCropCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    test_manifest_path = project_root / "data" / "manifests" / "test_manifest.json"
    _, eval_tf = get_transforms(tuple(checkpoint.get("input_size", [128, 128])))
    test_dataset = PlantDocDataset(test_manifest_path, class_to_idx, transform=eval_tf)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"Evaluating {len(test_dataset)} untouched held-out test images on {device}...")

    all_preds = []
    all_targets = []
    all_scores = []
    latencies = []

    for images, labels, _ in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        start_t = time.perf_counter()
        with torch.no_grad():
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            score, pred = torch.max(probs, dim=1)
        latencies.append((time.perf_counter() - start_t) * 1000.0)

        all_preds.append(pred.item())
        all_targets.append(labels.item())
        all_scores.append(score.item())

    accuracy = float(accuracy_score(all_targets, all_preds))
    prec, rec, f1, support = precision_recall_fscore_support(
        all_targets, all_preds, average=None, zero_division=0, labels=list(range(num_classes))
    )
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="weighted", zero_division=0
    )

    avg_latency = float(np.mean(latencies))
    print(f"\n--- HELD-OUT TEST EVALUATION RESULTS ---")
    print(f"Accuracy:        {accuracy * 100:.2f}%")
    print(f"Macro Precision: {macro_prec * 100:.2f}%")
    print(f"Macro Recall:    {macro_rec * 100:.2f}%")
    print(f"Macro F1-Score:  {macro_f1 * 100:.2f}%")
    print(f"Mean Latency:    {avg_latency:.2f} ms/sample")

    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))
    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
    plt.title("Held-Out Test Confusion Matrix (28 Classes)", fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90, fontsize=8)
    plt.yticks(tick_marks, class_names, fontsize=8)
    plt.ylabel("True Class", fontsize=11)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.tight_layout()
    cm_path = reports_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_path}")

    # Generate sample prediction visualization with Grad-CAM
    gradcam = GradCAM(model)
    sample_indices = np.random.choice(len(test_dataset), size=min(6, len(test_dataset)), replace=False)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, sample_idx in enumerate(sample_indices):
        item = test_dataset.items[sample_idx]
        file_path = item["file_path"]
        if os.name == "nt" and len(file_path) > 250 and not file_path.startswith("\\\\?\\"):
            safe_p = f"\\\\?\\{file_path}"
        else:
            safe_p = file_path

        with Image.open(safe_p) as pil_img:
            pil_rgb = pil_img.convert("RGB")
            crop_tensor = eval_tf(pil_rgb).unsqueeze(0).to(device)
            heatmap, pred_c, _ = gradcam.generate_heatmap(crop_tensor)
            overlay = gradcam.create_overlay_image(pil_rgb.resize((128, 128)), heatmap)

            pred_name = idx_to_class[pred_c]
            true_name = item["class_name"]
            is_correct = pred_name == true_name

            axes[idx].imshow(overlay)
            title_color = "#2d6a4f" if is_correct else "#d90429"
            axes[idx].set_title(
                f"True: {true_name[:20]}\nPred: {pred_name[:20]} ({'OK' if is_correct else 'ERR'})",
                color=title_color,
                fontsize=10,
            )
            axes[idx].axis("off")

    plt.tight_layout()
    sample_path = reports_dir / "sample_predictions.png"
    plt.savefig(sample_path, dpi=150)
    plt.close()
    gradcam.remove_hooks()
    print(f"Saved sample Grad-CAM predictions to {sample_path}")

    # Machine-readable test metrics
    metrics_data = {
        "dataset": "PlantDoc (Field Imagery Benchmark)",
        "license": "CC BY 4.0",
        "citation": "PlantDoc: A Dataset for Visual Plant Disease Detection (ACM CoDS-COMAD 2020)",
        "checkpoint_evaluated": str(checkpoint_path.name),
        "test_samples_count": len(test_dataset),
        "num_classes": num_classes,
        "test_accuracy": round(accuracy, 4),
        "macro_precision": round(float(macro_prec), 4),
        "macro_recall": round(float(macro_rec), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "average_latency_ms": round(avg_latency, 2),
        "per_class_f1": {class_names[i]: round(float(f1[i]), 4) for i in range(num_classes)},
        "per_class_support": {class_names[i]: int(support[i]) for i in range(num_classes)},
    }

    with open(reports_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    # Also save to artifacts/reports/
    (project_root / "artifacts" / "reports").mkdir(parents=True, exist_ok=True)
    with open(project_root / "artifacts" / "reports" / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    print(f"Saved test metrics report to {reports_dir / 'test_metrics.json'}")
    return metrics_data


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent.parent
    evaluate_test_set(root)
