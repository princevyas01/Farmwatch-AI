"""Held-out test set evaluation script for FarmWatch AI Enhanced CNN.

Computes:
- Accuracy, Balanced Accuracy, Macro Precision, Macro Recall, Macro F1
- Per-class metrics & support
- Confusion matrix plot saved to reports/focused_confusion_matrix.png
- Sample Grad-CAM overlay predictions saved to reports/focused_sample_predictions.png
- Machine-readable JSON metrics saved to reports/focused_test_metrics.json
- Direct comparative table against the 28-class baseline.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
    balanced_accuracy_score,
)
from PIL import Image

from src.models.custom_cnn import EnhancedCropCNN
from src.training.train_cnn import PlantDocDataset
from src.training.train_enhanced_cnn import get_enhanced_transforms
from src.explainability.gradcam import GradCAM
from src.utils.device import get_torch_device


def evaluate_enhanced_test_set(project_root: Path, checkpoint_path: Path = None) -> Dict[str, Any]:
    device = get_torch_device()
    if checkpoint_path is None:
        checkpoint_path = project_root / "models" / "enhanced_cnn.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"Loading enhanced checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    class_to_idx = checkpoint["class_to_idx"]
    num_classes = checkpoint["num_classes"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(num_classes)]

    model = EnhancedCropCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    test_manifest_path = project_root / "data" / "manifests" / "focused_test_manifest.json"
    _, eval_tf = get_enhanced_transforms((128, 128))
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
    bal_acc = float(balanced_accuracy_score(all_targets, all_preds))
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

    print(f"\n=======================================================")
    print(f"   ENHANCED MODEL HELD-OUT TEST EVALUATION RESULTS   ")
    print(f"=======================================================")
    print(f"Classes Evaluated:   {num_classes} well-represented field classes")
    print(f"Test Samples:        {len(test_dataset)}")
    print(f"Accuracy:            {accuracy * 100:.2f}%")
    print(f"Balanced Accuracy:   {bal_acc * 100:.2f}%")
    print(f"Macro Precision:     {macro_prec * 100:.2f}%")
    print(f"Macro Recall:        {macro_rec * 100:.2f}%")
    print(f"Macro F1-Score:      {macro_f1 * 100:.2f}%")
    print(f"Weighted F1-Score:   {weighted_f1 * 100:.2f}%")
    print(f"Mean Latency on CPU: {avg_latency:.2f} ms/sample")
    print(f"=======================================================\n")

    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
    plt.title(f"Enhanced CNN Confusion Matrix ({num_classes} Classes)", fontsize=13)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90, fontsize=9)
    plt.yticks(tick_marks, class_names, fontsize=9)
    plt.ylabel("True Class", fontsize=11)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.tight_layout()
    cm_path = reports_dir / "focused_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_path}")

    # Grad-CAM sample visualizations
    gradcam = GradCAM(model)
    np.random.seed(42)
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
                f"True: {true_name}\nPred: {pred_name} ({'OK' if is_correct else 'ERR'})",
                color=title_color,
                fontsize=9,
            )
            axes[idx].axis("off")

    plt.tight_layout()
    sample_path = reports_dir / "focused_sample_predictions.png"
    plt.savefig(sample_path, dpi=150)
    plt.close()
    gradcam.remove_hooks()
    print(f"Saved sample Grad-CAM predictions to {sample_path}")

    # Metrics JSON
    metrics_data = {
        "model_name": "EnhancedCropCNN",
        "dataset": "PlantDoc (Well-Represented Classes Subset)",
        "license": "CC BY 4.0",
        "checkpoint_evaluated": str(checkpoint_path.name),
        "test_samples_count": len(test_dataset),
        "num_classes": num_classes,
        "test_accuracy": round(accuracy, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_precision": round(float(macro_prec), 4),
        "macro_recall": round(float(macro_rec), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "average_latency_ms": round(avg_latency, 2),
        "per_class_f1": {class_names[i]: round(float(f1[i]), 4) for i in range(num_classes)},
        "per_class_precision": {class_names[i]: round(float(prec[i]), 4) for i in range(num_classes)},
        "per_class_recall": {class_names[i]: round(float(rec[i]), 4) for i in range(num_classes)},
        "per_class_support": {class_names[i]: int(support[i]) for i in range(num_classes)},
    }

    metrics_out = reports_dir / "focused_test_metrics.json"
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    # Load baseline for comparison
    baseline_path = reports_dir / "baseline_28class_metrics.json"
    if baseline_path.exists():
        with open(baseline_path, "r", encoding="utf-8") as f:
            base_m = json.load(f)

        print("\n=======================================================")
        print("          BASELINE VS ENHANCED MODEL COMPARISON         ")
        print("=======================================================")
        print(f"{'Metric':<20} | {'Baseline (28 Classes)':<22} | {'Enhanced (11 Classes)':<22}")
        print("-" * 70)
        print(f"{'Test Accuracy':<20} | {base_m['test_accuracy']*100:<21.2f}% | {accuracy*100:<21.2f}%")
        print(f"{'Macro Precision':<20} | {base_m['macro_precision']*100:<21.2f}% | {macro_prec*100:<21.2f}%")
        print(f"{'Macro Recall':<20} | {base_m['macro_recall']*100:<21.2f}% | {macro_rec*100:<21.2f}%")
        print(f"{'Macro F1':<20} | {base_m['macro_f1']*100:<21.2f}% | {macro_f1*100:<21.2f}%")
        print(f"{'Classes Evaluated':<20} | {base_m['num_classes']:<22} | {num_classes:<22}")
        print(f"{'Test Samples':<20} | {base_m['test_samples_count']:<22} | {len(test_dataset):<22}")
        print(f"{'Random Baseline':<20} | {100/base_m['num_classes']:<21.2f}% | {100/num_classes:<21.2f}%")
        print("=======================================================\n")

    return metrics_data


if __name__ == "__main__":
    evaluate_enhanced_test_set(PROJECT_ROOT)
