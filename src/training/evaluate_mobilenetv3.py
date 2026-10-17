"""Held-out test set evaluation script for MobileNetV3-Small Transfer Learning Model.

Evaluates on the exact untouched 214 held-out test images:
- Accuracy, Balanced Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1
- Per-class metrics & support
- Confusion matrix plot saved to reports/mobilenetv3_confusion_matrix.png
- Sample Grad-CAM overlays saved to reports/mobilenetv3_sample_predictions.png
- Machine-readable JSON metrics saved to reports/mobilenetv3_test_metrics.json
- Direct comparative table against the scratch-trained EnhancedCropCNN baseline (214 identical test samples)
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List

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

from src.training.train_cnn import PlantDocDataset
from src.training.train_mobilenetv3 import build_mobilenetv3_model, get_mobilenetv3_transforms
from src.explainability.gradcam import GradCAM
from src.utils.device import get_torch_device, get_device_info
from src.utils.dataset_paths import resolve_image_path, get_safe_path_str


def evaluate_mobilenetv3_test_set(
    project_root: Path = PROJECT_ROOT,
    checkpoint_path: Path = None,
) -> Dict[str, Any]:
    device = get_torch_device()
    device_info = get_device_info()

    if checkpoint_path is None:
        checkpoint_path = project_root / "models" / "mobilenetv3_small_transfer.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Candidate checkpoint not found at: {checkpoint_path}")

    print(f"Loading MobileNetV3-Small checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    class_to_idx = checkpoint["class_to_idx"]
    num_classes = checkpoint["num_classes"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(num_classes)]

    # Instantiate model and load state
    model = build_mobilenetv3_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    test_manifest_path = project_root / "data" / "manifests" / "focused_test_manifest.json"
    _, eval_tf = get_mobilenetv3_transforms((224, 224))
    test_dataset = PlantDocDataset(test_manifest_path, class_to_idx, transform=eval_tf)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"Evaluating {len(test_dataset)} untouched held-out test images on {device} ({device_info.get('device_name', '')})...")

    # Warm-up (10 passes)
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

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

    # Compute comprehensive metrics
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
    p95_latency = float(np.percentile(latencies, 95))
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))

    print("\n" + "=" * 65)
    print("MobileNetV3-Small (Transfer Learning) - Test Evaluation Summary")
    print("=" * 65)
    print(f"Test Accuracy:         {accuracy * 100:.2f}%")
    print(f"Balanced Accuracy:     {bal_acc * 100:.2f}%")
    print(f"Macro Precision:       {macro_prec * 100:.2f}%")
    print(f"Macro Recall:          {macro_rec * 100:.2f}%")
    print(f"Macro F1-Score:        {macro_f1 * 100:.2f}%")
    print(f"Weighted F1-Score:     {weighted_f1 * 100:.2f}%")
    print(f"Average CPU Latency:   {avg_latency:.2f} ms (p95: {p95_latency:.2f} ms)")
    print("=" * 65)

    print("\nPer-Class Breakdown:")
    per_class_dict = {}
    for i, name in enumerate(class_names):
        per_class_dict[name] = {
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1_score": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        print(f"  {name:30s} | P: {prec[i]*100:5.1f}% | R: {rec[i]*100:5.1f}% | F1: {f1[i]*100:5.1f}% | Supp: {support[i]:2d}")

    # Plot Confusion Matrix
    reports_dir = project_root / "reports"
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
    plt.title("Confusion Matrix - MobileNetV3-Small (11 Well-Represented Classes)")
    plt.colorbar(label="Number of images")
    tick_marks = np.arange(len(class_names))
    short_names = [n.replace(" leaf", "").replace("Tomato leaf ", "Tom_") for n in class_names]
    plt.xticks(tick_marks, short_names, rotation=45, ha="right", fontsize=9)
    plt.yticks(tick_marks, short_names, fontsize=9)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    plt.tight_layout()
    plt.ylabel("Ground Truth")
    plt.xlabel("Predicted Class")
    cm_path = reports_dir / "mobilenetv3_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved confusion matrix plot to: {cm_path}")

    # Generate sample Grad-CAM overlays (pick 4 representative test samples)
    sample_indices = [0, min(10, len(test_dataset)-1), min(25, len(test_dataset)-1), min(50, len(test_dataset)-1)]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Identify target conv layer for Grad-CAM in MobileNetV3-Small
    # features[-1] is Conv2dNormActivation with the final 1x1 conv layer
    target_conv_layer = model.features[-1]

    with GradCAM(model, target_layer=target_conv_layer) as gradcam:
        for ax_idx, s_idx in enumerate(sample_indices):
            item = test_dataset.items[s_idx]
            img_path = resolve_image_path(item)
            true_cls = item["class_name"]

            safe_path = get_safe_path_str(img_path)
            raw_img = Image.open(safe_path).convert("RGB")
            input_tensor = eval_tf(raw_img).unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(input_tensor)
                prob = torch.softmax(out, dim=1)
                score, pred_idx_val = torch.max(prob, dim=1)

            pred_cls = idx_to_class[pred_idx_val.item()]

            # GradCAM
            heatmap, _, _ = gradcam.generate_heatmap(input_tensor, target_class=pred_idx_val.item())
            resized_raw = raw_img.resize((224, 224))
            overlay = gradcam.create_overlay_image(resized_raw, heatmap, alpha=0.5)

            axes[ax_idx].imshow(overlay)
            axes[ax_idx].set_title(f"True: {true_cls[:12]}\nPred: {pred_cls[:12]} ({score.item()*100:.1f}%)", fontsize=9)
            axes[ax_idx].axis("off")

    plt.tight_layout()
    sample_preds_path = reports_dir / "mobilenetv3_sample_predictions.png"
    plt.savefig(sample_preds_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved sample Grad-CAM predictions to: {sample_preds_path}")

    # Head-to-Head Comparison with Baseline
    baseline_metrics_path = reports_dir / "focused_test_metrics.json"
    baseline_metrics = {}
    if baseline_metrics_path.exists():
        with open(baseline_metrics_path, "r", encoding="utf-8") as f:
            baseline_metrics = json.load(f)

    comparison = {
        "dataset": "PlantDoc (11 Well-Represented Classes)",
        "test_samples_count": len(test_dataset),
        "scratch_trained_baseline": {
            "model_name": baseline_metrics.get("model_name", "EnhancedCropCNN"),
            "architecture": "Custom CNN (EnhancedCropCNN)",
            "is_pretrained": False,
            "test_accuracy": baseline_metrics.get("test_accuracy"),
            "balanced_accuracy": baseline_metrics.get("balanced_accuracy"),
            "macro_precision": baseline_metrics.get("macro_precision"),
            "macro_recall": baseline_metrics.get("macro_recall"),
            "macro_f1": baseline_metrics.get("macro_f1"),
            "weighted_f1": baseline_metrics.get("weighted_f1"),
            "average_latency_ms": baseline_metrics.get("average_latency_ms"),
        },
        "transfer_learning_candidate": {
            "model_name": "MobileNetV3-Small",
            "architecture": "MobileNetV3-Small (ImageNet-1K pretrained)",
            "is_pretrained": True,
            "test_accuracy": round(accuracy, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "macro_precision": round(macro_prec, 4),
            "macro_recall": round(macro_rec, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "average_latency_ms": round(avg_latency, 2),
            "p95_latency_ms": round(p95_latency, 2),
        },
        "delta": {
            "test_accuracy_diff": round(accuracy - (baseline_metrics.get("test_accuracy") or 0.0), 4),
            "macro_f1_diff": round(macro_f1 - (baseline_metrics.get("macro_f1") or 0.0), 4),
            "balanced_accuracy_diff": round(bal_acc - (baseline_metrics.get("balanced_accuracy") or 0.0), 4),
            "latency_diff_ms": round(avg_latency - (baseline_metrics.get("average_latency_ms") or 0.0), 2),
        },
    }

    with open(reports_dir / "mobilenetv3_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    # Save test metrics JSON
    metrics_result = {
        "model_name": "MobileNetV3-Small",
        "dataset": "PlantDoc (Well-Represented Classes Subset)",
        "license": "CC BY 4.0",
        "checkpoint_evaluated": checkpoint_path.name,
        "is_pretrained": True,
        "pretrained_weights": "ImageNet-1K",
        "test_samples_count": len(test_dataset),
        "num_classes": num_classes,
        "test_accuracy": round(accuracy, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "average_latency_ms": round(avg_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "class_names": class_names,
        "per_class_f1": {name: data["f1_score"] for name, data in per_class_dict.items()},
        "per_class_precision": {name: data["precision"] for name, data in per_class_dict.items()},
        "per_class_recall": {name: data["recall"] for name, data in per_class_dict.items()},
        "per_class_support": {name: data["support"] for name, data in per_class_dict.items()},
        "per_class_metrics": per_class_dict,
        "confusion_matrix": cm.tolist(),
        "active_device": str(device),
        "device_name": device_info.get("device_name", ""),
    }

    with open(reports_dir / "mobilenetv3_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_result, f, indent=2)

    print("\nHead-to-Head Comparison with Scratch Baseline:")
    print(f"  Test Accuracy:     Baseline = {comparison['scratch_trained_baseline']['test_accuracy']*100:.2f}% | MobileNetV3 = {comparison['transfer_learning_candidate']['test_accuracy']*100:.2f}% (Delta: {comparison['delta']['test_accuracy_diff']*100:+.2f}%)")
    print(f"  Macro F1:          Baseline = {comparison['scratch_trained_baseline']['macro_f1']*100:.2f}% | MobileNetV3 = {comparison['transfer_learning_candidate']['macro_f1']*100:.2f}% (Delta: {comparison['delta']['macro_f1_diff']*100:+.2f}%)")
    print(f"  Balanced Accuracy: Baseline = {comparison['scratch_trained_baseline']['balanced_accuracy']*100:.2f}% | MobileNetV3 = {comparison['transfer_learning_candidate']['balanced_accuracy']*100:.2f}% (Delta: {comparison['delta']['balanced_accuracy_diff']*100:+.2f}%)")
    print(f"  Avg Latency:       Baseline = {comparison['scratch_trained_baseline']['average_latency_ms']:.2f} ms | MobileNetV3 = {comparison['transfer_learning_candidate']['average_latency_ms']:.2f} ms")

    return metrics_result


if __name__ == "__main__":
    evaluate_mobilenetv3_test_set()
