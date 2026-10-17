"""Evaluate MobileNetV3-Small confidence distribution and uncertainty handling on validation set.

Evaluates on the 200-image validation manifest (data/manifests/focused_val_manifest.json).
Computes:
- Overall validation accuracy
- Confidence distribution (mean, std, percentiles)
- Heuristic threshold (0.60) coverage and accuracy
- Binned calibration/confidence distribution
"""

import sys
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_mobilenetv3 import build_mobilenetv3_model, get_mobilenetv3_transforms
from src.utils.device import get_torch_device
from src.utils.dataset_paths import resolve_image_path, get_safe_path_str


def evaluate_validation_uncertainty():
    device = get_torch_device()
    checkpoint_path = PROJECT_ROOT / "models" / "mobilenetv3_small_transfer.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"Loading MobileNetV3 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_to_idx = checkpoint["class_to_idx"]
    num_classes = checkpoint["num_classes"]

    model = build_mobilenetv3_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    _, eval_tf = get_mobilenetv3_transforms((224, 224))

    val_manifest_path = PROJECT_ROOT / "data" / "manifests" / "focused_val_manifest.json"
    with open(val_manifest_path, "r", encoding="utf-8") as f:
        val_manifest = json.load(f)

    confs = []
    corrects = []

    for item in val_manifest:
        img_path = resolve_image_path(item)
        true_class = item["class_name"]
        true_idx = class_to_idx[true_class]
        safe_path = get_safe_path_str(img_path)
        with Image.open(safe_path) as img:
            img = img.convert("RGB")
            tensor = eval_tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_idx = int(np.argmax(probs))
        conf = float(probs[pred_idx])
        is_correct = (pred_idx == true_idx)
        confs.append(conf)
        corrects.append(is_correct)

    confs = np.array(confs)
    corrects = np.array(corrects)

    total = len(confs)
    overall_acc = float(np.mean(corrects))
    above_60 = confs >= 0.60
    coverage_60 = float(np.mean(above_60))
    acc_above_60 = float(np.mean(corrects[above_60])) if np.sum(above_60) > 0 else 0.0
    acc_below_60 = float(np.mean(corrects[~above_60])) if np.sum(~above_60) > 0 else 0.0

    bins = [0.0, 0.4, 0.6, 0.8, 1.01]
    bin_results = []
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        mask = (confs >= low) & (confs < high)
        n = int(np.sum(mask))
        acc = float(np.mean(corrects[mask])) if n > 0 else 0.0
        bin_results.append({
            "range": f"[{low:.1f}, {min(high, 1.0):.1f})",
            "count": n,
            "percentage": round(n / total * 100, 2),
            "accuracy": round(acc * 100, 2),
        })

    results = {
        "model": "MobileNetV3-Small Transfer Learning",
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "dataset_split": "focused_val_manifest.json (held-out validation partition)",
        "sample_count": total,
        "calibration_status": "uncalibrated_raw_softmax",
        "threshold_type": "heuristic",
        "confidence_distribution": {
            "mean": round(float(np.mean(confs)), 4),
            "std": round(float(np.std(confs)), 4),
            "min": round(float(np.min(confs)), 4),
            "p25": round(float(np.percentile(confs, 25)), 4),
            "median": round(float(np.percentile(confs, 50)), 4),
            "p75": round(float(np.percentile(confs, 75)), 4),
            "max": round(float(np.max(confs)), 4)
        },
        "overall_accuracy_pct": round(overall_acc * 100, 2),
        "heuristic_threshold_0_60": {
            "threshold": 0.60,
            "coverage_pct": round(coverage_60 * 100, 2),
            "covered_count": int(np.sum(above_60)),
            "accuracy_covered_pct": round(acc_above_60 * 100, 2),
            "uncovered_count": int(np.sum(~above_60)),
            "accuracy_uncovered_pct": round(acc_below_60 * 100, 2)
        },
        "confidence_bins": bin_results
    }

    out_file1 = PROJECT_ROOT / "reports" / "validation_uncertainty_metrics.json"
    out_file2 = PROJECT_ROOT / "reports" / "model_audit" / "validation_uncertainty_metrics.json"

    for p in [out_file1, out_file2]:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    print(f"Saved validation uncertainty metrics to {out_file1} and {out_file2}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    evaluate_validation_uncertainty()
