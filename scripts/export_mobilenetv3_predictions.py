"""Export detailed sample-level predictions and probabilities for MobileNetV3-Small.

Evaluates the saved MobileNetV3 checkpoint on the 214-image test manifest.
Generates:
- reports/mobilenetv3_test_predictions.csv
- reports/mobilenetv3_test_class_probabilities.csv
- reports/mobilenetv3_incorrect_predictions.csv
"""

import sys
import json
import csv
from pathlib import Path
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_mobilenetv3 import build_mobilenetv3_model, get_mobilenetv3_transforms
from src.training.train_cnn import PlantDocDataset
from src.utils.device import get_torch_device
from src.utils.dataset_paths import resolve_image_path, get_safe_path_str


def export_mobilenetv3_predictions():
    device = get_torch_device()
    checkpoint_path = PROJECT_ROOT / "models" / "mobilenetv3_small_transfer.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"Loading MobileNetV3 checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    class_to_idx = checkpoint["class_to_idx"]
    num_classes = checkpoint["num_classes"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(num_classes)]

    model = build_mobilenetv3_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    test_manifest_path = PROJECT_ROOT / "data" / "manifests" / "focused_test_manifest.json"
    with open(test_manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    _, eval_tf = get_mobilenetv3_transforms((224, 224))
    print(f"Evaluating {len(manifest)} test samples...")

    predictions_rows = []
    probabilities_rows = []
    incorrect_rows = []

    correct_count = 0
    total_count = len(manifest)

    for idx, item in enumerate(manifest):
        img_path = resolve_image_path(item)
        rel_path = item.get("relative_path", str(img_path.relative_to(PROJECT_ROOT)))
        sha256 = item.get("sha256", "")
        true_cls = item["class_name"]
        true_idx = class_to_idx[true_cls]

        safe_path = get_safe_path_str(img_path)
        pil_img = Image.open(safe_path).convert("RGB")
        tensor = eval_tf(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)

        # Sort probabilities descending
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        pred_idx = sorted_indices[0].item()
        pred_cls = idx_to_class[pred_idx]
        conf_score = sorted_probs[0].item()
        is_correct = (pred_cls == true_cls)

        if is_correct:
            correct_count += 1

        top1_cls = idx_to_class[sorted_indices[0].item()]
        top1_score = sorted_probs[0].item()
        top2_cls = idx_to_class[sorted_indices[1].item()]
        top2_score = sorted_probs[1].item()
        top3_cls = idx_to_class[sorted_indices[2].item()]
        top3_score = sorted_probs[2].item()

        # Row 1: predictions
        predictions_rows.append([
            idx,
            rel_path,
            sha256,
            true_cls,
            true_idx,
            pred_cls,
            pred_idx,
            is_correct,
            round(conf_score, 4),
            top1_cls,
            round(top1_score, 4),
            top2_cls,
            round(top2_score, 4),
            top3_cls,
            round(top3_score, 4),
        ])

        # Row 2: probabilities
        prob_row = [
            idx,
            rel_path,
            true_cls,
            pred_cls,
            is_correct,
        ] + [round(probs[c_i].item(), 6) for c_i in range(num_classes)]
        probabilities_rows.append(prob_row)

        # Row 3: incorrect
        if not is_correct:
            incorrect_rows.append([
                idx,
                rel_path,
                sha256,
                true_cls,
                pred_cls,
                round(conf_score, 4),
                top2_cls,
                round(top2_score, 4),
            ])

    print(f"Results: {correct_count}/{total_count} correct ({correct_count/total_count*100:.2f}% accuracy).")
    print(f"Incorrect: {len(incorrect_rows)}/{total_count} ({len(incorrect_rows)/total_count*100:.2f}% error rate).")

    reports_dir = PROJECT_ROOT / "reports"
    audit_dir = reports_dir / "model_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    # 1. Write predictions CSV
    pred_header = [
        "sample_id", "relative_image_path", "image_sha256", "true_class", "true_class_index",
        "predicted_class", "predicted_class_index", "is_correct", "confidence_score",
        "top_1_class", "top_1_score", "top_2_class", "top_2_score", "top_3_class", "top_3_score"
    ]
    for out_dir in [reports_dir, audit_dir]:
        out_csv = out_dir / "mobilenetv3_test_predictions.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(pred_header)
            writer.writerows(predictions_rows)
        print(f"Saved: {out_csv}")

    # 2. Write probabilities CSV
    prob_header = [
        "sample_id", "relative_image_path", "true_class", "predicted_class", "is_correct"
    ] + [f"prob_{c}" for c in class_names]
    for out_dir in [reports_dir, audit_dir]:
        out_csv = out_dir / "mobilenetv3_test_class_probabilities.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(prob_header)
            writer.writerows(probabilities_rows)
        print(f"Saved: {out_csv}")

    # 3. Write incorrect predictions CSV
    inc_header = [
        "sample_id", "relative_image_path", "image_sha256", "true_class", "predicted_class",
        "confidence_score", "top_2_class", "top_2_score"
    ]
    for out_dir in [reports_dir, audit_dir]:
        out_csv = out_dir / "mobilenetv3_incorrect_predictions.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(inc_header)
            writer.writerows(incorrect_rows)
        print(f"Saved: {out_csv}")


if __name__ == "__main__":
    export_mobilenetv3_predictions()
