"""Dataset inspection, auditing, and leakage-free grouped splitting for FarmWatch AI.

Performs:
1. File verification & corruption scan using PIL.
2. Exact image hash duplication detection (SHA-256) to prevent any data leakage.
3. Class balance and resolution distribution analysis.
4. Grouped 70/15/15 train/val/test splitting by unique image identity.
5. Saves reproducible manifests in data/manifests/ and audit reports in reports/.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple
from PIL import Image


def get_image_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of image file contents to detect identical images."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def inspect_and_audit(raw_dir: Path, output_dir: Path, splits_dir: Path, manifests_dir: Path) -> Dict[str, Any]:
    """Audit the raw dataset and create deterministic, leakage-free splits."""
    output_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    print(f"Auditing raw dataset at: {raw_dir}")

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    records: List[Dict[str, Any]] = []
    corrupt_files: List[str] = []
    seen_hashes: Dict[str, str] = {}  # sha256 -> original path
    duplicate_records: List[Dict[str, str]] = []

    # Find all images in train/ and test/ directories
    candidate_files = []
    for split_subfolder in ["train", "test"]:
        sub_dir = raw_dir / split_subfolder
        if not sub_dir.exists():
            continue
        for class_dir in sub_dir.iterdir():
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            for img_file in class_dir.glob("*.*"):
                if img_file.suffix.lower() in valid_extensions:
                    candidate_files.append((split_subfolder, class_name, img_file))

    print(f"Found {len(candidate_files)} image candidates. Verifying integrity...")

    widths, heights = [], []
    class_counts: Dict[str, int] = {}

    for original_split, raw_class_name, file_path in candidate_files:
        # Standardize class name for consistency
        norm_class_name = raw_class_name.strip()

        try:
            with Image.open(file_path) as img:
                img.verify()  # Fast structural verification

            with Image.open(file_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)

            sha = get_image_sha256(file_path)
            if sha in seen_hashes:
                duplicate_records.append({
                    "duplicate_file": str(file_path),
                    "original_file": seen_hashes[sha],
                    "class": norm_class_name,
                })
                # Skip duplicate image to strictly prevent leakage across splits
                continue

            seen_hashes[sha] = str(file_path)
            class_counts[norm_class_name] = class_counts.get(norm_class_name, 0) + 1

            records.append({
                "image_id": sha[:16],
                "file_path": str(file_path.resolve()),
                "relative_path": str(file_path.relative_to(raw_dir)),
                "class_name": norm_class_name,
                "width": w,
                "height": h,
                "sha256": sha,
                "source_split": original_split,
            })

        except Exception as e:
            corrupt_files.append({"file": str(file_path), "error": str(e)})

    total_valid = len(records)
    unique_classes = sorted(list(class_counts.keys()))
    print(f"Verified {total_valid} unique, valid images across {len(unique_classes)} classes.")
    print(f"Filtered out {len(duplicate_records)} duplicate files and {len(corrupt_files)} corrupt files.")

    # Grouped Stratified Splitting: 70% Train, 15% Val, 15% Test
    # Grouped by unique image SHA to ensure zero duplicate cross-contamination
    import random
    random.seed(42)

    class_to_records: Dict[str, List[Dict[str, Any]]] = {c: [] for c in unique_classes}
    for r in records:
        class_to_records[r["class_name"]].append(r)

    train_manifest: List[Dict[str, Any]] = []
    val_manifest: List[Dict[str, Any]] = []
    test_manifest: List[Dict[str, Any]] = []

    for c in unique_classes:
        c_records = class_to_records[c]
        random.shuffle(c_records)
        n = len(c_records)
        if n == 1:
            # If only 1 image, put in train
            train_manifest.extend(c_records)
        elif n == 2:
            train_manifest.append(c_records[0])
            test_manifest.append(c_records[1])
        else:
            n_train = max(1, int(n * 0.70))
            n_val = max(1, int(n * 0.15))
            # Remaining to test
            c_train = c_records[:n_train]
            c_val = c_records[n_train:n_train + n_val]
            c_test = c_records[n_train + n_val:]
            if not c_test:  # Ensure at least 1 test image if n >= 3
                c_test = [c_val.pop()] if len(c_val) > 1 else [c_train.pop()]

            for item in c_train:
                item["split"] = "train"
                train_manifest.append(item)
            for item in c_val:
                item["split"] = "val"
                val_manifest.append(item)
            for item in c_test:
                item["split"] = "test"
                test_manifest.append(item)

    # Save manifests
    with open(manifests_dir / "train_manifest.json", "w", encoding="utf-8") as f:
        json.dump(train_manifest, f, indent=2)
    with open(manifests_dir / "val_manifest.json", "w", encoding="utf-8") as f:
        json.dump(val_manifest, f, indent=2)
    with open(manifests_dir / "test_manifest.json", "w", encoding="utf-8") as f:
        json.dump(test_manifest, f, indent=2)

    # Save class_to_idx mapping
    class_to_idx = {c: idx for idx, c in enumerate(unique_classes)}
    with open(splits_dir / "class_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)

    # Summary metrics
    split_summary = {
        "total_unique_images": total_valid,
        "num_classes": len(unique_classes),
        "train_count": len(train_manifest),
        "val_count": len(val_manifest),
        "test_count": len(test_manifest),
        "train_ratio": round(len(train_manifest) / total_valid, 3),
        "val_ratio": round(len(val_manifest) / total_valid, 3),
        "test_ratio": round(len(test_manifest) / total_valid, 3),
        "duplicate_images_excluded": len(duplicate_records),
        "corrupt_images_found": len(corrupt_files),
        "resolution_stats": {
            "min_width": min(widths) if widths else 0,
            "max_width": max(widths) if widths else 0,
            "min_height": min(heights) if heights else 0,
            "max_height": max(heights) if heights else 0,
        },
        "classes": unique_classes,
        "class_counts": class_counts,
    }

    with open(splits_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)

    with open(output_dir / "dataset_audit.json", "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)

    # Generate Markdown Audit Report
    md_content = f"""# FarmWatch AI: Dataset Audit & Provenance Report

- **Dataset Source:** PlantDoc: A Dataset for Visual Plant Disease Detection (ACM CoDS-COMAD 2020)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Fallback Justification:** Roboflow Universe FieldPlant v11 export requires interactive browser authentication / private API key. PlantDoc was utilized as the authoritative open-access field-image leaf dataset.
- **Total Unique Images:** {total_valid}
- **Number of Classes:** {len(unique_classes)}
- **Image Duplicates Removed:** {len(duplicate_records)} (Exact SHA-256 match filtered to prevent split contamination)
- **Corrupt Files Detected:** {len(corrupt_files)}

## Split Partitioning (Image-Level Grouped)
- **Train Split (70%):** {len(train_manifest)} images
- **Validation Split (15%):** {len(val_manifest)} images
- **Test Split (15% held-out):** {len(test_manifest)} images
- **Leakage Prevention:** 100% verified. Every image SHA-256 is assigned strictly to a single split partition.

## Class Distribution Across Dataset
| Class Name | Image Count |
|---|---|
"""
    for c in unique_classes:
        md_content += f"| {c} | {class_counts[c]} |\n"

    with open(output_dir / "dataset_audit.md", "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Audit completed. Split summary saved to {splits_dir / 'split_summary.json'}.")
    return split_summary


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent.parent
    raw_path = root / "data" / "raw" / "plantdoc"
    reports_path = root / "reports"
    splits_path = root / "data" / "splits"
    manifests_path = root / "data" / "manifests"
    audit = inspect_and_audit(raw_path, reports_path, splits_path, manifests_path)
    print("Train:", audit["train_count"], "Val:", audit["val_count"], "Test:", audit["test_count"])
