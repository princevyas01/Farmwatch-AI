"""Package complete FarmWatch Model Audit Bundle into a verified ZIP archive."""

import zipfile
import hashlib
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_ZIP = PROJECT_ROOT / "reports" / "model_audit" / "FarmWatch_Model_Audit_Bundle.zip"
ROOT_ZIP = PROJECT_ROOT / "FarmWatch_Model_Audit_Bundle.zip"

files_to_pack = [
    # Root documents
    (PROJECT_ROOT / "reports" / "model_audit" / "README.md", "README.md"),
    (PROJECT_ROOT / "reports" / "model_audit" / "FarmWatch_Model_Audit_Report.md", "FarmWatch_Model_Audit_Report.md"),
    (PROJECT_ROOT / "reports" / "model_audit" / "checkpoint_analysis.json", "checkpoint_analysis.json"),
    (PROJECT_ROOT / "reports" / "model_audit" / "audit_execution.log", "audit_execution.log"),
    (PROJECT_ROOT / "requirements.txt", "requirements.txt"),
    (PROJECT_ROOT / "docs" / "MODEL_TRAINING.md", "docs/MODEL_TRAINING.md"),
    (PROJECT_ROOT / "docs" / "INTEGRATION_STATUS.md", "docs/INTEGRATION_STATUS.md"),

    # Model artifacts
    (PROJECT_ROOT / "models" / "mobilenetv3_small_transfer.pth", "models/mobilenetv3_small_transfer.pth"),
    (PROJECT_ROOT / "models" / "enhanced_cnn.pth", "models/enhanced_cnn.pth"),
    (PROJECT_ROOT / "models" / "cnn_from_scratch.pth", "models/cnn_from_scratch.pth"),
    (PROJECT_ROOT / "models" / "mobilenetv3_class_to_idx.json", "models/mobilenetv3_class_to_idx.json"),
    (PROJECT_ROOT / "models" / "focused_class_to_idx.json", "models/focused_class_to_idx.json"),
    (PROJECT_ROOT / "models" / "class_to_idx.json", "models/class_to_idx.json"),
    (PROJECT_ROOT / "models" / "mobilenetv3_preprocess_config.json", "models/mobilenetv3_preprocess_config.json"),
    (PROJECT_ROOT / "models" / "focused_preprocess_config.json", "models/focused_preprocess_config.json"),
    (PROJECT_ROOT / "models" / "preprocess_config.json", "models/preprocess_config.json"),

    # Dataset metadata
    (PROJECT_ROOT / "reports" / "model_audit" / "class_distribution.csv", "data_metadata/class_distribution.csv"),
    (PROJECT_ROOT / "data" / "manifests" / "focused_train_manifest.json", "data_metadata/focused_train_manifest.json"),
    (PROJECT_ROOT / "data" / "manifests" / "focused_val_manifest.json", "data_metadata/focused_val_manifest.json"),
    (PROJECT_ROOT / "data" / "manifests" / "focused_test_manifest.json", "data_metadata/focused_test_manifest.json"),
    (PROJECT_ROOT / "reports" / "dataset_audit.json", "data_metadata/dataset_audit.json"),
    (PROJECT_ROOT / "reports" / "focused_dataset_audit.json", "data_metadata/focused_dataset_audit.json"),

    # Training evidence
    (PROJECT_ROOT / "notebooks" / "FarmWatch_MobileNetV3_Transfer_Learning.ipynb", "training_evidence/FarmWatch_MobileNetV3_Transfer_Learning.ipynb"),
    (PROJECT_ROOT / "notebooks" / "FarmWatch_CNN_Training.ipynb", "training_evidence/FarmWatch_CNN_Training.ipynb"),
    (PROJECT_ROOT / "src" / "training" / "train_mobilenetv3.py", "training_evidence/train_mobilenetv3.py"),
    (PROJECT_ROOT / "src" / "training" / "train_enhanced_cnn.py", "training_evidence/train_enhanced_cnn.py"),
    (PROJECT_ROOT / "src" / "training" / "train_cnn.py", "training_evidence/train_cnn.py"),
    (PROJECT_ROOT / "src" / "training" / "evaluate_mobilenetv3.py", "training_evidence/evaluate_mobilenetv3.py"),
    (PROJECT_ROOT / "src" / "training" / "evaluate_enhanced_model.py", "training_evidence/evaluate_enhanced_model.py"),
    (PROJECT_ROOT / "src" / "training" / "evaluate_model.py", "training_evidence/evaluate_model.py"),
    (PROJECT_ROOT / "reports" / "mobilenetv3_training_history.csv", "training_evidence/mobilenetv3_training_history.csv"),
    (PROJECT_ROOT / "reports" / "mobilenetv3_training_curves.png", "training_evidence/mobilenetv3_training_curves.png"),
    (PROJECT_ROOT / "reports" / "mobilenetv3_validation_metrics.json", "training_evidence/mobilenetv3_validation_metrics.json"),
    (PROJECT_ROOT / "reports" / "model_audit" / "training_history.csv", "training_evidence/enhanced_cnn_training_history.csv"),
    (PROJECT_ROOT / "reports" / "focused_training_curves.png", "training_evidence/focused_training_curves.png"),
    (PROJECT_ROOT / "reports" / "baseline_28class_training_curves.png", "training_evidence/baseline_28class_training_curves.png"),

    # Evaluation evidence
    (PROJECT_ROOT / "reports" / "mobilenetv3_test_metrics.json", "evaluation_evidence/mobilenetv3_test_metrics.json"),
    (PROJECT_ROOT / "reports" / "mobilenetv3_test_predictions.csv", "evaluation_evidence/mobilenetv3_test_predictions.csv"),
    (PROJECT_ROOT / "reports" / "mobilenetv3_test_class_probabilities.csv", "evaluation_evidence/mobilenetv3_test_class_probabilities.csv"),
    (PROJECT_ROOT / "reports" / "mobilenetv3_incorrect_predictions.csv", "evaluation_evidence/mobilenetv3_incorrect_predictions.csv"),
    (PROJECT_ROOT / "reports" / "validation_uncertainty_metrics.json", "evaluation_evidence/validation_uncertainty_metrics.json"),
    (PROJECT_ROOT / "reports" / "mobilenetv3_comparison.json", "evaluation_evidence/mobilenetv3_comparison.json"),
    (PROJECT_ROOT / "reports" / "mobilenetv3_confusion_matrix.png", "evaluation_evidence/mobilenetv3_confusion_matrix.png"),
    (PROJECT_ROOT / "reports" / "mobilenetv3_sample_predictions.png", "evaluation_evidence/mobilenetv3_sample_predictions.png"),
    (PROJECT_ROOT / "reports" / "model_audit" / "recalculated_test_metrics.json", "evaluation_evidence/recalculated_test_metrics.json"),
    (PROJECT_ROOT / "reports" / "focused_test_metrics.json", "evaluation_evidence/focused_test_metrics.json"),
    (PROJECT_ROOT / "reports" / "model_audit" / "classification_report.txt", "evaluation_evidence/classification_report.txt"),
    (PROJECT_ROOT / "reports" / "model_audit" / "test_predictions.csv", "evaluation_evidence/test_predictions.csv"),
    (PROJECT_ROOT / "reports" / "model_audit" / "test_class_probabilities.csv", "evaluation_evidence/test_class_probabilities.csv"),
    (PROJECT_ROOT / "reports" / "model_audit" / "incorrect_predictions.csv", "evaluation_evidence/incorrect_predictions.csv"),
    (PROJECT_ROOT / "reports" / "model_audit" / "confusion_matrix_raw.csv", "evaluation_evidence/confusion_matrix_raw.csv"),
    (PROJECT_ROOT / "reports" / "model_audit" / "confusion_matrix_raw.png", "evaluation_evidence/confusion_matrix_raw.png"),
    (PROJECT_ROOT / "reports" / "model_audit" / "confusion_matrix_normalized.csv", "evaluation_evidence/confusion_matrix_normalized.csv"),
    (PROJECT_ROOT / "reports" / "model_audit" / "confusion_matrix_normalized.png", "evaluation_evidence/confusion_matrix_normalized.png"),
    (PROJECT_ROOT / "reports" / "model_audit" / "consistency_audit.json", "evaluation_evidence/consistency_audit.json"),

    # Visual error analysis
    (PROJECT_ROOT / "reports" / "model_audit" / "error_analysis" / "visual_error_catalog.csv", "visual_error_analysis/visual_error_catalog.csv"),

    # Source code
    (PROJECT_ROOT / "src" / "models" / "custom_cnn.py", "source_code/custom_cnn.py"),
    (PROJECT_ROOT / "src" / "explainability" / "gradcam.py", "source_code/gradcam.py"),
    (PROJECT_ROOT / "src" / "inference" / "model_service.py", "source_code/model_service.py"),
    (PROJECT_ROOT / "backend" / "app" / "services" / "recommendations.py", "source_code/recommendations.py"),
    (PROJECT_ROOT / "scripts" / "run_model_audit.py", "source_code/run_model_audit.py"),
    (PROJECT_ROOT / "scripts" / "export_mobilenetv3_predictions.py", "source_code/export_mobilenetv3_predictions.py"),
    (PROJECT_ROOT / "scripts" / "evaluate_validation_uncertainty.py", "source_code/evaluate_validation_uncertainty.py"),
    (PROJECT_ROOT / "src" / "utils" / "dataset_paths.py", "source_code/dataset_paths.py"),
]

# Add all PNGs in error_analysis
error_dir = PROJECT_ROOT / "reports" / "model_audit" / "error_analysis"
if error_dir.exists():
    for png_file in error_dir.glob("*.png"):
        files_to_pack.append((png_file, f"visual_error_analysis/{png_file.name}"))

print(f"Packaging {len(files_to_pack)} items into {BUNDLE_ZIP}...")

# Remove old zip if present
if BUNDLE_ZIP.exists():
    BUNDLE_ZIP.unlink()

with zipfile.ZipFile(BUNDLE_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for src_path, arc_name in files_to_pack:
        if not src_path.exists():
            print(f"WARNING: File not found: {src_path}")
            continue
        zf.write(src_path, arc_name)

# Copy to root folder as requested
shutil.copy2(BUNDLE_ZIP, ROOT_ZIP)

# Verification & Hashing
bundle_size = BUNDLE_ZIP.stat().st_size
bundle_mb = bundle_size / (1024 * 1024)

sha256 = hashlib.sha256()
with open(BUNDLE_ZIP, "rb") as f:
    while chunk := f.read(1024 * 1024):
        sha256.update(chunk)
bundle_hash = sha256.hexdigest()

with zipfile.ZipFile(BUNDLE_ZIP, "r") as zf:
    test_res = zf.testzip()
    namelist = zf.namelist()

if test_res is None:
    print(f"SUCCESS: ZIP package verified cleanly with {len(namelist)} items.")
else:
    print(f"ERROR: Corrupted file in zip: {test_res}")

print(f"Bundle File (reports): {BUNDLE_ZIP}")
print(f"Bundle File (root):    {ROOT_ZIP}")
print(f"Bundle Size:           {bundle_size:,} bytes ({bundle_mb:.2f} MB)")
print(f"Bundle SHA256:         {bundle_hash}")
