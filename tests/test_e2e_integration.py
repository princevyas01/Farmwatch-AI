"""End-to-End Integration Test for FarmWatch AI.

Executes genuine end-to-end verification using actual test set images,
the trained PyTorch checkpoint, Grad-CAM generation, and FastAPI endpoints.
"""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_full_e2e_system_flow():
    """Verify entire workflow from service discovery to inference on a real field image."""
    # 1. Check health
    health_res = client.get("/api/health")
    assert health_res.status_code == 200
    health_data = health_res.json()
    assert health_data["status"] == "healthy"
    assert health_data["model_loaded"] is True

    # 2. Check model info
    info_res = client.get("/api/model-info")
    assert info_res.status_code == 200
    info_data = info_res.json()
    assert info_data["model_architecture"] in ("MobileNetV3-Small", "EnhancedCropCNN")
    assert info_data["num_classes"] == 11
    assert info_data["is_pretrained"] in (True, False)

    # 3. Check metrics dynamically against active model report
    metrics_res = client.get("/api/metrics")
    assert metrics_res.status_code == 200
    metrics_data = metrics_res.json()
    assert metrics_data["test_accuracy"] is not None
    assert metrics_data["test_accuracy"] > 0.20
    assert metrics_data["macro_f1"] is not None
    assert metrics_data["macro_f1"] > 0.15
    assert metrics_data["test_samples_count"] == 214

    # 4. Load a real held-out test image from the manifest
    manifest_path = PROJECT_ROOT / "data" / "manifests" / "focused_test_manifest.json"
    assert manifest_path.exists(), "Focused test manifest missing"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert len(manifest) == 214, f"Expected 214 test samples, got {len(manifest)}"
    test_sample = manifest[0]
    sample_path = Path(test_sample["file_path"])
    assert sample_path.exists(), f"Sample image file not found: {sample_path}"

    with open(sample_path, "rb") as img_f:
        img_bytes = img_f.read()

    # 5. Execute full prediction on real image
    pred_res = client.post(
        "/api/predict",
        files={"file": (sample_path.name, img_bytes, "image/jpeg")},
        data={"generate_cam": "true"},
    )
    assert pred_res.status_code == 200
    pred_data = pred_res.json()

    assert pred_data["predicted_class"] in info_data["class_names"]
    assert 0.0 <= pred_data["model_score"] <= 1.0
    assert pred_data["confidence_level"] in ("high", "moderate", "low")
    assert isinstance(pred_data["is_uncertain"], bool)
    assert len(pred_data["top_predictions"]) >= 1
    assert len(pred_data["recommendations"]) > 0

    # Verify Grad-CAM base64 data URI was returned
    assert pred_data["gradcam_base64"] is not None
    assert pred_data["gradcam_base64"].startswith("data:image/png;base64,")

    # 6. Test with leaf ROI bounding box
    roi_bbox = json.dumps([0.15, 0.15, 0.85, 0.85])
    roi_res = client.post(
        "/api/predict",
        files={"file": (sample_path.name, img_bytes, "image/jpeg")},
        data={"bbox": roi_bbox, "generate_cam": "true"},
    )
    assert roi_res.status_code == 200
    roi_data = roi_res.json()
    assert roi_data["crop_applied"] is True
    assert roi_data["gradcam_base64"] is not None
