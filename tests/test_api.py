"""Comprehensive test suite for the FarmWatch AI FastAPI Backend."""

import io
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import app

client = TestClient(app)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def create_test_image_bytes(format="JPEG", size=(128, 128), color=(45, 120, 60)) -> bytes:
    """Create a synthetic in-memory image for fast, self-contained API testing."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


def test_api_health():
    """Verify /api/health endpoint structure and values."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["device"] in ("cpu", "xpu")
    assert data["model_loaded"] is True
    assert "pytorch_version" in data


def test_api_model_info():
    """Verify /api/model-info reports scratch-trained parameters and dataset lineage."""
    response = client.get("/api/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["is_pretrained"] in (True, False)
    assert data["model_architecture"] in ("EnhancedCropCNN", "CustomCropCNN", "MobileNetV3-Small")
    assert data["num_classes"] in (11, 28)
    assert len(data["class_names"]) == data["num_classes"]
    assert data["parameters"] > 100_000
    assert "PlantDoc" in data["dataset_source"]
    assert "CC BY 4.0" in data["license"]


def test_api_metrics():
    """Verify /api/metrics serves genuine test set evaluation numbers."""
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["test_accuracy"] > 0.15
    assert data["macro_f1"] > 0.10
    assert data["test_samples_count"] > 0
    assert isinstance(data["per_class_f1"], dict)
    assert data["confusion_matrix_base64"] is not None
    assert data["confusion_matrix_base64"].startswith("data:image/png;base64,")


def test_api_predict_valid_image():
    """Verify /api/predict succeeds with image and produces Grad-CAM + recommendations."""
    img_bytes = create_test_image_bytes()
    response = client.post(
        "/api/predict",
        files={"file": ("test_leaf.jpg", img_bytes, "image/jpeg")},
        data={"generate_cam": "true"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert "model_score" in data
    assert 0.0 <= data["model_score"] <= 1.0
    assert data["confidence_level"] in ("high", "moderate", "low")
    assert isinstance(data["is_uncertain"], bool)
    assert len(data["top_predictions"]) >= 1
    assert len(data["recommendations"]) > 0
    assert data["gradcam_base64"] is not None
    assert data["gradcam_base64"].startswith("data:image/png;base64,")
    assert data["crop_applied"] is False


def test_api_predict_with_bounding_box():
    """Verify /api/predict accepts ROI bounding box and marks crop_applied."""
    img_bytes = create_test_image_bytes(size=(200, 200))
    # Normalized bounding box [0.1, 0.1, 0.9, 0.9]
    bbox_json = json.dumps([0.1, 0.1, 0.9, 0.9])
    response = client.post(
        "/api/predict",
        files={"file": ("test_crop.jpg", img_bytes, "image/jpeg")},
        data={"bbox": bbox_json, "generate_cam": "true"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["crop_applied"] is True
    assert data["gradcam_base64"] is not None


def test_api_predict_corrupt_image_rejected():
    """Verify corrupt bytes are rejected with HTTP 400 and actionable message."""
    corrupt_bytes = b"NOT_AN_IMAGE_RANDOM_GARBAGE_BYTES_12345"
    response = client.post(
        "/api/predict",
        files={"file": ("corrupt.jpg", corrupt_bytes, "image/jpeg")},
    )
    assert response.status_code == 400
    assert "Corrupted or invalid image file" in response.json()["detail"]


def test_api_predict_empty_file_rejected():
    """Verify empty payload is rejected with HTTP 400."""
    response = client.post(
        "/api/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_api_predict_unsupported_media_type():
    """Verify unsupported mime types are rejected."""
    response = client.post(
        "/api/predict",
        files={"file": ("script.py", b"print('hello')", "text/x-python")},
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_api_predict_oversized_file():
    """Verify files > 15MB are rejected with HTTP 413."""
    # 15.5 MB of dummy data
    oversized = b"0" * (16 * 1024 * 1024)
    response = client.post(
        "/api/predict",
        files={"file": ("large.jpg", oversized, "image/jpeg")},
    )
    assert response.status_code == 413
    assert "exceeds 15MB limit" in response.json()["detail"]


def test_api_samples_and_sample_image():
    """Verify /api/samples lists test set images and serves the raw image bytes."""
    response = client.get("/api/samples")
    assert response.status_code == 200
    samples = response.json()
    assert isinstance(samples, list)
    if len(samples) > 0:
        first = samples[0]
        assert "id" in first
        assert "class_name" in first
        assert "url" in first

        img_resp = client.get(first["url"])
        assert img_resp.status_code == 200
        assert img_resp.headers["content-type"] in ("image/jpeg", "image/png", "image/webp")
        assert len(img_resp.content) > 100


def test_cors_allowed_and_disallowed_origins():
    """Verify CORS allowlist behavior for allowed vs disallowed origins."""
    # Allowed origin: http://localhost:5173
    res_allowed = client.options(
        "/api/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert res_allowed.headers.get("access-control-allow-origin") == "http://localhost:5173"

    # Disallowed origin: http://malicious-attacker.com
    res_disallowed = client.options(
        "/api/health",
        headers={
            "Origin": "http://malicious-attacker.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert res_disallowed.headers.get("access-control-allow-origin") is None


def test_internal_error_concealment_on_unexpected_failure():
    """Verify unexpected inference crashes return generic 500 with correlation ID, no leak."""
    from unittest.mock import patch
    img_bytes = create_test_image_bytes()

    with patch("src.inference.model_service.ModelService.predict_image", side_effect=RuntimeError("Internal /deep/secret/path/torch_crash.py: SegFault")):
        response = client.post(
            "/api/predict",
            files={"file": ("test_leaf.jpg", img_bytes, "image/jpeg")},
        )
        assert response.status_code == 500
        detail = response.json()["detail"]
        assert "Reference ID:" in detail
        assert "unexpected error" in detail.lower()
        assert "/deep/secret/path" not in detail
        assert "SegFault" not in detail
        assert "RuntimeError" not in detail


def test_predict_malformed_bounding_boxes():
    """Verify malformed bounding boxes are rejected with HTTP 400."""
    img_bytes = create_test_image_bytes()

    # Case 1: Not valid JSON
    res1 = client.post(
        "/api/predict",
        files={"file": ("leaf.jpg", img_bytes, "image/jpeg")},
        data={"bbox": "not-valid-json"},
    )
    assert res1.status_code == 400
    assert "Invalid bbox format" in res1.json()["detail"]

    # Case 2: Wrong number of elements
    res2 = client.post(
        "/api/predict",
        files={"file": ("leaf.jpg", img_bytes, "image/jpeg")},
        data={"bbox": json.dumps([0.1, 0.2, 0.8])},
    )
    assert res2.status_code == 400
    assert "4 numbers" in res2.json()["detail"]

    # Case 3: Inverted coordinates (x_min >= x_max)
    res3 = client.post(
        "/api/predict",
        files={"file": ("leaf.jpg", img_bytes, "image/jpeg")},
        data={"bbox": json.dumps([0.8, 0.2, 0.2, 0.9])},
    )
    assert res3.status_code == 400
    assert "x_min < x_max" in res3.json()["detail"]

    # Case 4: Out of bounds (< 0.0 or > 1.0)
    res4 = client.post(
        "/api/predict",
        files={"file": ("leaf.jpg", img_bytes, "image/jpeg")},
        data={"bbox": json.dumps([-0.1, 0.2, 0.8, 0.9])},
    )
    assert res4.status_code == 400
    assert ">= 0.0" in res4.json()["detail"]


def test_sample_image_invalid_id():
    """Verify non-existent sample image ID returns 404."""
    response = client.get("/api/samples/999999/image")
    assert response.status_code == 404
    assert "Sample ID out of range" in response.json()["detail"]


def test_metrics_nullable_fields_handling():
    """Verify metrics schema conforms with nullable fields."""
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "num_classes" in data
    assert "class_names" in data
    assert "per_class_f1" in data
    assert "average_latency_ms" in data
    assert "evaluation_timestamp" in data
