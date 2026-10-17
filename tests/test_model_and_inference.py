"""Test suite verifying trained model artifacts, metrics, and inference contract."""

import json
from pathlib import Path
import pytest
import torch
from PIL import Image

from src.models.custom_cnn import CustomCropCNN
from src.inference_example import predict_leaf_image, load_model_and_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_checkpoint_integrity():
    checkpoint_path = PROJECT_ROOT / "models" / "cnn_from_scratch.pth"
    assert checkpoint_path.exists(), "Trained checkpoint does not exist"

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in checkpoint
    assert "class_to_idx" in checkpoint
    assert "num_classes" in checkpoint
    assert checkpoint["num_classes"] == 28

    model = CustomCropCNN(num_classes=28)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Verify forward pass with dummy tensor
    dummy_input = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(dummy_input)
    assert out.shape == (1, 28)


def test_configurations_and_metrics():
    class_to_idx_path = PROJECT_ROOT / "models" / "class_to_idx.json"
    preprocess_config_path = PROJECT_ROOT / "models" / "preprocess_config.json"
    metrics_path = PROJECT_ROOT / "reports" / "test_metrics.json"

    assert class_to_idx_path.exists()
    assert preprocess_config_path.exists()
    assert metrics_path.exists()

    with open(class_to_idx_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    assert len(class_to_idx) == 28

    with open(preprocess_config_path, "r", encoding="utf-8") as f:
        preprocess_config = json.load(f)
    assert preprocess_config["input_resolution"] == [128, 128]
    assert len(preprocess_config["classes"]) == 28

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert metrics["test_samples_count"] == 410
    assert metrics["test_accuracy"] > 0.05
    assert "macro_f1" in metrics


def test_live_inference_on_sample():
    test_manifest_path = PROJECT_ROOT / "data" / "manifests" / "test_manifest.json"
    assert test_manifest_path.exists()

    with open(test_manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert len(manifest) == 410

    sample_img_path = Path(manifest[0]["file_path"])
    assert sample_img_path.exists()

    result = predict_leaf_image(sample_img_path, generate_cam=True)
    assert "predicted_class" in result
    assert "model_score" in result
    assert "is_uncertain" in result
    assert "top_3_candidates" in result
    assert len(result["top_3_candidates"]) == 3
    assert "gradcam_overlay_saved" in result
    assert Path(result["gradcam_overlay_saved"]).exists()


def test_enhanced_model_integrity_and_inference():
    enhanced_ckpt_path = PROJECT_ROOT / "models" / "enhanced_cnn.pth"
    assert enhanced_ckpt_path.exists()

    ckpt = torch.load(enhanced_ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["num_classes"] == 11
    assert ckpt["architecture"] == "EnhancedCropCNN"

    manifest_path = PROJECT_ROOT / "data" / "manifests" / "focused_test_manifest.json"
    assert manifest_path.exists()
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert len(manifest) == 214

    sample_img = Path(manifest[0]["file_path"])
    result = predict_leaf_image(sample_img, checkpoint_path=enhanced_ckpt_path, generate_cam=True)
    assert "predicted_class" in result
    assert "model_score" in result
    assert result["predicted_class"] in ckpt["class_to_idx"]
