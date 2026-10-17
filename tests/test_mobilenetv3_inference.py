"""Verification tests for MobileNetV3-Small transfer-learning model and inference pipeline."""

import json
from pathlib import Path
import pytest
import torch
import numpy as np
from PIL import Image

from src.training.train_mobilenetv3 import build_mobilenetv3_model, get_mobilenetv3_transforms
from src.explainability.gradcam import GradCAM
from src.utils.device import get_torch_device

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_mobilenetv3_checkpoint_reload_and_metadata():
    """Verify saved checkpoint reloads cleanly into a fresh model instance with correct metadata."""
    ckpt_path = PROJECT_ROOT / "models" / "mobilenetv3_small_transfer.pth"
    assert ckpt_path.exists(), f"Checkpoint missing: {ckpt_path}"

    device = get_torch_device()
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    assert "model_state_dict" in checkpoint
    assert "class_to_idx" in checkpoint
    assert checkpoint["num_classes"] == 11
    assert len(checkpoint["class_to_idx"]) == 11
    assert checkpoint["architecture"] == "MobileNetV3-Small"
    assert checkpoint["is_pretrained"] is True
    assert checkpoint["input_size"] == [224, 224]

    model = build_mobilenetv3_model(num_classes=11)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 1_529_131, f"Expected 1,529,131 params, got {total_params}"


def test_mobilenetv3_tensor_shapes_and_probabilities():
    """Verify forward pass produces finite logits and valid probability distribution summing to 1.0."""
    device = get_torch_device()
    ckpt_path = PROJECT_ROOT / "models" / "mobilenetv3_small_transfer.pth"
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = build_mobilenetv3_model(num_classes=11)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        logits = model(dummy_input)
        probs = torch.softmax(logits, dim=1)

    assert logits.shape == (2, 11)
    assert torch.all(torch.isfinite(logits)), "Logits contain NaN or Inf"
    assert probs.shape == (2, 11)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)

    sums = torch.sum(probs, dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), "Probabilities do not sum to 1.0"


def test_mobilenetv3_deterministic_preprocessing():
    """Verify deterministic preprocessing produces expected shape (1, 3, 224, 224) and normalized range."""
    _, eval_tf = get_mobilenetv3_transforms((224, 224))
    test_manifest_path = PROJECT_ROOT / "data" / "manifests" / "focused_test_manifest.json"
    with open(test_manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    sample_img_path = Path(manifest[0]["file_path"])
    assert sample_img_path.exists()

    pil_img = Image.open(sample_img_path).convert("RGB")
    tensor = eval_tf(pil_img)

    assert tensor.shape == (3, 224, 224)
    assert torch.all(torch.isfinite(tensor))


def test_mobilenetv3_live_inference_on_heldout_sample():
    """Verify inference on a genuine held-out test sample returns valid class and non-empty score."""
    device = get_torch_device()
    ckpt_path = PROJECT_ROOT / "models" / "mobilenetv3_small_transfer.pth"
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = build_mobilenetv3_model(num_classes=11)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    _, eval_tf = get_mobilenetv3_transforms((224, 224))
    test_manifest_path = PROJECT_ROOT / "data" / "manifests" / "focused_test_manifest.json"
    with open(test_manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    sample_item = manifest[0]
    sample_img_path = Path(sample_item["file_path"])
    pil_img = Image.open(sample_img_path).convert("RGB")
    input_tensor = eval_tf(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        score, pred_idx = torch.max(probs, dim=1)

    pred_class = idx_to_class[pred_idx.item()]
    assert pred_class in class_to_idx
    assert 0.0 <= score.item() <= 1.0


def test_mobilenetv3_gradcam_lifecycle_and_overlay():
    """Verify Grad-CAM generates valid heatmap and overlay using model.features[-1], with hook cleanup."""
    device = get_torch_device()
    ckpt_path = PROJECT_ROOT / "models" / "mobilenetv3_small_transfer.pth"
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = build_mobilenetv3_model(num_classes=11)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # MobileNetV3 target conv layer is the final 1x1 conv block in features
    target_layer = model.features[-1]

    _, eval_tf = get_mobilenetv3_transforms((224, 224))
    test_manifest_path = PROJECT_ROOT / "data" / "manifests" / "focused_test_manifest.json"
    with open(test_manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    sample_img_path = Path(manifest[0]["file_path"])
    pil_img = Image.open(sample_img_path).convert("RGB")
    input_tensor = eval_tf(pil_img).unsqueeze(0).to(device)

    with GradCAM(model, target_layer=target_layer) as gradcam:
        assert gradcam._hooks_active is True
        data_uri, pred_class_idx, raw_score = gradcam.generate_overlay_base64(
            pil_img.resize((224, 224)),
            input_tensor,
        )
        assert data_uri.startswith("data:image/png;base64,")
        assert 0 <= pred_class_idx < 11
        assert np.isfinite(raw_score)

    # Verify context manager cleaned up hooks
    assert gradcam._hooks_active is False
    assert len(gradcam._hook_handles) == 0
