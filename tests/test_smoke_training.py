"""Smoke test verifying model training pipeline, forward/backward pass, and Grad-CAM."""

import tempfile
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

from src.models.custom_cnn import CustomCropCNN
from src.explainability.gradcam import GradCAM
from src.utils.device import get_torch_device, get_device_info


def test_device_selection():
    device = get_torch_device()
    assert device.type in ("cpu", "xpu")
    info = get_device_info()
    assert "active_device" in info
    assert "pytorch_version" in info


def test_custom_cnn_forward_backward():
    device = get_torch_device()
    num_classes = 28
    batch_size = 2
    model = CustomCropCNN(num_classes=num_classes).to(device)

    # 1. Forward pass
    dummy_input = torch.randn(batch_size, 3, 128, 128, device=device)
    logits = model(dummy_input)
    assert logits.shape == (batch_size, num_classes)

    # 2. Backward pass
    criterion = nn.CrossEntropyLoss()
    target = torch.tensor([0, 1], device=device)
    loss = criterion(logits, target)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()

    # Check gradients
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None

    # 3. Optimizer step
    optimizer.step()
    model.eval()
    expected_logits = model(dummy_input)

    # 4. Save and reload checkpoint
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        torch.save({"model_state_dict": model.state_dict(), "num_classes": num_classes}, tmp_path)
        reloaded_model = CustomCropCNN(num_classes=num_classes).to(device)
        checkpoint = torch.load(tmp_path, map_location=device, weights_only=False)
        reloaded_model.load_state_dict(checkpoint["model_state_dict"])
        reloaded_model.eval()
        new_logits = reloaded_model(dummy_input)
        assert torch.allclose(expected_logits, new_logits, atol=1e-5)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_gradcam_overlay_generation():
    device = get_torch_device()
    model = CustomCropCNN(num_classes=28).to(device)
    model.eval()

    gradcam = GradCAM(model)
    test_tensor = torch.randn(1, 3, 128, 128, device=device)

    heatmap, pred_c, raw_score = gradcam.generate_heatmap(test_tensor)
    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (128, 128)
    assert 0.0 <= heatmap.min() <= heatmap.max() <= 1.0

    # Test overlay with dummy PIL image
    dummy_pil = Image.fromarray(np.uint8(np.random.randint(0, 255, (128, 128, 3))))
    overlay = gradcam.create_overlay_image(dummy_pil, heatmap)
    assert overlay.size == (128, 128)

    data_uri, _, _ = gradcam.generate_overlay_base64(dummy_pil, test_tensor)
    assert data_uri.startswith("data:image/png;base64,")
    gradcam.remove_hooks()
