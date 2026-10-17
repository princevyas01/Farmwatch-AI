"""Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.

Provides visual explanations of decisions made by convolutional neural networks.
Outputs heatmaps highlighting discriminative leaf regions.
Includes strict hook lifecycle management and resource cleanup.
"""

import io
import base64
import logging
from typing import Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.cm as cm

logger = logging.getLogger("farmwatch.gradcam")


class GradCAM:
    """Computes Grad-CAM heatmaps for a given PyTorch convolutional network."""

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None, device: Optional[Any] = None):
        self._hook_handles = []
        self._hooks_active = False
        self.model = model
        self.device = device
        self.model.eval()

        if target_layer is None:
            if hasattr(model, "get_target_layer_for_cam"):
                self.target_layer = model.get_target_layer_for_cam()
            else:
                conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
                if not conv_layers:
                    raise ValueError("No Conv2d layers found in model for Grad-CAM.")
                self.target_layer = conv_layers[-1]
        else:
            self.target_layer = target_layer

        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._hook_handles = []
        self._hooks_active = False
        self._register_hooks()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

    def _register_hooks(self) -> None:
        if self._hooks_active:
            return

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        def forward_hook(module, input, output):
            self.activations = output.detach()

        h_fwd = self.target_layer.register_forward_hook(forward_hook)
        h_bwd = self.target_layer.register_full_backward_hook(backward_hook)
        self._hook_handles.extend([h_fwd, h_bwd])
        self._hooks_active = True

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """Generate Grad-CAM heatmap normalized to [0, 1].

        Enforces gradient tracking specifically for backward pass without altering
        model evaluation mode.
        """
        if not self._hooks_active:
            self._register_hooks()

        try:
            with torch.enable_grad():
                # Ensure input tensor requires grad if needed for backprop
                input_tensor = input_tensor.clone().detach().requires_grad_(True)
                self.model.zero_grad()
                logits = self.model(input_tensor)

                if target_class is None:
                    target_class = int(torch.argmax(logits, dim=1).item())

                score = logits[0, target_class]
                raw_score = float(score.item())

                score.backward(retain_graph=True)

            if self.gradients is None or self.activations is None:
                raise RuntimeError("Gradients or activations were not captured. Check layer hooks.")

            # Pool gradients across spatial dimensions (channels weights)
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

            # Weighted combination of forward activation maps
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
            cam = F.relu(cam)

            # Resize to input tensor spatial dimensions
            h, w = input_tensor.shape[2], input_tensor.shape[3]
            cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)

            cam_np = cam.squeeze().cpu().numpy()

            # Normalize to [0, 1]
            cam_min, cam_max = cam_np.min(), cam_np.max()
            if cam_max - cam_min > 1e-8:
                heatmap = (cam_np - cam_min) / (cam_max - cam_min)
            else:
                heatmap = np.zeros_like(cam_np)

            return heatmap, target_class, raw_score

        finally:
            # Clear forward/backward buffers to prevent memory growth
            self.gradients = None
            self.activations = None

    def create_overlay_image(
        self,
        original_pil_image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.45,
        colormap_name: str = "jet",
    ) -> Image.Image:
        """Create RGB image overlay blending the original image with the colored heatmap."""
        orig_w, orig_h = original_pil_image.size
        heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize(
            (orig_w, orig_h), resample=Image.Resampling.BILINEAR
        )
        heatmap_norm = np.array(heatmap_resized, dtype=np.float32) / 255.0

        try:
            import matplotlib
            cmap = matplotlib.colormaps[colormap_name]
        except Exception:
            cmap = cm.get_cmap(colormap_name)

        rgba_heatmap = cmap(heatmap_norm)
        rgb_heatmap = (rgba_heatmap[:, :, :3] * 255).astype(np.uint8)
        heatmap_img = Image.fromarray(rgb_heatmap)

        orig_rgb = original_pil_image.convert("RGB")
        return Image.blend(orig_rgb, heatmap_img, alpha=alpha)

    def generate_overlay_base64(
        self,
        pil_image: Image.Image,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        alpha: float = 0.45,
    ) -> Tuple[str, int, float]:
        """Convenience method returning base64 PNG data URI for web/API transport."""
        heatmap, pred_idx, raw_score = self.generate_heatmap(input_tensor, target_class)
        overlay = self.create_overlay_image(pil_image, heatmap, alpha=alpha)

        buffered = io.BytesIO()
        overlay.save(buffered, format="PNG")
        encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
        data_uri = f"data:image/png;base64,{encoded}"
        return data_uri, pred_idx, raw_score

    def remove_hooks(self) -> None:
        """Clean up PyTorch hooks safely."""
        for handle in self._hook_handles:
            try:
                handle.remove()
            except Exception as e:
                logger.debug("Hook handle removal note: %s", e)
        self._hook_handles.clear()
        self._hooks_active = False

    def __del__(self):
        if hasattr(self, "_hook_handles"):
            self.remove_hooks()
