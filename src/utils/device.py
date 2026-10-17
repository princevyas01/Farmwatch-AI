"""Hardware detection and device selection abstraction for PyTorch.

Supports Intel Arc GPU via torch.xpu when properly initialized and drivers are available,
with a guaranteed and robust CPU fallback. Never assumes CUDA on Intel hardware.
"""

import logging
from typing import Tuple, Dict, Any
import torch

logger = logging.getLogger("farmwatch.device")


def get_torch_device() -> torch.device:
    """Detect and return the best available compute device.

    Checks for Intel Arc XPU support first. If unavailable or unstable,
    safely falls back to CPU.
    """
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        try:
            # Perform a micro tensor allocation to test driver and backend stability
            test_tensor = torch.zeros(1, device="xpu")
            del test_tensor
            logger.info("Intel XPU backend detected and operational. Using device: xpu")
            return torch.device("xpu")
        except Exception as e:
            logger.warning(
                f"Intel XPU reported available but initialization failed: {e}. Falling back to CPU."
            )
            return torch.device("cpu")

    logger.info("Using compute device: cpu")
    return torch.device("cpu")


def get_device_info() -> Dict[str, Any]:
    """Return structured system and compute device diagnostic metadata."""
    device = get_torch_device()
    info: Dict[str, Any] = {
        "active_device": str(device),
        "pytorch_version": torch.__version__,
        "xpu_backend_present": hasattr(torch, "xpu"),
        "xpu_is_available": hasattr(torch, "xpu") and torch.xpu.is_available(),
        "cuda_is_available": torch.cuda.is_available(),
    }

    if device.type == "xpu" and hasattr(torch.xpu, "get_device_name"):
        try:
            info["device_name"] = torch.xpu.get_device_name(0)
        except Exception:
            info["device_name"] = "Intel Arc Graphics (XPU)"
    elif device.type == "cpu":
        info["device_name"] = "CPU (Intel Core Ultra 5 125H)"
    else:
        info["device_name"] = str(device)

    return info
