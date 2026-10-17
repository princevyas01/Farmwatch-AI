"""Dataset path resolution utilities for FarmWatch AI.

Provides cross-platform, reproducible dataset path resolution:
- Supports repository-relative paths (e.g. data/raw/plantdoc/train/...)
- Supports configurable dataset root via FARMWATCH_DATASET_ROOT environment variable
- Supports Windows extended path prefixes (\\\\?\\) for long paths
"""

import os
from pathlib import Path
from typing import Optional, Union, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_dataset_root() -> Path:
    """Return the configured dataset root directory.
    
    Checks FARMWATCH_DATASET_ROOT environment variable first,
    defaulting to PROJECT_ROOT / 'data' / 'raw' / 'plantdoc'.
    """
    env_root = os.getenv("FARMWATCH_DATASET_ROOT")
    if env_root and env_root.strip():
        custom_root = Path(env_root.strip()).resolve()
        if custom_root.exists():
            return custom_root
    return (PROJECT_ROOT / "data" / "raw" / "plantdoc").resolve()


def resolve_image_path(item_or_path: Union[str, Path, Dict[str, Any]]) -> Path:
    """Resolve an image file path from a manifest item or raw path string.
    
    Tries in priority order:
    1. If already an existing Path or absolute path that exists.
    2. Path relative to PROJECT_ROOT.
    3. Path relative to get_dataset_root() using relative_path if present.
    4. Path relative to get_dataset_root() using filename/basename.
    """
    if isinstance(item_or_path, dict):
        fp = item_or_path.get("file_path", "")
        rel_p = item_or_path.get("relative_path", "")
    else:
        fp = str(item_or_path)
        rel_p = ""

    path_obj = Path(fp)

    # 1. Direct path exists
    if path_obj.exists():
        return path_obj.resolve()

    # 2. Relative to PROJECT_ROOT
    candidate = (PROJECT_ROOT / path_obj).resolve()
    if candidate.exists():
        return candidate

    # 3. Relative to dataset root via relative_path
    dataset_root = get_dataset_root()
    if rel_p:
        candidate = (dataset_root / rel_p).resolve()
        if candidate.exists():
            return candidate

    # 4. Relative to dataset root directly via fp
    candidate = (dataset_root / path_obj).resolve()
    if candidate.exists():
        return candidate

    # Return best effort candidate relative to PROJECT_ROOT
    return (PROJECT_ROOT / path_obj).resolve()


def get_safe_path_str(path: Union[str, Path]) -> str:
    """Return safe path string, handling Windows extended length prefixes when necessary."""
    path_str = str(path)
    if os.name == "nt" and len(path_str) > 240 and not path_str.startswith("\\\\?\\"):
        return f"\\\\?\\{path_str}"
    return path_str
