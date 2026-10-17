"""FastAPI API routes for FarmWatch AI."""

import json
import uuid
import math
import logging
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException, status
from fastapi.responses import Response

from backend.app.schemas.predict import (
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    MetricsResponse,
)
from backend.app.services.inference import ModelService, get_model_service
from src.utils.dataset_paths import resolve_image_path, get_dataset_root

logger = logging.getLogger("farmwatch.api")
router = APIRouter(prefix="/api", tags=["FarmWatch AI"])

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MAX_IMAGE_SIZE = 15 * 1024 * 1024  # 15 MB
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "application/octet-stream",
}


@router.get("/health", response_model=HealthResponse)
def get_health(service: ModelService = Depends(get_model_service)):
    """Health check returning compute device and model state."""
    return service.get_health()


@router.get("/model-info", response_model=ModelInfoResponse)
def get_model_info(service: ModelService = Depends(get_model_service)):
    """Architectural metadata, dataset provenance, and training lineage."""
    return service.get_model_info()


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics(service: ModelService = Depends(get_model_service)):
    """Verified test set metrics, class-level performance, and confusion matrix."""
    try:
        return service.get_metrics()
    except FileNotFoundError as e:
        logger.error("Metrics report not found: %s", e)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Metrics report not found: {str(e)}",
        )
    except ValueError as val_err:
        logger.error("Metrics report validation failure: %s", val_err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Metrics validation failure: {str(val_err)}",
        )


@router.post("/predict", response_model=PredictionResponse)
async def predict_image(
    file: UploadFile = File(..., description="Leaf or field crop photo (JPEG/PNG/WebP, max 15MB)"),
    bbox: Optional[str] = Form(None, description="Optional bounding box JSON string [x_min, y_min, x_max, y_max]"),
    generate_cam: bool = Form(True, description="Whether to generate Grad-CAM feature attention overlay"),
    service: ModelService = Depends(get_model_service),
):
    """Run leaf condition classification and Grad-CAM explainability on an uploaded image."""
    # 1. Validate MIME type
    if file.content_type and file.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: '{file.content_type}'. Please upload JPEG, PNG, or WebP.",
        )

    # 2. Read file content with size limit
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )
    if len(content) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded file exceeds 15MB limit ({len(content) / (1024 * 1024):.2f} MB).",
        )

    # 3. Parse and validate optional bounding box
    parsed_bbox: Optional[List[float]] = None
    if bbox:
        try:
            bbox_data = json.loads(bbox)
            if not isinstance(bbox_data, list) or len(bbox_data) != 4:
                raise ValueError("Bounding box must be a list of 4 numbers [x_min, y_min, x_max, y_max].")

            parsed = []
            for v in bbox_data:
                num = float(v)
                if not math.isfinite(num):
                    raise ValueError(f"Non-finite bounding box coordinate: {v}")
                parsed.append(num)

            if parsed[0] >= parsed[2] or parsed[1] >= parsed[3]:
                raise ValueError("Bounding box coordinates must satisfy x_min < x_max and y_min < y_max.")

            parsed_bbox = parsed
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid bbox format: {str(e)}",
            )

    # 4. Perform prediction with safe error boundaries
    try:
        response = service.predict_image(
            image_bytes=content,
            bbox=parsed_bbox,
            generate_cam=generate_cam,
        )
        return response
    except ValueError as val_err:
        # Client validation error (e.g. unreadable image or dimension violation)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as exc:
        request_id = str(uuid.uuid4())
        logger.exception("Internal inference failure [ref: %s]: %s", request_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during inference. Reference ID: {request_id}",
        )


@router.get("/samples")
def list_samples():
    """Return a curated list of genuine held-out test set samples for quick demo testing."""
    test_manifest_path = PROJECT_ROOT / "data" / "manifests" / "focused_test_manifest.json"
    if not test_manifest_path.exists():
        return []

    try:
        with open(test_manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        samples = []
        seen_classes = set()
        for idx, item in enumerate(manifest):
            cls = item["class_name"]
            img_path = resolve_image_path(item)
            if cls not in seen_classes and img_path.exists():
                seen_classes.add(cls)
                samples.append({
                    "id": idx,
                    "class_name": cls,
                    "filename": img_path.name,
                    "url": f"/api/samples/{idx}/image",
                })
                if len(samples) >= 8:
                    break
        return samples
    except Exception as e:
        logger.warning("Could not list samples: %s", e)
        return []


@router.get("/samples/{sample_id}/image")
def get_sample_image(sample_id: int):
    """Retrieve raw image bytes for a test sample by its manifest index."""
    test_manifest_path = PROJECT_ROOT / "data" / "manifests" / "focused_test_manifest.json"
    if not test_manifest_path.exists():
        raise HTTPException(status_code=404, detail="Test manifest not found.")

    try:
        with open(test_manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Could not read test manifest.")

    if sample_id < 0 or sample_id >= len(manifest):
        raise HTTPException(status_code=404, detail="Sample ID out of range.")

    img_path = resolve_image_path(manifest[sample_id])

    # Path traversal security check: must be inside project tree or dataset root
    dataset_root = get_dataset_root()
    is_safe = False
    try:
        if img_path.is_relative_to(PROJECT_ROOT) or img_path.is_relative_to(dataset_root):
            is_safe = True
    except Exception:
        is_safe = False

    if not is_safe:
        logger.warning("Path traversal attempt blocked: %s", img_path)
        raise HTTPException(status_code=403, detail="Access denied: invalid file location.")

    if not img_path.exists() or not img_path.is_file():
        raise HTTPException(status_code=404, detail="Sample file missing from disk.")

    # Determine accurate media type from file extension
    ext = img_path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        media_type = "image/jpeg"
    elif ext == ".png":
        media_type = "image/png"
    elif ext == ".webp":
        media_type = "image/webp"
    else:
        media_type = "application/octet-stream"

    with open(img_path, "rb") as f:
        data = f.read()

    return Response(content=data, media_type=media_type)
