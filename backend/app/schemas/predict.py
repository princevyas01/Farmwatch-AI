"""Pydantic schemas for the FarmWatch AI inference API."""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "healthy"})
    device: str = Field(..., json_schema_extra={"example": "cpu"})
    device_name: str = Field(..., json_schema_extra={"example": "CPU (Intel Core Ultra 5 125H)"})
    pytorch_version: str = Field(..., json_schema_extra={"example": "2.14.0"})
    model_loaded: bool = Field(..., json_schema_extra={"example": True})


class ModelInfoResponse(BaseModel):
    model_architecture: str = Field(..., json_schema_extra={"example": "EnhancedCropCNN"})
    num_classes: int = Field(..., json_schema_extra={"example": 11})
    class_names: List[str]
    input_resolution: List[int] = Field(..., json_schema_extra={"example": [128, 128]})
    active_device: str = Field(..., json_schema_extra={"example": "cpu"})
    dataset_source: str = Field(..., json_schema_extra={"example": "PlantDoc Benchmark (IEEE Access 2023)"})
    license: str = Field(..., json_schema_extra={"example": "CC BY 4.0"})
    citation: str = Field(..., json_schema_extra={"example": "PlantDoc: A Dataset for Visual Plant Disease Detection"})
    parameters: int = Field(..., json_schema_extra={"example": 1207787})
    is_pretrained: bool = Field(..., json_schema_extra={"example": False})


class PredictionResponse(BaseModel):
    predicted_class: str
    model_score: float
    confidence_level: str  # "high", "moderate", "low"
    is_uncertain: bool
    score_interpretation: str
    top_predictions: List[Dict[str, Any]]
    recommendations: List[str]
    gradcam_base64: Optional[str] = None
    explanation_available: bool = False
    crop_applied: bool = False
    disclaimer: str = (
        "Model prediction is an uncalibrated statistical estimate for visible leaf condition pattern matching. "
        "It does not constitute agronomic diagnosis or causal proof. For high-impact field management, "
        "always consult a certified agronomist or plant pathologist."
    )


class MetricsResponse(BaseModel):
    model_name: Optional[str] = None
    checkpoint_name: Optional[str] = None
    test_accuracy: Optional[float] = None
    macro_precision: Optional[float] = None
    macro_recall: Optional[float] = None
    macro_f1: Optional[float] = None
    test_samples_count: Optional[int] = None
    num_classes: int
    class_names: List[str]
    per_class_f1: Dict[str, float]
    average_latency_ms: Optional[float] = None
    evaluation_timestamp: Optional[str] = None
    confusion_matrix_base64: Optional[str] = None
