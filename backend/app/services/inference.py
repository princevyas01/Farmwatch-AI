"""Inference service adapter for FastAPI backend."""
from src.inference.model_service import ModelService, get_model_service

__all__ = ["ModelService", "get_model_service"]
