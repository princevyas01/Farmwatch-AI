"""FarmWatch AI - FastAPI Application Entrypoint.

GenHack V1.0 PS 12: Predicting Crop Stress From Field Images.
Provides high-performance inference, Grad-CAM visual attention mapping,
and verified test evaluation metrics for the FarmWatch AI frontend.
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.endpoints import router as api_router
from backend.app.services.inference import get_model_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("farmwatch.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm and load PyTorch model singleton on startup."""
    logger.info("Pre-warming ModelService and loading PyTorch weights...")
    service = get_model_service()
    health = service.get_health()
    logger.info("Loaded architecture '%s' on device: %s (%s)", service.architecture, health.device, health.device_name)
    yield
    logger.info("Shutting down application.")


app = FastAPI(
    title="FarmWatch AI Backend",
    version="1.0.0",
    description="Automated visible crop and leaf condition classification using scratch-trained PyTorch CNN with Grad-CAM explainability.",
    lifespan=lifespan,
)

# Configure CORS with explicit, environment-configurable allowlist
raw_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000",
)
allowed_origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip() and origin.strip() != "*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,  # Unauthenticated API does not use cookies
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "Authorization", "X-Requested-With"],
)

# Mount API routes
app.include_router(api_router)


@app.get("/")
def root():
    """Root endpoint with API summary and quick links."""
    return {
        "project": "FarmWatch AI",
        "problem_statement": "GenHack V1.0 PS 12 - Predicting Crop Stress From Field Images",
        "feature": "Visible Crop/Leaf Condition Classification",
        "docs_url": "/docs",
        "health_url": "/api/health",
        "model_info_url": "/api/model-info",
        "metrics_url": "/api/metrics",
    }
