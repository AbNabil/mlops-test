"""
FastAPI model serving endpoint.

From a DevOps perspective this is just a regular web service — it happens to
load an ML model instead of querying a database. The patterns are identical:
health checks, structured logging, versioned endpoints, graceful startup.
"""

import os
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Add project root to path so we can import src/
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import IrisPredictor

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("iris-api")

# ── Global model instance (loaded once at startup) ────────────────────────────
predictor: Optional[IrisPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model when the server starts, release on shutdown."""
    global predictor
    model_dir = os.getenv("MODEL_DIR", "models")
    logger.info(f"Loading model from {model_dir}...")
    try:
        predictor = IrisPredictor(model_dir=model_dir)
        logger.info("Model loaded successfully")
    except FileNotFoundError as e:
        logger.error(f"Failed to load model: {e}")
        # We still start the server — /health will report degraded state
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Iris Classifier API",
    description="MLOps demo — serving a scikit-learn model via FastAPI",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response schemas ────────────────────────────────────────────────
class PredictRequest(BaseModel):
    sepal_length: float = Field(..., gt=0, description="Sepal length in cm", example=5.1)
    sepal_width:  float = Field(..., gt=0, description="Sepal width in cm",  example=3.5)
    petal_length: float = Field(..., gt=0, description="Petal length in cm", example=1.4)
    petal_width:  float = Field(..., gt=0, description="Petal width in cm",  example=0.2)


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    all_probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str          # "ok" | "degraded"
    model_loaded: bool
    version: str


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    """
    Health check endpoint — wire this to your load balancer / k8s liveness probe.
    Returns 200 even when the model isn't loaded so the container stays up and
    you can diagnose the issue without a crash loop.
    """
    return {
        "status": "ok" if predictor is not None else "degraded",
        "model_loaded": predictor is not None,
        "version": app.version,
    }


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(request: PredictRequest):
    """Run inference on a single Iris sample."""
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run the training job first.",
        )

    features = {
        "sepal_length": request.sepal_length,
        "sepal_width":  request.sepal_width,
        "petal_length": request.petal_length,
        "petal_width":  request.petal_width,
    }

    logger.info(f"Prediction request: {features}")
    result = predictor.predict(features)
    logger.info(f"Prediction result: {result['prediction']} ({result['confidence']:.2%})")

    return result


@app.get("/model/info", tags=["ops"])
def model_info():
    """Return metadata about the currently loaded model."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "class_names": predictor.class_names,
        "feature_names": predictor.feature_names,
        "model_type": type(predictor.model).__name__,
    }
