"""
FastAPI gateway — local model management and prediction endpoint.
Wraps the Seldon predictor components for standalone testing.
"""

import logging
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image

from gateway.schemas import (
    HealthResponse,
    ModelInfo,
    PredictionRequest,
    PredictionResponse,
)
from monitoring.metrics import (
    ACTIVE_REQUESTS,
    ERROR_COUNT,
    PREDICTION_LATENCY,
    REQUEST_COUNT,
    get_metrics,
    get_content_type,
)
from serving.predictor import ModelPredictor
from serving.router import EnsembleRouter, create_canary_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Model Serving Gateway",
    version="1.0.0",
    docs_url="/docs",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# model registry — loaded on first use
_predictors: dict[str, ModelPredictor] = {}
_router: EnsembleRouter | None = None


def _get_predictor(model_type: str) -> ModelPredictor:
    if model_type not in _predictors:
        pred = ModelPredictor(model_type=model_type, model_name=model_type)
        pred.load()
        _predictors[model_type] = pred
    return _predictors[model_type]


@app.post("/predict/image", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """Run image classification on an uploaded image file."""
    ACTIVE_REQUESTS.inc()
    start = time.perf_counter()
    try:
        content = await file.read()
        img = Image.open(BytesIO(content)).convert("RGB")

        predictor = _get_predictor("image")
        results = predictor.model.predict(img, top_k=5)

        latency = (time.perf_counter() - start) * 1000
        PREDICTION_LATENCY.labels(model_type="image").observe(latency / 1000)
        REQUEST_COUNT.labels(model_type="image", status="200").inc()

        return PredictionResponse(
            model_type="image",
            model_name="resnet18",
            predictions=results,
            latency_ms=round(latency, 2),
        )
    except Exception as e:
        ERROR_COUNT.labels(model_type="image").inc()
        REQUEST_COUNT.labels(model_type="image", status="500").inc()
        logger.exception("Image prediction failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        ACTIVE_REQUESTS.dec()


@app.post("/predict/text", response_model=PredictionResponse)
async def predict_text(request: PredictionRequest):
    """Run text classification on input text."""
    ACTIVE_REQUESTS.inc()
    start = time.perf_counter()
    try:
        predictor = _get_predictor("text")
        texts = request.data if isinstance(request.data, list) else [request.data or ""]

        predictions = []
        for text in texts:
            result = predictor.model.predict(str(text), predictor.vocab)
            predictions.append(result)

        latency = (time.perf_counter() - start) * 1000
        PREDICTION_LATENCY.labels(model_type="text").observe(latency / 1000)
        REQUEST_COUNT.labels(model_type="text", status="200").inc()

        return PredictionResponse(
            model_type="text",
            model_name="lstm_sentiment",
            predictions=predictions,
            latency_ms=round(latency, 2),
        )
    except Exception as e:
        ERROR_COUNT.labels(model_type="text").inc()
        REQUEST_COUNT.labels(model_type="text", status="500").inc()
        logger.exception("Text prediction failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        ACTIVE_REQUESTS.dec()


@app.post("/predict/seldon", response_model=PredictionResponse)
async def predict_seldon(request: PredictionRequest):
    """Generic Seldon-style predict endpoint (matches Seldon's REST API)."""
    ACTIVE_REQUESTS.inc()
    start = time.perf_counter()
    try:
        predictor = _get_predictor(request.model_type)
        data = np.array(request.data) if request.data else np.array([])
        result = predictor.predict(data)

        latency = (time.perf_counter() - start) * 1000
        PREDICTION_LATENCY.labels(model_type=request.model_type).observe(latency / 1000)

        return PredictionResponse(
            model_type=request.model_type,
            predictions=result.get("predictions", []),
            latency_ms=round(latency, 2),
        )
    except Exception as e:
        ERROR_COUNT.labels(model_type=request.model_type).inc()
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        ACTIVE_REQUESTS.dec()


@app.get("/models", response_model=list[ModelInfo])
async def list_models():
    """List all loaded models and their versions."""
    from models.utils import list_versions
    models = []
    for model_type in ["image", "text"]:
        versions = list_versions(f"{model_type}_classifier")
        models.append(ModelInfo(
            name=f"{model_type}_classifier",
            model_type=model_type,
            versions=versions,
            status="loaded" if model_type in _predictors else "available",
        ))
    return models


@app.get("/models/{model_type}/versions")
async def get_model_versions(model_type: str):
    from models.utils import list_versions
    name = f"{model_type}_classifier"
    versions = list_versions(name)
    return {"model": name, "versions": versions}


@app.get("/router/stats")
async def router_stats():
    """Get ensemble router statistics."""
    global _router
    if _router is None:
        _router = create_canary_router(canary_weight=0.1)
    return _router.get_stats()


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        models_loaded=list(_predictors.keys()),
    )


@app.get("/metrics")
async def prometheus_metrics():
    return Response(content=get_metrics(), media_type=get_content_type())
