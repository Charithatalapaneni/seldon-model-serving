"""
Pydantic request/response models for the gateway API.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    model_type: str = Field(default="image", description="image or text")
    data: list | str | None = None
    top_k: int = 5


class PredictionResult(BaseModel):
    label: str
    class_index: int = 0
    confidence: float = 0.0
    probabilities: dict = Field(default_factory=dict)


class PredictionResponse(BaseModel):
    model_type: str
    model_name: str = ""
    predictions: list[PredictionResult | dict] = []
    latency_ms: float = 0.0


class ModelInfo(BaseModel):
    name: str
    model_type: str
    versions: list[str] = []
    status: str = "loaded"


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str] = []
