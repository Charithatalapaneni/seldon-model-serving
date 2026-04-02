"""
Prometheus metrics for model serving.
"""

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

REGISTRY = CollectorRegistry()

REQUEST_COUNT = Counter(
    "model_serving_requests_total",
    "Total prediction requests",
    ["model_type", "status"],
    registry=REGISTRY,
)

ERROR_COUNT = Counter(
    "model_serving_errors_total",
    "Prediction errors",
    ["model_type"],
    registry=REGISTRY,
)

PREDICTION_LATENCY = Histogram(
    "model_serving_prediction_seconds",
    "Prediction latency",
    ["model_type"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    registry=REGISTRY,
)

BATCH_SIZE = Histogram(
    "model_serving_batch_size",
    "Batch size per prediction request",
    ["model_type"],
    buckets=[1, 2, 4, 8, 16, 32, 64, 128],
    registry=REGISTRY,
)

ACTIVE_REQUESTS = Gauge(
    "model_serving_active_requests",
    "In-flight requests",
    registry=REGISTRY,
)

MODELS_LOADED = Gauge(
    "model_serving_models_loaded",
    "Number of models currently loaded",
    registry=REGISTRY,
)


def get_metrics() -> bytes:
    return generate_latest(REGISTRY)


def get_content_type() -> str:
    return CONTENT_TYPE_LATEST
