"""
Python client for batch predictions against the gateway or a Seldon endpoint.
"""

import logging
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_GATEWAY_URL = "http://localhost:8000"
DEFAULT_SELDON_URL = "http://localhost:9000/api/v1.0/predictions"


class PredictionClient:
    """Client for making predictions against the model serving gateway."""

    def __init__(self, base_url: str = DEFAULT_GATEWAY_URL, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def predict_image(self, image_path: str | Path, top_k: int = 5) -> dict:
        """Upload an image file for classification."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        with open(path, "rb") as f:
            resp = self._client.post(
                f"{self.base_url}/predict/image",
                files={"file": (path.name, f, "image/jpeg")},
            )
        resp.raise_for_status()
        return resp.json()

    def predict_text(self, text: str | list[str]) -> dict:
        """Run text classification."""
        data = text if isinstance(text, list) else [text]
        resp = self._client.post(
            f"{self.base_url}/predict/text",
            json={"model_type": "text", "data": data},
        )
        resp.raise_for_status()
        return resp.json()

    def predict_batch_text(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        """Run text classification in batches."""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self.predict_text(batch)
            results.extend(resp.get("predictions", []))
        return results

    def predict_seldon(self, data, model_type: str = "image") -> dict:
        """Generic Seldon-style prediction."""
        resp = self._client.post(
            f"{self.base_url}/predict/seldon",
            json={"model_type": model_type, "data": data},
        )
        resp.raise_for_status()
        return resp.json()

    def list_models(self) -> list[dict]:
        resp = self._client.get(f"{self.base_url}/models")
        resp.raise_for_status()
        return resp.json()

    def health(self) -> dict:
        resp = self._client.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class SeldonClient:
    """Client for talking directly to a Seldon Core endpoint."""

    def __init__(self, url: str = DEFAULT_SELDON_URL, timeout: float = 30.0):
        self.url = url
        self._client = httpx.Client(timeout=timeout)

    def predict(self, data: list) -> dict:
        payload = {"data": {"ndarray": data}}
        resp = self._client.post(self.url, json=payload)
        resp.raise_for_status()
        return resp.json()

    def predict_raw(self, payload: dict) -> dict:
        resp = self._client.post(self.url, json=payload)
        resp.raise_for_status()
        return resp.json()

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
