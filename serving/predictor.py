"""
Seldon-compatible model predictor. Implements predict() following
the Seldon Python wrapper interface so it can be dropped into a
SeldonDeployment as-is.

Also works standalone for local testing.
"""

import logging
from typing import Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class ModelPredictor:
    """
    Generic predictor that wraps a PyTorch model for Seldon Core.
    Handles both image and text model types.
    """

    def __init__(self, model_type: str = "image", model_name: str = "default"):
        self.model_type = model_type
        self.model_name = model_name
        self.model = None
        self.vocab = None
        self.device = None
        self.ready = False

    def load(self):
        """Called by Seldon on startup to load the model."""
        from models.utils import get_device
        self.device = get_device()

        if self.model_type == "image":
            from models.image_classifier import load_image_classifier
            self.model = load_image_classifier(device=self.device)
        elif self.model_type == "text":
            from models.text_classifier import load_text_classifier
            self.model = load_text_classifier(device=self.device)
            self._load_vocab()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.ready = True
        logger.info("Predictor loaded: type=%s, device=%s", self.model_type, self.device)

    def _load_vocab(self):
        """Load vocab for text models."""
        import json
        from models.utils import ARTIFACTS_DIR
        vocab_path = ARTIFACTS_DIR / "text_classifier" / "vocab.json"
        if vocab_path.exists():
            with open(vocab_path) as f:
                self.vocab = json.load(f)
            logger.info("Loaded vocab (%d tokens)", len(self.vocab))
        else:
            # build a minimal default vocab
            from models.text_classifier import build_vocab
            self.vocab = build_vocab([], max_size=5000)
            logger.warning("No saved vocab found, using empty default")

    def predict(self, X: np.ndarray, names: list = None, meta: dict = None) -> dict:
        """
        Seldon predict interface.

        For image models: X is a numpy array (batch of images or raw bytes).
        For text models: X is a list of strings.

        Returns dict with predictions.
        """
        if not self.ready:
            self.load()

        try:
            if self.model_type == "image":
                return self._predict_image(X)
            elif self.model_type == "text":
                return self._predict_text(X)
        except Exception as e:
            logger.exception("Prediction failed")
            return {"error": str(e)}

    def _predict_image(self, X: np.ndarray) -> dict:
        if isinstance(X, np.ndarray) and X.ndim >= 3:
            # raw pixel data
            tensor = torch.from_numpy(X).float()
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            tensor = tensor.to(self.device)
            logits = self.model.predict_tensor(tensor)
            probs = torch.softmax(logits, dim=1)
            return {
                "predictions": probs.cpu().numpy().tolist(),
            }
        else:
            # assume it's a single image array
            from PIL import Image
            img = Image.fromarray(X.astype(np.uint8))
            results = self.model.predict(img, top_k=5)
            return {"predictions": results}

    def _predict_text(self, X) -> dict:
        results = []
        texts = X if isinstance(X, list) else [str(X)]
        for text in texts:
            pred = self.model.predict(str(text), self.vocab)
            results.append(pred)
        return {"predictions": results}

    def health_status(self) -> dict:
        return {
            "status": "ready" if self.ready else "not_loaded",
            "model_type": self.model_type,
            "model_name": self.model_name,
        }
