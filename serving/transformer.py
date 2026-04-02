"""
Pre/post-processing transformer for Seldon inference pipelines.
Implements transform_input() and transform_output() per the Seldon interface.
"""

import logging
from io import BytesIO

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ImageTransformer:
    """Handles image pre/post-processing in a Seldon pipeline."""

    def __init__(self, target_size: int = 224):
        self.target_size = target_size
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def transform_input(self, X: np.ndarray, names: list = None, meta: dict = None) -> np.ndarray:
        """Preprocess: resize, center crop, normalize."""
        try:
            if isinstance(X, bytes):
                img = Image.open(BytesIO(X)).convert("RGB")
            elif isinstance(X, np.ndarray):
                if X.ndim == 1:
                    img = Image.open(BytesIO(X.tobytes())).convert("RGB")
                else:
                    img = Image.fromarray(X.astype(np.uint8)).convert("RGB")
            else:
                img = Image.fromarray(np.array(X).astype(np.uint8)).convert("RGB")

            # resize keeping aspect ratio, then center crop
            w, h = img.size
            scale = self.target_size / min(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.BILINEAR)

            left = (new_w - self.target_size) // 2
            top = (new_h - self.target_size) // 2
            img = img.crop((left, top, left + self.target_size, top + self.target_size))

            arr = np.array(img, dtype=np.float32) / 255.0
            arr = (arr - self.mean) / self.std
            # HWC -> CHW
            arr = arr.transpose(2, 0, 1)

            logger.info("Preprocessed image to shape %s", arr.shape)
            return arr.astype(np.float32)

        except Exception as e:
            logger.exception("Image preprocessing failed")
            raise

    def transform_output(self, X: np.ndarray, names: list = None, meta: dict = None) -> dict:
        """Postprocess: format raw model output into labeled predictions."""
        if isinstance(X, dict):
            return X
        # if raw logits/probs, just pass through
        return {"predictions": X.tolist() if isinstance(X, np.ndarray) else X}


class TextTransformer:
    """Handles text pre/post-processing in a Seldon pipeline."""

    def __init__(self, max_length: int = 128, lowercase: bool = True):
        self.max_length = max_length
        self.lowercase = lowercase

    def transform_input(self, X, names: list = None, meta: dict = None):
        """Clean and normalize text input."""
        if isinstance(X, str):
            texts = [X]
        elif isinstance(X, np.ndarray):
            texts = X.flatten().tolist()
        elif isinstance(X, list):
            texts = X
        else:
            texts = [str(X)]

        cleaned = []
        for text in texts:
            t = str(text).strip()
            if self.lowercase:
                t = t.lower()
            # truncate
            words = t.split()[:self.max_length]
            cleaned.append(" ".join(words))

        logger.info("Preprocessed %d text(s)", len(cleaned))
        return cleaned

    def transform_output(self, X, names: list = None, meta: dict = None) -> dict:
        """Format model output with labels."""
        if isinstance(X, dict):
            return X

        labels = ["negative", "positive"]
        if isinstance(X, np.ndarray):
            predictions = []
            for row in X:
                idx = int(np.argmax(row))
                predictions.append({
                    "label": labels[idx] if idx < len(labels) else str(idx),
                    "confidence": float(np.max(row)),
                    "probabilities": {labels[i]: float(row[i]) for i in range(len(row)) if i < len(labels)},
                })
            return {"predictions": predictions}

        return {"predictions": X}
