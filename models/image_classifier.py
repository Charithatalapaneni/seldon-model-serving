"""
ResNet18-based image classifier. Uses torchvision pretrained weights
and maps predictions to ImageNet class labels.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

logger = logging.getLogger(__name__)

# imagenet labels (top-level, trimmed for the common ones)
IMAGENET_LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
_labels: Optional[list[str]] = None


def _get_labels() -> list[str]:
    global _labels
    if _labels is None:
        try:
            import urllib.request
            resp = urllib.request.urlopen(IMAGENET_LABELS_URL)
            _labels = [line.decode("utf-8").strip() for line in resp.readlines()]
        except Exception:
            # fallback: just use indices
            _labels = [str(i) for i in range(1000)]
    return _labels


class ImageClassifier(nn.Module):
    """Wraps a pretrained ResNet18 for inference."""

    def __init__(self, num_classes: int = 1000, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.model = models.resnet18(weights=weights)

        if num_classes != 1000:
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """PIL image -> normalized tensor with batch dim."""
        return self.transform(image).unsqueeze(0)

    @torch.no_grad()
    def predict(self, image: Image.Image, top_k: int = 5) -> list[dict]:
        """Run inference on a PIL image, return top-k predictions."""
        self.eval()
        tensor = self.preprocess(image).to(next(self.parameters()).device)
        output = self.forward(tensor)
        probs = torch.softmax(output, dim=1)
        top_probs, top_indices = probs.topk(top_k, dim=1)

        labels = _get_labels()
        results = []
        for i in range(top_k):
            idx = top_indices[0][i].item()
            label = labels[idx] if idx < len(labels) else str(idx)
            results.append({
                "label": label,
                "class_index": idx,
                "confidence": round(top_probs[0][i].item(), 4),
            })
        return results

    @torch.no_grad()
    def predict_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Raw tensor in, logits out. For pipeline use."""
        self.eval()
        return self.forward(tensor)


def load_image_classifier(
    pretrained: bool = True,
    device: Optional[torch.device] = None,
) -> ImageClassifier:
    from models.utils import get_device
    device = device or get_device()
    model = ImageClassifier(pretrained=pretrained)
    model.to(device)
    model.eval()
    logger.info("Image classifier loaded (device=%s)", device)
    return model
