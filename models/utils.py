"""
Model save/load helpers, version tagging, device selection.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_model(
    model: nn.Module,
    name: str,
    version: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Path:
    """Save model state dict with version tag."""
    version = version or datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = ARTIFACTS_DIR / name / version
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / "model.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "model_name": name,
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata or {},
    }, model_path)

    logger.info("Saved %s v%s to %s", name, version, model_path)
    return model_path


def load_model(
    model: nn.Module,
    name: str,
    version: str = "latest",
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Load model state dict. 'latest' picks the most recent version."""
    device = device or get_device()
    model_dir = ARTIFACTS_DIR / name

    if version == "latest":
        versions = sorted(model_dir.iterdir()) if model_dir.exists() else []
        if not versions:
            raise FileNotFoundError(f"No saved versions for model '{name}'")
        version_dir = versions[-1]
    else:
        version_dir = model_dir / version

    model_path = version_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    logger.info(
        "Loaded %s v%s (saved %s)",
        checkpoint.get("model_name", name),
        checkpoint.get("version", version),
        checkpoint.get("timestamp", "unknown"),
    )
    return model


def list_versions(name: str) -> list[str]:
    """List available versions for a model."""
    model_dir = ARTIFACTS_DIR / name
    if not model_dir.exists():
        return []
    return sorted([d.name for d in model_dir.iterdir() if d.is_dir()])


def get_model_info(name: str, version: str = "latest") -> dict:
    """Get metadata for a saved model."""
    model_dir = ARTIFACTS_DIR / name
    if version == "latest":
        versions = sorted(model_dir.iterdir()) if model_dir.exists() else []
        if not versions:
            return {}
        version_dir = versions[-1]
    else:
        version_dir = model_dir / version

    model_path = version_dir / "model.pt"
    if not model_path.exists():
        return {}

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    return {
        "name": checkpoint.get("model_name", name),
        "version": checkpoint.get("version", version),
        "timestamp": checkpoint.get("timestamp"),
        "metadata": checkpoint.get("metadata", {}),
    }
