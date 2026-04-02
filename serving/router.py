"""
Ensemble router for Seldon deployments. Supports traffic splitting
(A/B testing), averaging, and majority voting across model replicas.
"""

import logging
import random
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class EnsembleRouter:
    """
    Routes requests across multiple model versions.
    Seldon calls route() to pick which child predictor handles the request.
    """

    def __init__(
        self,
        n_models: int = 2,
        strategy: str = "traffic_split",
        weights: Optional[list[float]] = None,
    ):
        self.n_models = n_models
        self.strategy = strategy
        # default: even split
        self.weights = weights or [1.0 / n_models] * n_models
        self._request_count = 0

        logger.info(
            "Router initialized: strategy=%s, n_models=%d, weights=%s",
            strategy, n_models, self.weights,
        )

    def route(self, X: np.ndarray, names: list = None, meta: dict = None) -> int:
        """
        Seldon route interface. Returns the index of the model to use.
        """
        self._request_count += 1

        if self.strategy == "traffic_split":
            return self._traffic_split()
        elif self.strategy == "round_robin":
            return self._round_robin()
        elif self.strategy == "random":
            return random.randint(0, self.n_models - 1)
        else:
            return 0

    def _traffic_split(self) -> int:
        """Weighted random selection based on traffic split percentages."""
        r = random.random()
        cumulative = 0.0
        for i, w in enumerate(self.weights):
            cumulative += w
            if r <= cumulative:
                return i
        return self.n_models - 1

    def _round_robin(self) -> int:
        return self._request_count % self.n_models

    def aggregate(self, X: list[np.ndarray], names: list = None, meta: dict = None) -> np.ndarray:
        """
        Aggregate predictions from multiple models.
        Used when strategy is 'average' or 'vote'.
        """
        if not X:
            return np.array([])

        if self.strategy == "average":
            return np.mean(X, axis=0)
        elif self.strategy == "vote":
            # majority voting on argmax
            votes = [np.argmax(pred, axis=-1) for pred in X]
            stacked = np.stack(votes, axis=0)
            # mode along model axis
            from scipy import stats
            result = stats.mode(stacked, axis=0, keepdims=False)
            return result.mode
        else:
            # default: just return first model's output
            return X[0]

    def get_stats(self) -> dict:
        return {
            "strategy": self.strategy,
            "n_models": self.n_models,
            "weights": self.weights,
            "total_requests": self._request_count,
        }


def create_canary_router(
    canary_weight: float = 0.1,
) -> EnsembleRouter:
    """Convenience: create a router for canary deployment (stable vs canary)."""
    return EnsembleRouter(
        n_models=2,
        strategy="traffic_split",
        weights=[1.0 - canary_weight, canary_weight],
    )
