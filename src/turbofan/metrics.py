"""Point-prediction metrics for RUL."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(p) - np.asarray(y))))


def nasa_score(y: np.ndarray, p: np.ndarray) -> float:
    """PHM08 asymmetric score: late predictions (p > y) are penalised more than early ones."""
    d = np.asarray(p, dtype=float) - np.asarray(y, dtype=float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def bootstrap_ci(y: np.ndarray, p: np.ndarray, fn: Callable[[np.ndarray, np.ndarray], float],
                 n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap CI over test engines."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    stats = np.array([fn(y[i], p[i]) for i in idx])
    a = (1 - level) / 2
    return float(np.quantile(stats, a)), float(np.quantile(stats, 1 - a))
