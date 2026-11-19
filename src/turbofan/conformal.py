"""Split conformal prediction and conformalized quantile regression (Romano et al., 2019)."""

from __future__ import annotations

import math

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The ceil((n+1)(1-alpha))-th smallest calibration score (finite-sample valid)."""
    scores = np.sort(np.asarray(scores, dtype=float))
    n = len(scores)
    if n == 0:
        raise ValueError("need at least one calibration score")
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return float("inf")
    return float(scores[k - 1])


def split_conformal(cal_pred: np.ndarray, cal_y: np.ndarray, alpha: float) -> float:
    """Half-width q such that pred +/- q covers with probability >= 1 - alpha."""
    return conformal_quantile(np.abs(np.asarray(cal_y) - np.asarray(cal_pred)), alpha)


def cqr(cal_lo: np.ndarray, cal_hi: np.ndarray, cal_y: np.ndarray, alpha: float) -> float:
    """CQR correction q; the interval is [lo - q, hi + q]."""
    cal_y = np.asarray(cal_y)
    scores = np.maximum(np.asarray(cal_lo) - cal_y, cal_y - np.asarray(cal_hi))
    return conformal_quantile(scores, alpha)


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    y = np.asarray(y)
    return float(np.mean((y >= lo) & (y <= hi)))


def mean_width(lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean(np.asarray(hi) - np.asarray(lo)))
