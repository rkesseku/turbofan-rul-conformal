"""Cost-based maintenance scheduling from RUL predictions.

At the moment an engine is inspected we schedule maintenance ``trigger`` cycles ahead.
If the engine fails before then (trigger >= true RUL) we pay an unscheduled-failure cost;
otherwise we pay the preventive cost plus a charge for every cycle of useful life given up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostModel:
    preventive: float = 1.0
    failure: float = 10.0  # unscheduled failure costs 10x a planned shop visit
    waste_per_cycle: float = 0.01  # 100 unused cycles cost as much as one planned visit


def policy_cost(true_rul: np.ndarray, trigger: np.ndarray, cm: CostModel) -> np.ndarray:
    """Per-engine cost of maintaining ``trigger`` cycles from now."""
    true_rul = np.asarray(true_rul, dtype=float)
    trigger = np.clip(np.asarray(trigger, dtype=float), 0.0, None)
    fail = trigger >= true_rul
    return np.where(fail, cm.failure, cm.preventive + cm.waste_per_cycle * (true_rul - trigger))


def tune_margin(pred: np.ndarray, true_rul: np.ndarray, margins: np.ndarray,
                cm: CostModel) -> tuple[float, np.ndarray]:
    """Pick the safety margin m (trigger = pred - m) minimising mean cost on calibration engines."""
    costs = np.array([policy_cost(true_rul, pred - m, cm).mean() for m in margins])
    return float(margins[int(np.argmin(costs))]), costs


def summarise(true_rul: np.ndarray, trigger: np.ndarray, cm: CostModel) -> dict[str, float]:
    c = policy_cost(true_rul, trigger, cm)
    fail = np.clip(trigger, 0, None) >= np.asarray(true_rul)
    return {
        "mean_cost": float(c.mean()),
        "failure_rate": float(fail.mean()),
        "mean_wasted_cycles": float(
            np.mean(np.where(fail, 0.0, true_rul - np.clip(trigger, 0, None)))
        ),
    }
