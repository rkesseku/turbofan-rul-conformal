"""Synthetic C-MAPSS-format files for tests and smoke runs (no NASA data needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _engine(rng: np.random.Generator, life: int, n_cond: int) -> np.ndarray:
    t = np.arange(1, life + 1)
    health = np.exp(3.0 * (t / life)) - 1.0  # exponential degradation to failure
    cond = rng.integers(0, n_cond, size=life)
    os_ = np.stack([cond * 10.0, cond * 0.1, np.full(life, 100.0)], axis=1)
    sens = np.empty((life, 21))
    for j in range(21):
        level = 500.0 + 3.0 * j + 20.0 * cond  # operating condition shifts every sensor
        if j % 4 == 0:
            sens[:, j] = level  # constant sensor (should be dropped)
        else:
            sign = 1.0 if j % 2 else -1.0
            sens[:, j] = level + sign * 0.5 * health + rng.normal(0, 0.3, life)
    return np.column_stack([np.zeros(life), t, os_, sens])


def write_subset(root: str | Path, name: str = "FD001", n_train: int = 30, n_test: int = 12,
                 n_cond: int = 1, seed: int = 0) -> Path:
    """Write train_/test_/RUL_ files mimicking C-MAPSS for ``n_train`` and ``n_test`` engines."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    train = []
    for u in range(1, n_train + 1):
        e = _engine(rng, int(rng.integers(120, 220)), n_cond)
        e[:, 0] = u
        train.append(e)
    test, rul = [], []
    for u in range(1, n_test + 1):
        life = int(rng.integers(120, 220))
        cut = int(rng.integers(20, life - 5))
        e = _engine(rng, life, n_cond)[:cut]
        e[:, 0] = u
        test.append(e)
        rul.append(life - cut)
    fmt = "%.4f"
    np.savetxt(root / f"train_{name}.txt", np.vstack(train), fmt=fmt)
    np.savetxt(root / f"test_{name}.txt", np.vstack(test), fmt=fmt)
    np.savetxt(root / f"RUL_{name}.txt", np.array(rul), fmt="%d")
    return root
