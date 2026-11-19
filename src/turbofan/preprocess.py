"""Operating-condition normalisation, sensor selection, sliding windows and window features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.cluster import KMeans

from .data import OS_COLS, SENSOR_COLS


class ConditionNormalizer:
    """Standardise each sensor within its operating condition, fitted on training engines only.

    FD002/FD004 run under six operating conditions that shift sensor levels far more than
    degradation does; clustering the three operating settings and standardising per cluster
    removes that shift. Sensors that are (near-)constant or unrelated to RUL are dropped.
    """

    def __init__(self, n_conditions: int = 1, min_unique: int = 4, min_abs_corr: float = 0.05):
        self.n_conditions = n_conditions
        self.min_unique = min_unique
        self.min_abs_corr = min_abs_corr
        self.km: KMeans | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.sensors_: list[str] = []

    def _conditions(self, df: pd.DataFrame) -> np.ndarray:
        if self.km is None:
            return np.zeros(len(df), dtype=int)
        return self.km.predict(df[OS_COLS].to_numpy(float))

    def fit(self, df: pd.DataFrame) -> ConditionNormalizer:
        if self.n_conditions > 1:
            self.km = KMeans(self.n_conditions, n_init=10, random_state=0)
            self.km.fit(df[OS_COLS].to_numpy(float))
        cond = self._conditions(df)
        raw = df[SENSOR_COLS].to_numpy(float)
        k = self.n_conditions
        self.mean_ = np.zeros((k, raw.shape[1]))
        self.std_ = np.ones((k, raw.shape[1]))
        nunique = np.zeros((k, raw.shape[1]))
        for c in range(k):
            block = raw[cond == c]
            if len(block) == 0:
                continue
            self.mean_[c] = block.mean(axis=0)
            sd = block.std(axis=0)
            self.std_[c] = np.where(sd > 1e-12, sd, 1.0)
            nunique[c] = [len(np.unique(block[:, j])) for j in range(raw.shape[1])]
        z = (raw - self.mean_[cond]) / self.std_[cond]
        keep = []
        for j, s in enumerate(SENSOR_COLS):
            if np.median(nunique[:, j]) < self.min_unique or z[:, j].std() < 1e-6:
                continue
            if "rul" in df:
                corr = pd.Series(z[:, j]).corr(df["rul"].reset_index(drop=True), method="spearman")
                if not np.isfinite(corr) or abs(corr) < self.min_abs_corr:
                    continue
            keep.append(s)
        self.sensors_ = keep
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.mean_ is None:
            raise RuntimeError("fit() must be called first")
        cond = self._conditions(df)
        raw = df[SENSOR_COLS].to_numpy(float)
        z = (raw - self.mean_[cond]) / self.std_[cond]
        out = df[["unit", "cycle"]].copy()
        idx = [SENSOR_COLS.index(s) for s in self.sensors_]
        for j, s in zip(idx, self.sensors_, strict=True):
            out[s] = z[:, j]
        out["cycle_s"] = df["cycle"].to_numpy(float) / 100.0
        for col in ("rul", "rul_true"):
            if col in df:
                out[col] = df[col].to_numpy(float)
        return out

    @property
    def channels(self) -> list[str]:
        return [*self.sensors_, "cycle_s"]


@dataclass
class Windows:
    """Sliding windows ending at each cycle (or only the last cycle for test engines)."""

    X: np.ndarray  # (n, window, channels), float32
    y: np.ndarray  # capped RUL at the window end
    y_true: np.ndarray  # uncapped RUL at the window end
    unit: np.ndarray
    cycle: np.ndarray


def make_windows(df: pd.DataFrame, channels: list[str], window: int, last_only: bool = False,
                 stride: int = 1) -> Windows:
    """Build windows per engine; engines shorter than ``window`` are front-padded with row 0."""
    Xs, ys, yts, us, cs = [], [], [], [], []
    for unit, g in df.groupby("unit", sort=True):
        a = g[channels].to_numpy(np.float32)
        y = g["rul"].to_numpy(float)
        yt = g["rul_true"].to_numpy(float)
        cyc = g["cycle"].to_numpy(int)
        if len(a) < window:
            pad = window - len(a)
            a = np.vstack([np.repeat(a[:1], pad, axis=0), a])
            y = np.concatenate([np.repeat(y[:1], pad), y])
            yt = np.concatenate([np.repeat(yt[:1], pad), yt])
            cyc = np.concatenate([np.repeat(cyc[:1], pad), cyc])
        w = sliding_window_view(a, window, axis=0).transpose(0, 2, 1)
        ends = np.arange(window - 1, len(a))
        if last_only:
            w, ends = w[-1:], ends[-1:]
        elif stride > 1:
            keep = np.arange(len(ends) - 1, -1, -stride)[::-1]
            w, ends = w[keep], ends[keep]
        Xs.append(np.ascontiguousarray(w))
        ys.append(y[ends])
        yts.append(yt[ends])
        us.append(np.full(len(ends), unit))
        cs.append(cyc[ends])
    return Windows(np.concatenate(Xs), np.concatenate(ys), np.concatenate(yts),
                   np.concatenate(us), np.concatenate(cs))


def window_features(X: np.ndarray) -> np.ndarray:
    """Per-channel summary features: last, mean, std, min, max, slope, last-5 mean."""
    n, w, c = X.shape
    t = np.arange(w, dtype=float) - (w - 1) / 2.0
    slope = np.einsum("nwc,w->nc", X, t) / float((t**2).sum())
    feats = [X[:, -1, :], X.mean(1), X.std(1), X.min(1), X.max(1), slope,
             X[:, -min(5, w):, :].mean(1)]
    return np.concatenate(feats, axis=1).astype(np.float32)
