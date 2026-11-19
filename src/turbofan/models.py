"""Model zoo: gradient boosting, 1D-CNN, LSTM, and a Weibull AFT survival baseline."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from lifelines import WeibullAFTFitter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import nn

# ----------------------------------------------------------------------------- trees


def fit_lgbm(X: np.ndarray, y: np.ndarray, quantile: float | None = None, seed: int = 0,
             n_estimators: int = 600) -> lgb.LGBMRegressor:
    params = dict(n_estimators=n_estimators, learning_rate=0.03, num_leaves=31,
                  min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  random_state=seed, verbose=-1)
    if quantile is not None:
        params.update(objective="quantile", alpha=quantile)
    return lgb.LGBMRegressor(**params).fit(X, y)


def fit_xgb(
    X: np.ndarray, y: np.ndarray, seed: int = 0, n_estimators: int = 600
) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=0.03, max_depth=6,
                             subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                             tree_method="hist", random_state=seed, n_jobs=-1)
    return model.fit(X, y)


# ----------------------------------------------------------------------------- neural nets


class CNN1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, 32, 5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (n, window, channels)
        h = self.conv(x.transpose(1, 2))
        z = torch.cat([h.mean(dim=2), h[:, :, -1]], dim=1)
        return self.head(z).squeeze(-1)


class LSTMNet(nn.Module):
    def __init__(self, channels: int, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(channels, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def train_nn(kind: str, X: np.ndarray, y: np.ndarray, cap: float, epochs: int = 40,
             seed: int = 0, batch: int = 256, lr: float = 1e-3, patience: int = 6) -> nn.Module:
    """Train with MSE on y/cap; early stopping on a 10% random hold-out of the fit windows."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model: nn.Module = CNN1D(X.shape[2]) if kind == "cnn" else LSTMNet(X.shape[2])
    idx = rng.permutation(len(X))
    n_val = max(1, len(X) // 10)
    va, tr = idx[:n_val], idx[n_val:]
    Xt = torch.from_numpy(X.astype(np.float32))
    yt = torch.from_numpy((y / cap).astype(np.float32))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, bad = np.inf, None, 0
    for _ in range(epochs):
        model.train()
        perm = tr[rng.permutation(len(tr))]
        for i in range(0, len(perm), batch):
            b = perm[i : i + batch]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(Xt[b]), yt[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = nn.functional.mse_loss(model(Xt[va]), yt[va]).item()
        if v < best - 1e-6:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_nn(model: nn.Module, X: np.ndarray, cap: float, batch: int = 2048) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(model(torch.from_numpy(X[i : i + batch].astype(np.float32))).numpy())
    return np.clip(np.concatenate(out) * cap, 0.0, cap)


# ----------------------------------------------------------------------------- survival


class WeibullRUL:
    """Weibull accelerated-failure-time model of remaining life given the current window.

    Each (engine, cycle) snapshot is a subject whose duration is its uncapped remaining
    life (all training engines run to failure, so every duration is observed). Window
    features are standardised and reduced with PCA for a stable fit; the predicted median
    remaining life is the point estimate (capped like the other models).
    """

    def __init__(self, n_components: int = 8, penalizer: float = 0.01):
        self.n_components = n_components
        self.penalizer = penalizer

    def fit(self, F: np.ndarray, rul_true: np.ndarray) -> WeibullRUL:
        self.scaler = StandardScaler().fit(F)
        k = min(self.n_components, F.shape[1], len(F) - 1)
        self.pca = PCA(k, random_state=0).fit(self.scaler.transform(F))
        df = self._frame(F)
        df["T"] = np.asarray(rul_true, dtype=float) + 1.0
        df["E"] = 1
        self.aft = WeibullAFTFitter(penalizer=self.penalizer).fit(df, "T", "E")
        return self

    def _frame(self, F: np.ndarray) -> pd.DataFrame:
        Z = self.pca.transform(self.scaler.transform(F))
        return pd.DataFrame(Z, columns=[f"pc{i}" for i in range(Z.shape[1])])

    def predict(self, F: np.ndarray, cap: float) -> np.ndarray:
        med = np.asarray(self.aft.predict_median(self._frame(F)), dtype=float) - 1.0
        return np.clip(np.nan_to_num(med, nan=cap, posinf=cap), 0.0, cap)
