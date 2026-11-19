"""Model zoo: gradient boosting, 1D-CNN, LSTM, and a Weibull AFT survival baseline.

Model selection (boosting rounds, network epochs and the small network tuning grid) always
uses a group split of the *fit* engines: whole engines are held out for early stopping, so
overlapping windows from one engine never sit on both sides. Calibration and test engines
are never seen here.
"""

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

N_THREADS = 2  # shared machine: cap every model at two threads

# ----------------------------------------------------------------------------- group split


def group_holdout(units: np.ndarray, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks (train, holdout) over rows, holding out ``frac`` of the distinct engines.

    Every row of a given engine lands on the same side, so no engine is on both sides.
    """
    units = np.asarray(units)
    uniq = np.sort(np.unique(units))
    if len(uniq) < 2:
        raise ValueError("need at least two engines for a group hold-out")
    rng = np.random.default_rng(seed)
    n_hold = min(len(uniq) - 1, max(1, int(round(frac * len(uniq)))))
    held = rng.permutation(uniq)[:n_hold]
    va = np.isin(units, held)
    return ~va, va


# ----------------------------------------------------------------------------- trees


def fit_lgbm(X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
             quantile: float | None = None, seed: int = 0, n_estimators: int = 2000,
             patience: int = 100) -> lgb.LGBMRegressor:
    """LightGBM with the number of rounds chosen by early stopping on held-out engines."""
    params = dict(n_estimators=n_estimators, learning_rate=0.03, num_leaves=31,
                  min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  random_state=seed, verbose=-1, n_jobs=N_THREADS)
    metric = "l2"
    if quantile is not None:
        params.update(objective="quantile", alpha=quantile)
        metric = "quantile"
    model = lgb.LGBMRegressor(**params)
    model.fit(X, y, eval_set=[(X_val, y_val)], eval_metric=metric,
              callbacks=[lgb.early_stopping(patience, first_metric_only=True, verbose=False)])
    return model


def fit_xgb(X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, seed: int = 0,
            n_estimators: int = 2000, patience: int = 100) -> xgb.XGBRegressor:
    """XGBoost with the number of rounds chosen by early stopping on held-out engines."""
    model = xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=0.03, max_depth=6,
                             subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                             tree_method="hist", random_state=seed, n_jobs=N_THREADS,
                             early_stopping_rounds=patience)
    return model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)


def best_rounds(model) -> int:
    """Boosting rounds kept after early stopping."""
    if isinstance(model, lgb.LGBMRegressor):
        return int(model.best_iteration_ or model.n_estimators)
    return int(model.best_iteration) + 1


# ----------------------------------------------------------------------------- neural nets


class CNN1D(nn.Module):
    def __init__(self, channels: int, width: int = 64):
        super().__init__()
        c1 = width // 2
        self.conv = nn.Sequential(
            nn.Conv1d(channels, c1, 5, padding=2), nn.ReLU(),
            nn.Conv1d(c1, width, 5, padding=2), nn.ReLU(),
            nn.Conv1d(width, width, 3, padding=1), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(2 * width, width), nn.ReLU(), nn.Dropout(0.2),
                                  nn.Linear(width, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (n, window, channels)
        h = self.conv(x.transpose(1, 2))
        z = torch.cat([h.mean(dim=2), h[:, :, -1]], dim=1)
        return self.head(z).squeeze(-1)


class LSTMNet(nn.Module):
    def __init__(self, channels: int, width: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(channels, width, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(width, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def train_nn(kind: str, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
             cap: float, epochs: int = 40, seed: int = 0, batch: int = 256, lr: float = 1e-3,
             width: int = 64, patience: int = 6) -> tuple[nn.Module, dict]:
    """Train with MSE on y/cap; early stopping on windows of held-out engines.

    Returns the best-epoch model and a record with the held-out RMSE (cycles) and epochs run.
    """
    torch.manual_seed(seed)
    torch.set_num_threads(N_THREADS)
    rng = np.random.default_rng(seed)
    model: nn.Module = CNN1D(X.shape[2], width) if kind == "cnn" else LSTMNet(X.shape[2], width)
    Xt = torch.from_numpy(X.astype(np.float32))
    yt = torch.from_numpy((y / cap).astype(np.float32))
    Xv = torch.from_numpy(X_val.astype(np.float32))
    yv = torch.from_numpy((y_val / cap).astype(np.float32))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, bad, best_ep, ep = np.inf, None, 0, 0, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(len(X))
        for i in range(0, len(perm), batch):
            b = perm[i : i + batch]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(Xt[b]), yt[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = sum(nn.functional.mse_loss(model(Xv[i : i + 4096]), yv[i : i + 4096],
                                           reduction="sum").item()
                    for i in range(0, len(Xv), 4096)) / len(Xv)
        if v < best - 1e-6:
            best, bad, best_ep = v, 0, ep
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"val_rmse": float(np.sqrt(best) * cap), "best_epoch": best_ep,
                   "epochs_run": ep}


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
