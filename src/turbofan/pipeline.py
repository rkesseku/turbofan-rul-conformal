"""End-to-end experiment: fit models, conformalise, evaluate on official test sets, decide."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import conformal as cf
from .data import N_CONDITIONS, SUBSETS, load_subset, split_units
from .decision import CostModel, summarise, tune_margin
from .metrics import bootstrap_ci, mae, nasa_score, rmse
from .models import WeibullRUL, fit_lgbm, fit_xgb, predict_nn, train_nn
from .preprocess import ConditionNormalizer, make_windows, window_features


@dataclass
class Config:
    cap: float = 125.0
    window: int = 30
    cal_frac: float = 0.25
    alpha: float = 0.10
    seed: int = 0
    epochs: int = 40
    n_estimators: int = 600
    survival_stride: int = 5
    n_boot: int = 2000
    subsets: tuple[str, ...] = SUBSETS
    cost: CostModel = field(default_factory=CostModel)


def run_subset(root: str | Path, name: str, cfg: Config, log=print) -> dict:
    t0 = time.time()
    sub = load_subset(root, name, cap=cfg.cap)
    fit_u, cal_u = split_units(sub.train["unit"].to_numpy(), cfg.cal_frac, cfg.seed)
    fit_df = sub.train[sub.train["unit"].isin(fit_u)]
    cal_df = sub.train[sub.train["unit"].isin(cal_u)]

    norm = ConditionNormalizer(N_CONDITIONS.get(name, 1)).fit(fit_df)
    ch = norm.channels
    w_fit = make_windows(norm.transform(fit_df), ch, cfg.window)
    w_cal = make_windows(norm.transform(cal_df), ch, cfg.window)
    w_test = make_windows(norm.transform(sub.test), ch, cfg.window, last_only=True)
    F_fit, F_cal, F_test = (window_features(w.X) for w in (w_fit, w_cal, w_test))
    log(f"[{name}] {len(fit_u)} fit / {len(cal_u)} cal / {len(w_test.y)} test engines; "
        f"{len(w_fit.y)} fit windows; sensors={norm.sensors_}")

    preds_cal: dict[str, np.ndarray] = {}
    preds_test: dict[str, np.ndarray] = {}

    def clip(p: np.ndarray) -> np.ndarray:
        return np.clip(p, 0.0, cfg.cap)

    m = fit_lgbm(F_fit, w_fit.y, seed=cfg.seed, n_estimators=cfg.n_estimators)
    preds_cal["LightGBM"], preds_test["LightGBM"] = clip(m.predict(F_cal)), clip(m.predict(F_test))
    lo_m = fit_lgbm(F_fit, w_fit.y, quantile=cfg.alpha / 2, seed=cfg.seed,
                    n_estimators=cfg.n_estimators)
    hi_m = fit_lgbm(F_fit, w_fit.y, quantile=1 - cfg.alpha / 2, seed=cfg.seed,
                    n_estimators=cfg.n_estimators)
    m = fit_xgb(F_fit, w_fit.y, seed=cfg.seed, n_estimators=cfg.n_estimators)
    preds_cal["XGBoost"], preds_test["XGBoost"] = clip(m.predict(F_cal)), clip(m.predict(F_test))
    log(f"[{name}] trees done ({time.time() - t0:.0f}s)")

    for kind, label in (("cnn", "1D-CNN"), ("lstm", "LSTM")):
        net = train_nn(kind, w_fit.X, w_fit.y, cfg.cap, epochs=cfg.epochs, seed=cfg.seed)
        preds_cal[label] = predict_nn(net, w_cal.X, cfg.cap)
        preds_test[label] = predict_nn(net, w_test.X, cfg.cap)
        log(f"[{name}] {label} done ({time.time() - t0:.0f}s)")

    s = slice(None, None, cfg.survival_stride)
    wb = WeibullRUL().fit(F_fit[s], w_fit.y_true[s])
    preds_cal["Weibull AFT"], preds_test["Weibull AFT"] = (wb.predict(F_cal, cfg.cap),
                                                           wb.predict(F_test, cfg.cap))

    y_test, y_test_true = w_test.y, w_test.y_true
    models = {}
    for k, p in preds_test.items():
        lo_ci, hi_ci = bootstrap_ci(y_test, p, rmse, n_boot=cfg.n_boot, seed=cfg.seed)
        models[k] = {"rmse": rmse(y_test, p), "rmse_ci95": [lo_ci, hi_ci], "mae": mae(y_test, p),
                     "nasa_score": nasa_score(y_test, p), "rmse_cal": rmse(w_cal.y, preds_cal[k])}

    # --- conformal intervals (target = capped RUL, alpha = cfg.alpha)
    intervals, bounds = {}, {}
    for k in preds_test:
        q = cf.split_conformal(preds_cal[k], w_cal.y, cfg.alpha)
        lo, hi = np.clip(preds_test[k] - q, 0, None), preds_test[k] + q
        bounds[f"SC-{k}"] = (lo, hi)
        intervals[f"SC-{k}"] = {"q": q, "coverage": cf.coverage(y_test, lo, hi),
                                "width": cf.mean_width(lo, hi)}
    cal_lo, cal_hi = lo_m.predict(F_cal), hi_m.predict(F_cal)
    raw_lo, raw_hi = lo_m.predict(F_test), hi_m.predict(F_test)
    intervals["QR-LightGBM (uncalibrated)"] = {
        "q": 0.0, "coverage": cf.coverage(y_test, raw_lo, raw_hi),
        "width": cf.mean_width(raw_lo, raw_hi)}
    q = cf.cqr(cal_lo, cal_hi, w_cal.y, cfg.alpha)
    lo, hi = np.clip(raw_lo - q, 0, None), raw_hi + q
    bounds["CQR-LightGBM"] = (lo, hi)
    intervals["CQR-LightGBM"] = {"q": q, "coverage": cf.coverage(y_test, lo, hi),
                                 "width": cf.mean_width(lo, hi)}

    # --- decision layer (true, uncapped RUL decides failure)
    cm = cfg.cost
    margins = np.arange(0, 101, 1.0)
    point_cal = preds_cal["LightGBM"]
    best_m, cal_costs = tune_margin(point_cal, w_cal.y_true, margins, cm)
    point = preds_test["LightGBM"]
    q_sc = intervals["SC-LightGBM"]["q"]
    q_cqr = intervals["CQR-LightGBM"]["q"]
    policies = {
        "point estimate": point,
        "split-conformal lower bound": point - q_sc,
        "CQR lower bound": raw_lo - q_cqr,
        f"cost-tuned margin (m={best_m:.0f})": point - best_m,
    }
    decision = {k: summarise(y_test_true, v, cm) for k, v in policies.items()}
    test_costs = [float(summarise(y_test_true, point - mm, cm)["mean_cost"]) for mm in margins]

    out = {
        "n_fit_engines": int(len(fit_u)), "n_cal_engines": int(len(cal_u)),
        "n_test_engines": int(len(y_test)), "n_fit_windows": int(len(w_fit.y)),
        "n_cal_windows": int(len(w_cal.y)), "sensors": norm.sensors_, "models": models,
        "intervals": intervals, "decision": decision, "tuned_margin": best_m,
        "runtime_s": round(time.time() - t0, 1),
    }
    arrays = {"y_test": y_test, "y_test_true": y_test_true, "preds_test": preds_test,
              "bounds": bounds, "margins": margins, "cal_costs": cal_costs,
              "test_costs": np.array(test_costs)}
    log(f"[{name}] finished in {out['runtime_s']}s")
    return {"metrics": out, "arrays": arrays}


def run_all(root: str | Path, out_dir: str | Path, cfg: Config, log=print) -> dict:
    from .plots import make_all

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {name: run_subset(root, name, cfg, log=log) for name in cfg.subsets}
    cfg_d = asdict(cfg)
    cfg_d["subsets"] = list(cfg.subsets)
    payload = {"config": cfg_d, "subsets": {k: v["metrics"] for k, v in results.items()}}
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    make_all({k: v["arrays"] for k, v in results.items()}, payload, out_dir)
    return payload
