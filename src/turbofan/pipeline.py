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
from .models import (
    WeibullRUL,
    best_rounds,
    fit_lgbm,
    fit_xgb,
    group_holdout,
    predict_nn,
    train_nn,
)
from .preprocess import ConditionNormalizer, make_windows, window_features


@dataclass
class Config:
    cap: float = 125.0
    window: int = 30
    cal_frac: float = 0.25
    val_frac: float = 0.15  # share of fit engines held out for early stopping / tuning
    alpha: float = 0.10
    split_seed: int = 0  # fit/calibration engine split (fixed across training seeds)
    seeds: tuple[int, ...] = (0, 1, 2)  # training seeds (model init + early-stopping engines)
    epochs: int = 40
    patience: int = 6
    n_estimators: int = 2000  # upper bound; rounds are chosen by early stopping
    tree_patience: int = 100
    nn_lrs: tuple[float, ...] = (1e-3, 3e-3)
    nn_widths: tuple[int, ...] = (32, 64)  # 64 = original width
    survival_stride: int = 5
    n_boot: int = 2000
    subsets: tuple[str, ...] = SUBSETS
    cost: CostModel = field(default_factory=CostModel)


NETS = (("cnn", "1D-CNN"), ("lstm", "LSTM"))


def _mean_sd(vals: list[float]) -> tuple[float, float]:
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), (float(a.std(ddof=1)) if len(a) > 1 else 0.0)


def _evaluate_seed(cfg: Config, w_cal, w_test, preds_cal: dict, preds_test: dict,
                   q_cal: tuple, q_test: tuple) -> tuple[dict, dict, dict, dict]:
    """Point metrics, conformal intervals and maintenance policies for one training seed."""
    y_test, y_test_true = w_test.y, w_test.y_true
    models = {k: {"rmse": rmse(y_test, p), "mae": mae(y_test, p),
                  "nasa_score": nasa_score(y_test, p), "rmse_cal": rmse(w_cal.y, preds_cal[k])}
              for k, p in preds_test.items()}
    intervals, bounds = {}, {}
    for k in preds_test:
        q = cf.split_conformal(preds_cal[k], w_cal.y, cfg.alpha)
        lo, hi = np.clip(preds_test[k] - q, 0, None), preds_test[k] + q
        bounds[f"SC-{k}"] = (lo, hi)
        intervals[f"SC-{k}"] = {"q": q, "coverage": cf.coverage(y_test, lo, hi),
                                "width": cf.mean_width(lo, hi)}
    (cal_lo, cal_hi), (raw_lo, raw_hi) = q_cal, q_test
    intervals["QR-LightGBM (uncalibrated)"] = {
        "q": 0.0, "coverage": cf.coverage(y_test, raw_lo, raw_hi),
        "width": cf.mean_width(raw_lo, raw_hi)}
    q = cf.cqr(cal_lo, cal_hi, w_cal.y, cfg.alpha)
    lo, hi = np.clip(raw_lo - q, 0, None), raw_hi + q
    bounds["CQR-LightGBM"] = (lo, hi)
    intervals["CQR-LightGBM"] = {"q": q, "coverage": cf.coverage(y_test, lo, hi),
                                 "width": cf.mean_width(lo, hi)}

    # decision layer (true, uncapped RUL decides failure); margin tuned on calibration engines
    cm = cfg.cost
    margins = np.arange(0, 101, 1.0)
    best_m, cal_costs = tune_margin(preds_cal["LightGBM"], w_cal.y_true, margins, cm)
    point = preds_test["LightGBM"]
    policies = {
        "point estimate": point,
        "split-conformal lower bound": point - intervals["SC-LightGBM"]["q"],
        "CQR lower bound": raw_lo - intervals["CQR-LightGBM"]["q"],
        "cost-tuned margin": point - best_m,
    }
    decision = {k: dict(summarise(y_test_true, v, cm)) for k, v in policies.items()}
    decision["cost-tuned margin"]["margin"] = float(best_m)
    test_costs = [float(summarise(y_test_true, point - mm, cm)["mean_cost"]) for mm in margins]
    arrays = {"y_test": y_test, "y_test_true": y_test_true, "preds_test": preds_test,
              "bounds": bounds, "margins": margins, "cal_costs": cal_costs,
              "test_costs": np.array(test_costs), "tuned_margin": float(best_m)}
    return models, intervals, decision, arrays


def _aggregate(per_seed: list[dict]) -> dict:
    """{name: {metric: value}} per seed -> {name: {metric_mean, metric_sd, metric_seeds}}."""
    out = {}
    for name in per_seed[0]:
        rec = {}
        for metric in per_seed[0][name]:
            vals = [float(s[name][metric]) for s in per_seed]
            mu, sd = _mean_sd(vals)
            rec[f"{metric}_mean"], rec[f"{metric}_sd"], rec[f"{metric}_seeds"] = mu, sd, vals
        out[name] = rec
    return out


def run_subset(root: str | Path, name: str, cfg: Config, log=print) -> dict:
    t0 = time.time()
    sub = load_subset(root, name, cap=cfg.cap)
    fit_u, cal_u = split_units(sub.train["unit"].to_numpy(), cfg.cal_frac, cfg.split_seed)
    fit_df = sub.train[sub.train["unit"].isin(fit_u)]
    cal_df = sub.train[sub.train["unit"].isin(cal_u)]

    norm = ConditionNormalizer(N_CONDITIONS.get(name, 1)).fit(fit_df)
    ch = norm.channels
    w_fit = make_windows(norm.transform(fit_df), ch, cfg.window)
    w_cal = make_windows(norm.transform(cal_df), ch, cfg.window)
    w_test = make_windows(norm.transform(sub.test), ch, cfg.window, last_only=True)
    F_fit, F_cal, F_test = (window_features(w.X) for w in (w_fit, w_cal, w_test))
    # calibration engines never enter training, early stopping or tuning
    assert not set(w_fit.unit.tolist()) & set(w_cal.unit.tolist())
    log(f"[{name}] {len(fit_u)} fit / {len(cal_u)} cal / {len(w_test.y)} test engines; "
        f"{len(w_fit.y)} fit windows; sensors={norm.sensors_}")

    def clip(p: np.ndarray) -> np.ndarray:
        return np.clip(p, 0.0, cfg.cap)

    # Weibull AFT: no hyperparameters to select, fitted once on all fit engines.
    s = slice(None, None, cfg.survival_stride)
    wb = WeibullRUL().fit(F_fit[s], w_fit.y_true[s])
    wb_cal, wb_test = wb.predict(F_cal, cfg.cap), wb.predict(F_test, cfg.cap)

    nn_choice: dict[str, dict] = {}
    tuning: dict[str, list] = {}
    per_seed_models, per_seed_int, per_seed_dec, selection = [], [], [], []
    arrays0: dict = {}
    for si, seed in enumerate(cfg.seeds):
        tr, va = group_holdout(w_fit.unit, cfg.val_frac, seed)
        sel: dict = {"seed": seed, "n_train_engines": int(len(np.unique(w_fit.unit[tr]))),
                     "n_es_engines": int(len(np.unique(w_fit.unit[va])))}
        preds_cal: dict[str, np.ndarray] = {}
        preds_test: dict[str, np.ndarray] = {}
        Ftr, ytr, Fva, yva = F_fit[tr], w_fit.y[tr], F_fit[va], w_fit.y[va]

        m = fit_lgbm(Ftr, ytr, Fva, yva, seed=seed, n_estimators=cfg.n_estimators,
                     patience=cfg.tree_patience)
        preds_cal["LightGBM"] = clip(m.predict(F_cal))
        preds_test["LightGBM"] = clip(m.predict(F_test))
        sel["LightGBM_rounds"] = best_rounds(m)
        qm = {}
        for tag, qq in (("lo", cfg.alpha / 2), ("hi", 1 - cfg.alpha / 2)):
            qm[tag] = fit_lgbm(Ftr, ytr, Fva, yva, quantile=qq, seed=seed,
                               n_estimators=cfg.n_estimators, patience=cfg.tree_patience)
            sel[f"LightGBM_q{tag}_rounds"] = best_rounds(qm[tag])
        m = fit_xgb(Ftr, ytr, Fva, yva, seed=seed, n_estimators=cfg.n_estimators,
                    patience=cfg.tree_patience)
        preds_cal["XGBoost"] = clip(m.predict(F_cal))
        preds_test["XGBoost"] = clip(m.predict(F_test))
        sel["XGBoost_rounds"] = best_rounds(m)
        log(f"[{name}] seed {seed}: trees done ({time.time() - t0:.0f}s) {sel}")

        Xtr, Xva = w_fit.X[tr], w_fit.X[va]
        for kind, label in NETS:
            if si == 0:  # small tuning grid, scored on the held-out fit engines of the first seed
                best_net, grid = None, []
                for lr in cfg.nn_lrs:
                    for width in cfg.nn_widths:
                        t1 = time.time()
                        net, rec = train_nn(kind, Xtr, ytr, Xva, yva, cfg.cap,
                                            epochs=cfg.epochs, seed=seed, lr=lr, width=width,
                                            patience=cfg.patience)
                        rec.update(lr=lr, width=width, seconds=round(time.time() - t1, 1))
                        grid.append(rec)
                        log(f"[{name}] {label} tune {rec}")
                        if best_net is None or rec["val_rmse"] < nn_choice[label]["val_rmse"]:
                            best_net, nn_choice[label] = net, dict(rec)
                tuning[label] = grid
                net = best_net
                sel[f"{label}_epochs"] = nn_choice[label]["best_epoch"]
            else:
                c = nn_choice[label]
                net, rec = train_nn(kind, Xtr, ytr, Xva, yva, cfg.cap, epochs=cfg.epochs,
                                    seed=seed, lr=c["lr"], width=c["width"],
                                    patience=cfg.patience)
                sel[f"{label}_epochs"] = rec["best_epoch"]
            preds_cal[label] = predict_nn(net, w_cal.X, cfg.cap)
            preds_test[label] = predict_nn(net, w_test.X, cfg.cap)
            log(f"[{name}] seed {seed}: {label} done ({time.time() - t0:.0f}s)")

        preds_cal["Weibull AFT"], preds_test["Weibull AFT"] = wb_cal, wb_test
        q_cal = (qm["lo"].predict(F_cal), qm["hi"].predict(F_cal))
        q_test = (qm["lo"].predict(F_test), qm["hi"].predict(F_test))
        mod, intv, dec, arr = _evaluate_seed(cfg, w_cal, w_test, preds_cal, preds_test,
                                             q_cal, q_test)
        per_seed_models.append(mod)
        per_seed_int.append(intv)
        per_seed_dec.append(dec)
        selection.append(sel)
        if si == 0:
            arrays0 = arr

    models = _aggregate(per_seed_models)
    for k, p in arrays0["preds_test"].items():  # bootstrap CI over test engines, first seed
        models[k]["rmse_ci95_seed0"] = list(
            bootstrap_ci(w_test.y, p, rmse, n_boot=cfg.n_boot, seed=cfg.split_seed))
    out = {
        "n_fit_engines": int(len(fit_u)), "n_cal_engines": int(len(cal_u)),
        "n_test_engines": int(len(w_test.y)), "n_fit_windows": int(len(w_fit.y)),
        "n_cal_windows": int(len(w_cal.y)), "sensors": norm.sensors_,
        "seeds": list(cfg.seeds), "models": models,
        "intervals": _aggregate(per_seed_int), "decision": _aggregate(per_seed_dec),
        "nn_tuning": tuning, "nn_choice": nn_choice, "selection": selection,
        "runtime_s": round(time.time() - t0, 1),
    }
    log(f"[{name}] finished in {out['runtime_s']}s")
    return {"metrics": out, "arrays": arrays0}


def run_all(root: str | Path, out_dir: str | Path, cfg: Config, log=print,
            cache_dir: str | Path | None = None) -> dict:
    """Run every subset; with ``cache_dir``, finished subsets are pickled and reused on restart."""
    import pickle

    from .plots import make_all

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in cfg.subsets:
        path = Path(cache_dir) / f"{name}.pkl" if cache_dir else None
        if path is not None and path.exists():
            results[name] = pickle.loads(path.read_bytes())
            log(f"[{name}] loaded from cache {path}")
            continue
        results[name] = run_subset(root, name, cfg, log=log)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(pickle.dumps(results[name]))
    cfg_d = asdict(cfg)
    for k in ("subsets", "seeds", "nn_lrs", "nn_widths"):
        cfg_d[k] = list(cfg_d[k])
    payload = {"config": cfg_d, "subsets": {k: v["metrics"] for k, v in results.items()}}
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    make_all({k: v["arrays"] for k, v in results.items()}, payload, out_dir)
    return payload
