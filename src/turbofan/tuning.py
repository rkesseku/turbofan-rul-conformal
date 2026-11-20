"""Optuna search for the LightGBM point model, chosen on validation engines, tested once.

The search reads the *fit* engines only. Within them, the held-out engines of the first training
seed (``group_holdout(units, val_frac, seed=0)``, the same engines the networks' tuning grid is
scored on) are the validation engines; the rest are the search's training engines. Whole engines
sit on one side, so overlapping windows never leak across. Calibration and test engines are never
read during the search: the condition normaliser and sensor selection are fitted on the fit
engines, and calibration and test windows are only built after the search has finished.

Each trial trains LightGBM on the search's training engines, early-stops the number of trees on
the validation engines (patience as in ``run-all``) and is scored by validation RMSE in cycles.
The fixed recipe (``DEFAULTS``) is enqueued as trial 0, so the selected recipe is at least as
good as the default *on the validation engines*. Each subset is tuned separately because the
subsets differ in operating conditions, fault modes and size.

Both recipes are then run through the ``run-all`` protocol for the LightGBM point model
(training seeds 0, 1, 2; each seed holds out its own early-stopping engines; split conformal on
the calibration engines; decision policies from the point prediction) and scored once on the
test engines, with a paired bootstrap over test engines of tuned minus default.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import conformal as cf
from .data import N_CONDITIONS, SUBSETS, Subset, load_subset, split_units
from .decision import policy_cost, summarise, tune_margin
from .metrics import mae, nasa_score, rmse
from .models import LGBM_DEFAULTS, best_rounds, fit_lgbm, group_holdout
from .pipeline import Config
from .preprocess import ConditionNormalizer, make_windows, window_features

DEFAULTS = {k: LGBM_DEFAULTS[k] for k in ("num_leaves", "learning_rate", "min_child_samples",
                                          "subsample", "colsample_bytree", "reg_lambda")}
# Ranges searched; every range contains the default. Scale-type parameters are searched on a log
# scale; ``reg_lambda`` (default 0, which a log scale cannot hold) on a log1p scale. The number of
# trees is not sampled: early stopping on the validation engines chooses it (cap 2000).
SPACE = {
    "num_leaves": (7, 255),
    "learning_rate": (0.01, 0.2),
    "min_child_samples": (5, 300),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.3, 1.0),
    "reg_lambda": (0.0, 100.0),
}
LOG = {"num_leaves", "learning_rate", "min_child_samples"}
INTS = {"num_leaves", "min_child_samples"}
N_JOBS = 13  # threads for the tuning run (the only experiment on the machine)


def suggest_params(trial) -> dict:
    """Sample one recipe; ``reg_lambda`` is drawn as ``log1p_reg_lambda`` so that 0 is reachable."""
    out = {}
    for k, (lo, hi) in SPACE.items():
        if k == "reg_lambda":
            out[k] = float(np.expm1(trial.suggest_float("log1p_reg_lambda", np.log1p(lo),
                                                        np.log1p(hi))))
        elif k in INTS:
            out[k] = trial.suggest_int(k, lo, hi, log=k in LOG)
        else:
            out[k] = trial.suggest_float(k, lo, hi, log=k in LOG)
    return out


def to_trial_params(params: dict) -> dict:
    """Inverse of ``suggest_params`` naming, used to enqueue the default recipe."""
    out = {k: v for k, v in params.items() if k != "reg_lambda"}
    out["log1p_reg_lambda"] = float(np.log1p(params["reg_lambda"]))
    return out


def from_trial_params(tp: dict) -> dict:
    out = {k: v for k, v in tp.items() if k != "log1p_reg_lambda"}
    out["reg_lambda"] = float(np.expm1(tp["log1p_reg_lambda"]))
    return {**DEFAULTS, **out}


@dataclass
class TuneConfig:
    run: Config = field(default_factory=Config)
    n_trials: int = 100
    sampler_seed: int = 0
    search_seed: int = 0  # validation engines = early-stopping engines of this training seed
    n_jobs: int = N_JOBS
    timeout: float | None = None  # seconds per subset
    subsets: tuple[str, ...] = SUBSETS


# ----------------------------------------------------------------------------- data


def fit_side(sub: Subset, name: str, cfg: Config) -> dict:
    """Fit/calibration engine split, normaliser and fit-engine windows (no calibration or test
    rows are used here)."""
    fit_u, cal_u = split_units(sub.train["unit"].to_numpy(), cfg.cal_frac, cfg.split_seed)
    fit_df = sub.train[sub.train["unit"].isin(fit_u)]
    norm = ConditionNormalizer(N_CONDITIONS.get(name, 1)).fit(fit_df)
    w_fit = make_windows(norm.transform(fit_df), norm.channels, cfg.window)
    return {"fit_u": fit_u, "cal_u": cal_u, "norm": norm, "w_fit": w_fit,
            "F_fit": window_features(w_fit.X)}


def search_split(fs: dict, cfg: Config, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(train, validation) row masks over the fit windows, grouped by engine."""
    return group_holdout(fs["w_fit"].unit, cfg.val_frac, seed)


# ----------------------------------------------------------------------------- search


def tune_lgbm(F: np.ndarray, y: np.ndarray, units: np.ndarray, tr: np.ndarray, va: np.ndarray,
              n_trials: int = 100, sampler_seed: int = 0, model_seed: int = 0,
              n_estimators: int = 2000, patience: int = 100, n_jobs: int = N_JOBS,
              timeout: float | None = None, verbose: bool = True) -> dict:
    """TPE search minimising validation RMSE. Reads the fit-engine arrays passed in only."""
    import optuna

    if set(units[tr].tolist()) & set(units[va].tolist()):
        raise ValueError("training and validation engines overlap")
    Ftr, ytr, Fva, yva = F[tr], y[tr], F[va], y[va]

    def objective(trial) -> float:
        params = suggest_params(trial)
        t0 = time.perf_counter()
        m = fit_lgbm(Ftr, ytr, Fva, yva, seed=model_seed, n_estimators=n_estimators,
                     patience=patience, params=params, n_jobs=n_jobs)
        value = rmse(yva, m.predict(Fva))
        trial.set_user_attr("rounds", best_rounds(m))
        trial.set_user_attr("seconds", time.perf_counter() - t0)
        return value

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=sampler_seed))
    study.enqueue_trial(to_trial_params(DEFAULTS))

    def report(study, trial) -> None:
        if verbose and (trial.number % 10 == 0 or trial.number == n_trials - 1):
            print(f"  trial {trial.number:3d} val RMSE {trial.value:.3f} "
                  f"(best {study.best_value:.3f}, {trial.user_attrs['rounds']} trees, "
                  f"{trial.user_attrs['seconds']:.1f}s)", flush=True)

    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=[report])
    seconds = time.perf_counter() - t0
    trials = [{"number": t.number, "val_rmse": t.value, "params": from_trial_params(t.params),
               "rounds": t.user_attrs.get("rounds"), "seconds": t.user_attrs.get("seconds")}
              for t in study.trials if t.value is not None]
    best = study.best_trial
    return {
        "best_params": from_trial_params(best.params),
        "best_trial": best.number,
        "best_val_rmse": float(best.value),
        "best_rounds": best.user_attrs["rounds"],
        "default_val_rmse": float(trials[0]["val_rmse"]),
        "default_rounds": trials[0]["rounds"],
        "n_trials": len(trials),
        "n_train_engines": int(len(np.unique(units[tr]))),
        "n_val_engines": int(len(np.unique(units[va]))),
        "seconds": seconds,
        "trials": trials,
    }


def search_subset(sub: Subset, name: str, tcfg: TuneConfig, verbose: bool = True) -> dict:
    """Run the search for one subset from the fit engines only."""
    cfg = tcfg.run
    fs = fit_side(sub, name, cfg)
    tr, va = search_split(fs, cfg, tcfg.search_seed)
    return tune_lgbm(fs["F_fit"], fs["w_fit"].y, fs["w_fit"].unit, tr, va,
                     n_trials=tcfg.n_trials, sampler_seed=tcfg.sampler_seed,
                     model_seed=tcfg.search_seed, n_estimators=cfg.n_estimators,
                     patience=cfg.tree_patience, n_jobs=tcfg.n_jobs, timeout=tcfg.timeout,
                     verbose=verbose)


# ----------------------------------------------------------------------------- bootstrap


def cluster_bootstrap_rmse_diff(y: np.ndarray, p_ref: np.ndarray, p_new: np.ndarray,
                                groups: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """RMSE(new) - RMSE(ref) over windows, with a 95% bootstrap interval resampling engines."""
    g, inv = np.unique(groups, return_inverse=True)
    n = np.bincount(inv, minlength=len(g)).astype(float)
    s_ref = np.bincount(inv, weights=(p_ref - y) ** 2, minlength=len(g))
    s_new = np.bincount(inv, weights=(p_new - y) ** 2, minlength=len(g))
    idx = np.random.default_rng(seed).integers(0, len(g), size=(n_boot, len(g)))
    nn = n[idx].sum(1)
    d = np.sqrt(s_new[idx].sum(1) / nn) - np.sqrt(s_ref[idx].sum(1) / nn)
    est = float(np.sqrt(s_new.sum() / n.sum()) - np.sqrt(s_ref.sum() / n.sum()))
    return {"diff": est, "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]}


def _test_metrics(y, y_true, p, lo, hi, cm, margin):
    """Per-engine quantities whose (sum or mean) over engines gives each reported metric."""
    return {
        "sq_err": (p - y) ** 2,
        "nasa": np.array([nasa_score([a], [b]) for a, b in zip(y, p, strict=True)]),
        "covered": ((y >= lo) & (y <= hi)).astype(float),
        "width": hi - lo,
        "cost_sc_lower": policy_cost(y_true, lo, cm),
        "cost_margin": policy_cost(y_true, p - margin, cm),
    }


AGG = {"rmse": ("sq_err", "rmse"), "nasa_score": ("nasa", "sum"), "coverage": ("covered", "mean"),
       "width": ("width", "mean"), "cost_sc_lower": ("cost_sc_lower", "mean"),
       "cost_margin": ("cost_margin", "mean")}


def _agg(x: np.ndarray, how: str, axis: int = -1) -> np.ndarray:
    if how == "rmse":
        return np.sqrt(x.mean(axis))
    return x.sum(axis) if how == "sum" else x.mean(axis)


def paired_test_bootstrap(per_eng_ref: list[dict], per_eng_new: list[dict], n_boot: int = 2000,
                          seed: int = 0) -> dict:
    """Seed-mean metric difference (new - ref) with a 95% paired bootstrap over test engines.

    The same resampled engines are used for both recipes and every seed.
    """
    n = len(per_eng_ref[0]["sq_err"])
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    out = {}
    for metric, (key, how) in AGG.items():
        ref = np.stack([r[key] for r in per_eng_ref])  # (seeds, engines)
        new = np.stack([r[key] for r in per_eng_new])
        est = float(np.mean(_agg(new, how) - _agg(ref, how)))
        boots = np.mean(_agg(new[:, idx], how) - _agg(ref[:, idx], how), axis=0)
        out[metric] = {"diff": est, "ci95": [float(np.quantile(boots, 0.025)),
                                             float(np.quantile(boots, 0.975))]}
    return out


# ----------------------------------------------------------------------------- protocol


def evaluate_recipe(fs: dict, w_cal, w_test, F_cal, F_test, params: dict | None, cfg: Config,
                    n_jobs: int) -> tuple[list[dict], list[dict]]:
    """The ``run-all`` protocol for the LightGBM point model under one recipe."""
    clip = lambda p: np.clip(p, 0.0, cfg.cap)  # noqa: E731
    runs, per_eng = [], []
    margins = np.arange(0, 101, 1.0)
    for seed in cfg.seeds:
        tr, va = search_split(fs, cfg, seed)
        F, y = fs["F_fit"], fs["w_fit"].y
        m = fit_lgbm(F[tr], y[tr], F[va], y[va], seed=seed, n_estimators=cfg.n_estimators,
                     patience=cfg.tree_patience, params=params, n_jobs=n_jobs)
        p_cal, p_test = clip(m.predict(F_cal)), clip(m.predict(F_test))
        q = cf.split_conformal(p_cal, w_cal.y, cfg.alpha)
        lo, hi = np.clip(p_test - q, 0, None), p_test + q
        best_m, _ = tune_margin(p_cal, w_cal.y_true, margins, cfg.cost)
        e = _test_metrics(w_test.y, w_test.y_true, p_test, lo, hi, cfg.cost, best_m)
        runs.append({
            "seed": seed, "rounds": best_rounds(m), "es_rmse": rmse(y[va], m.predict(F[va])),
            "rmse": rmse(w_test.y, p_test), "mae": mae(w_test.y, p_test),
            "nasa_score": nasa_score(w_test.y, p_test), "q": q,
            "coverage": cf.coverage(w_test.y, lo, hi), "width": cf.mean_width(lo, hi),
            "cost_sc_lower": float(e["cost_sc_lower"].mean()),
            "cost_margin": float(e["cost_margin"].mean()), "margin": best_m,
            "failure_rate_sc_lower": summarise(w_test.y_true, lo, cfg.cost)["failure_rate"],
        })
        per_eng.append(e)
    return runs, per_eng


def _summary(runs: list[dict]) -> dict:
    out = {}
    for k in ("rmse", "mae", "nasa_score", "coverage", "width", "cost_sc_lower", "cost_margin",
              "failure_rate_sc_lower", "es_rmse"):
        v = np.array([r[k] for r in runs], dtype=float)
        out[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                  "seeds": v.tolist()}
    out["rounds"] = [r["rounds"] for r in runs]
    return out


def tune_subset(root: str | Path, name: str, tcfg: TuneConfig, log=print) -> dict:
    cfg = tcfg.run
    t0 = time.perf_counter()
    sub = load_subset(root, name, cap=cfg.cap)
    log(f"[{name}] searching ({tcfg.n_trials} trials, {tcfg.n_jobs} threads)")
    search = search_subset(sub, name, tcfg, verbose=True)
    log(f"[{name}] search done in {search['seconds']:.0f}s: val RMSE default "
        f"{search['default_val_rmse']:.3f} -> best {search['best_val_rmse']:.3f} "
        f"(trial {search['best_trial']})")

    # validation comparison of the two recipes on the search split (engine-cluster bootstrap)
    fs = fit_side(sub, name, cfg)
    tr, va = search_split(fs, cfg, tcfg.search_seed)
    F, y, u = fs["F_fit"], fs["w_fit"].y, fs["w_fit"].unit
    preds = {}
    for arm, params in (("default", None), ("tuned", search["best_params"])):
        m = fit_lgbm(F[tr], y[tr], F[va], y[va], seed=tcfg.search_seed,
                     n_estimators=cfg.n_estimators, patience=cfg.tree_patience, params=params,
                     n_jobs=tcfg.n_jobs)
        preds[arm] = m.predict(F[va])
    val_cmp = cluster_bootstrap_rmse_diff(y[va], preds["default"], preds["tuned"], u[va],
                                          n_boot=cfg.n_boot, seed=cfg.split_seed)

    # only now build calibration and test windows, with the normaliser fitted on fit engines
    cal_df = sub.train[sub.train["unit"].isin(fs["cal_u"])]
    norm = fs["norm"]
    w_cal = make_windows(norm.transform(cal_df), norm.channels, cfg.window)
    w_test = make_windows(norm.transform(sub.test), norm.channels, cfg.window, last_only=True)
    F_cal, F_test = window_features(w_cal.X), window_features(w_test.X)
    arms, per_eng = {}, {}
    for arm, params in (("default", None), ("tuned", search["best_params"])):
        runs, pe = evaluate_recipe(fs, w_cal, w_test, F_cal, F_test, params, cfg, tcfg.n_jobs)
        arms[arm] = {"summary": _summary(runs), "runs": runs}
        per_eng[arm] = pe
    test_cmp = paired_test_bootstrap(per_eng["default"], per_eng["tuned"], n_boot=cfg.n_boot,
                                     seed=cfg.split_seed)
    out = {
        "n_fit_engines": int(len(fs["fit_u"])), "n_cal_engines": int(len(fs["cal_u"])),
        "n_test_engines": int(len(w_test.y)),
        "val_engines": sorted(int(x) for x in np.unique(u[va])),
        "search": search, "validation_comparison": {
            "default_val_rmse": rmse(y[va], preds["default"]),
            "tuned_val_rmse": rmse(y[va], preds["tuned"]), "tuned_minus_default": val_cmp},
        "default_params": dict(DEFAULTS), "tuned_params": search["best_params"],
        "arms": arms, "test_comparison": test_cmp,
        "seconds": time.perf_counter() - t0,
    }
    s = arms["tuned"]["summary"]
    log(f"[{name}] test RMSE default {arms['default']['summary']['rmse']['mean']:.3f} "
        f"tuned {s['rmse']['mean']:.3f}; diff {test_cmp['rmse']['diff']:+.3f} "
        f"{test_cmp['rmse']['ci95']} ({out['seconds']:.0f}s)")
    return out


def run_tuning(root: str | Path, out_dir: str | Path, tcfg: TuneConfig | None = None,
               log=print) -> dict:
    """Tune LightGBM per subset on validation engines, then compare tuned vs default on test."""
    tcfg = tcfg or TuneConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    subsets = {name: tune_subset(root, name, tcfg, log=log) for name in tcfg.subsets}
    rc = asdict(tcfg.run)
    for k in ("subsets", "seeds", "nn_lrs", "nn_widths"):
        rc[k] = list(rc[k])
    result = {
        "protocol": {
            "model": "LightGBM point model (squared loss) on window summaries",
            "search_data": "fit engines only; validation engines = the 15% of fit engines held "
            f"out for early stopping by training seed {tcfg.search_seed}; calibration and test "
            "engines never read during the search",
            "objective": "validation RMSE (cycles, capped truth) of a model trained on the other "
            "fit engines, trees early-stopped on the validation engines (patience "
            f"{tcfg.run.tree_patience}, cap {tcfg.run.n_estimators})",
            "sampler": f"TPE (seed {tcfg.sampler_seed}); trial 0 = default recipe; no pruning",
            "per_subset": True,
            "final": "run-all protocol for the LightGBM point model: seeds "
            f"{list(tcfg.run.seeds)}, each with its own early-stopping engines; split "
            f"conformal (alpha {tcfg.run.alpha}) on the calibration engines; decisions from the "
            "point prediction; one evaluation on the official test engines",
            "bootstrap": f"{tcfg.run.n_boot} paired resamples of test engines (seed-mean "
            "difference, tuned - default); validation: resamples of validation engines",
            "n_jobs": tcfg.n_jobs,
            "total_seconds": time.perf_counter() - t0,
        },
        "config": {"n_trials": tcfg.n_trials, "sampler_seed": tcfg.sampler_seed,
                   "search_seed": tcfg.search_seed, "timeout": tcfg.timeout,
                   "space": {k: list(v) for k, v in SPACE.items()}, "run": rc},
        "subsets": subsets,
    }
    (out / "tuning.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "tuning.md").write_text(tuning_markdown(result), encoding="utf-8")
    return result


def merge_tuning(parts: list[str | Path], out_dir: str | Path) -> dict:
    """Combine ``tuning.json`` files from per-subset runs (e.g. run in parallel) into one."""
    results = [json.loads((Path(d) / "tuning.json").read_text(encoding="utf-8")) for d in parts]
    subsets: dict = {}
    for r in results:
        subsets.update(r["subsets"])
    merged = dict(results[0])
    merged["subsets"] = {k: subsets[k] for k in SUBSETS if k in subsets}
    secs = [r["protocol"]["total_seconds"] for r in results]
    merged["protocol"]["total_seconds"] = max(secs)
    merged["protocol"]["parallel_runs"] = {
        "n": len(results), "seconds_each": secs,
        "note": "subsets run as separate processes at the same time; total = wall clock"}
    merged["config"]["run"]["subsets"] = list(merged["subsets"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tuning.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
    (out / "tuning.md").write_text(tuning_markdown(merged), encoding="utf-8")
    return merged


# ----------------------------------------------------------------------------- report


def _ci(d: dict, nd: int = 2) -> str:
    lo, hi = d["ci95"]
    return f"{d['diff']:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def _fmt(v) -> str:
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def tuning_markdown(result: dict) -> str:
    """Human-readable summary of ``tuning.json``."""
    p = result["protocol"]
    lines = [
        "# Does tuning help? Optuna search for the LightGBM point model",
        "",
        f"Search: {p['sampler']}; objective = {p['objective']}. Data: {p['search_data']}. "
        f"Final comparison: {p['final']}. Bootstrap: {p['bootstrap']}. "
        f"Threads: {p['n_jobs']} per run"
        + (f", {p['parallel_runs']['n']} subsets run in parallel" if "parallel_runs" in p else "")
        + f". Total wall-clock time {p['total_seconds'] / 60:.1f} min.",
        "",
        "## Validation (search split, seed 0)",
        "",
        "| Subset | Trials | Search time (s) | Default val RMSE | Tuned val RMSE | Tuned - default "
        "[95% CI] | Best trial | Trees (default / tuned) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in result["subsets"].items():
        s, v = r["search"], r["validation_comparison"]
        lines.append(
            f"| {name} | {s['n_trials']} | {s['seconds']:.0f} | {v['default_val_rmse']:.2f} "
            f"| {v['tuned_val_rmse']:.2f} | {_ci(v['tuned_minus_default'])} | {s['best_trial']} "
            f"| {s['default_rounds']} / {s['best_rounds']} |")
    lines += ["", "## Tuned parameters", "",
              "| Subset | " + " | ".join(DEFAULTS) + " |", "|---|" + "---|" * len(DEFAULTS),
              "| default | " + " | ".join(_fmt(v) for v in DEFAULTS.values()) + " |"]
    for name, r in result["subsets"].items():
        lines.append(f"| {name} | "
                     + " | ".join(_fmt(r["tuned_params"][k]) for k in DEFAULTS) + " |")
    lines += [
        "", "## Test (mean over seeds)", "",
        "| Subset | Recipe | RMSE (± SD) | NASA score | SC coverage | SC width | Cost, SC lower "
        "bound | Cost, tuned margin | Trees |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in result["subsets"].items():
        for arm, a in r["arms"].items():
            s = a["summary"]
            lines.append(
                f"| {name} | {arm} | {s['rmse']['mean']:.2f} ± {s['rmse']['sd']:.2f} "
                f"| {s['nasa_score']['mean']:.0f} | {s['coverage']['mean']:.3f} "
                f"| {s['width']['mean']:.1f} | {s['cost_sc_lower']['mean']:.2f} "
                f"| {s['cost_margin']['mean']:.2f} | {', '.join(map(str, s['rounds']))} |")
    lines += [
        "", "Tuned minus default on test, 95% paired bootstrap interval over test engines:", "",
        "| Subset | RMSE | NASA score | SC coverage | SC width | Cost, SC lower bound "
        "| Cost, tuned margin |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in result["subsets"].items():
        c = r["test_comparison"]
        lines.append(f"| {name} | {_ci(c['rmse'])} | {_ci(c['nasa_score'], 0)} "
                     f"| {_ci(c['coverage'], 3)} | {_ci(c['width'], 1)} "
                     f"| {_ci(c['cost_sc_lower'])} | {_ci(c['cost_margin'])} |")
    lines.append("")
    return "\n".join(lines)
