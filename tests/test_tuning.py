import json

import numpy as np
import pytest

from turbofan.data import load_subset
from turbofan.pipeline import Config
from turbofan.synthetic import write_subset
from turbofan.tuning import (
    DEFAULTS,
    TuneConfig,
    cluster_bootstrap_rmse_diff,
    fit_side,
    search_split,
    search_subset,
)

pytest.importorskip("optuna")


def _tcfg(n_trials: int = 6) -> TuneConfig:
    run = Config(subsets=("FD001",), window=20, n_estimators=60, tree_patience=10, n_boot=50,
                 seeds=(0, 1))
    return TuneConfig(run=run, n_trials=n_trials, n_jobs=2, subsets=("FD001",))


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("cmapss_tune")
    write_subset(root, "FD001", n_train=24, n_test=8, seed=3)
    return root


def test_trial0_is_default_and_tuned_not_worse(synth):
    sub = load_subset(synth, "FD001")
    res = search_subset(sub, "FD001", _tcfg(), verbose=False)
    t0 = res["trials"][0]
    assert t0["number"] == 0
    for k, v in DEFAULTS.items():
        assert t0["params"][k] == pytest.approx(v)
    assert res["best_val_rmse"] <= res["default_val_rmse"]
    assert res["default_val_rmse"] == pytest.approx(t0["val_rmse"])


def test_search_ignores_calibration_and_test_rows(synth):
    cfg = _tcfg()
    sub = load_subset(synth, "FD001")
    clean = search_subset(sub, "FD001", cfg, verbose=False)
    fs = fit_side(sub, "FD001", cfg.run)
    sensors = [c for c in sub.train.columns if c.startswith("s")]
    rng = np.random.default_rng(0)
    cal_rows = sub.train["unit"].isin(fs["cal_u"])
    sub.train.loc[cal_rows, sensors] = rng.normal(1e4, 1e3, (int(cal_rows.sum()), len(sensors)))
    sub.train.loc[cal_rows, ["rul", "rul_true"]] = 0.0
    sub.test.loc[:, sensors] = rng.normal(-1e4, 1e3, (len(sub.test), len(sensors)))
    sub.test.loc[:, ["rul", "rul_true"]] = 999.0
    dirty = search_subset(sub, "FD001", cfg, verbose=False)
    assert [t["val_rmse"] for t in clean["trials"]] == [t["val_rmse"] for t in dirty["trials"]]
    assert [t["params"] for t in clean["trials"]] == [t["params"] for t in dirty["trials"]]
    assert clean["best_params"] == dirty["best_params"]


def test_validation_engines_disjoint(synth):
    cfg = _tcfg().run
    sub = load_subset(synth, "FD001")
    fs = fit_side(sub, "FD001", cfg)
    tr, va = search_split(fs, cfg, seed=0)
    units = fs["w_fit"].unit
    val, train = set(units[va].tolist()), set(units[tr].tolist())
    assert val and train
    assert not val & train
    assert not val & set(fs["cal_u"].tolist())
    # test engines are a separate file; the search never builds their windows, and the
    # validation engines are fit engines (training-file units), not calibration ones
    assert val <= set(fs["fit_u"].tolist())
    assert not set(fs["fit_u"].tolist()) & set(fs["cal_u"].tolist())


def test_cluster_bootstrap_zero_for_identical():
    y = np.arange(20.0)
    p = y + 1
    d = cluster_bootstrap_rmse_diff(y, p, p, np.repeat(np.arange(5), 4), n_boot=50)
    assert d["diff"] == 0 and d["ci95"] == [0.0, 0.0]


def test_cli_tune_end_to_end(synth, tmp_path):
    from turbofan.cli import main

    main(["tune", "--data", str(synth), "--out", str(tmp_path), "--subsets", "FD001",
          "--trials", "4", "--seeds", "0", "1", "--n-jobs", "2", "--max-rounds", "60",
          "--n-boot", "50", "--window", "20"])
    res = json.loads((tmp_path / "tuning.json").read_text())
    r = res["subsets"]["FD001"]
    assert r["search"]["n_trials"] == 4
    assert set(r["arms"]) == {"default", "tuned"}
    assert len(r["arms"]["tuned"]["runs"]) == 2
    for k in ("rmse", "nasa_score", "coverage", "width"):
        lo, hi = r["test_comparison"][k]["ci95"]
        assert lo <= hi
    v = r["validation_comparison"]
    assert v["tuned_val_rmse"] <= v["default_val_rmse"] + 1e-9
    assert (tmp_path / "tuning.md").read_text().startswith("# Does tuning help?")
