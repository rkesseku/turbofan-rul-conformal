import json

import numpy as np
import pytest

from turbofan import conformal as cf
from turbofan.data import load_subset, split_units
from turbofan.decision import CostModel, policy_cost, summarise, tune_margin
from turbofan.metrics import bootstrap_ci, nasa_score, rmse
from turbofan.models import group_holdout
from turbofan.preprocess import ConditionNormalizer, make_windows, window_features
from turbofan.synthetic import write_subset


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("cmapss")
    write_subset(root, "FD001", n_train=20, n_test=8, seed=1)
    write_subset(root, "FD002", n_train=20, n_test=8, n_cond=6, seed=2)
    return root


def test_rul_targets(synth):
    sub = load_subset(synth, "FD001", cap=125)
    g = sub.train[sub.train.unit == 1]
    assert g["rul_true"].iloc[-1] == 0
    assert g["rul"].max() == 125
    rul = np.loadtxt(synth / "RUL_FD001.txt")
    last = sub.test.groupby("unit").tail(1)
    np.testing.assert_allclose(last["rul_true"].to_numpy(), rul)


def test_split_units_disjoint():
    fit, cal = split_units(np.arange(1, 101), 0.25, seed=0)
    assert len(cal) == 25 and len(fit) == 75
    assert not set(fit) & set(cal)


def test_group_holdout_keeps_engines_whole(synth):
    sub = load_subset(synth, "FD001")
    fit_u, cal_u = split_units(sub.train["unit"].to_numpy(), 0.25, seed=0)
    fit_df = sub.train[sub.train["unit"].isin(fit_u)]
    norm = ConditionNormalizer(1).fit(fit_df)
    w = make_windows(norm.transform(fit_df), norm.channels, window=20)
    for seed in range(5):
        tr, va = group_holdout(w.unit, 0.15, seed)
        assert tr.any() and va.any()
        assert not (tr & va).any() and (tr | va).all()
        # no engine contributes windows to both the training and early-stopping sides
        assert not set(w.unit[tr].tolist()) & set(w.unit[va].tolist())
        # the early-stopping engines come from the fit engines only, never calibration
        assert set(w.unit[va].tolist()) <= set(fit_u.tolist())
        assert not set(w.unit[va].tolist()) & set(cal_u.tolist())
    assert len(np.unique(w.unit[group_holdout(w.unit, 0.15, 0)[1]])) == round(0.15 * len(fit_u))


def test_normalizer_drops_constant_and_removes_condition_shift(synth):
    sub = load_subset(synth, "FD002")
    norm = ConditionNormalizer(6).fit(sub.train)
    assert "s1" not in norm.sensors_  # every 4th synthetic sensor is constant
    z = norm.transform(sub.train)
    # after per-condition standardisation, condition no longer explains sensor level
    assert abs(z[norm.sensors_[0]].mean()) < 0.1


def test_windows_and_padding(synth):
    sub = load_subset(synth, "FD001")
    norm = ConditionNormalizer(1).fit(sub.train)
    df = norm.transform(sub.test)
    w = make_windows(df, norm.channels, window=400, last_only=True)  # longer than any engine
    assert w.X.shape == (8, 400, len(norm.channels))
    w2 = make_windows(norm.transform(sub.train), norm.channels, window=30)
    assert len(w2.y) == len(sub.train) - 29 * 20
    F = window_features(w2.X)
    assert F.shape == (len(w2.y), 7 * len(norm.channels))
    assert np.isfinite(F).all()


def test_conformal_quantile_finite_sample():
    assert cf.conformal_quantile(np.arange(1, 10), 0.1) == 9.0  # ceil(10*0.9)=9th
    assert cf.conformal_quantile(np.arange(1, 5), 0.1) == float("inf")


def test_split_conformal_and_cqr_coverage():
    rng = np.random.default_rng(0)
    covs_sc, covs_cqr = [], []
    for _ in range(200):
        y_cal, y_te = rng.normal(size=500), rng.normal(size=500)
        q = cf.split_conformal(np.zeros(500), y_cal, 0.1)
        covs_sc.append(cf.coverage(y_te, -q, q))
        qq = cf.cqr(np.full(500, -0.5), np.full(500, 0.5), y_cal, 0.1)
        covs_cqr.append(cf.coverage(y_te, -0.5 - qq, 0.5 + qq))
    assert abs(np.mean(covs_sc) - 0.9) < 0.01
    assert abs(np.mean(covs_cqr) - 0.9) < 0.01
    assert cf.mean_width(np.zeros(3), np.ones(3)) == 1.0


def test_nasa_score_asymmetry():
    assert nasa_score([0], [0]) == 0.0
    late = nasa_score([50], [60])
    early = nasa_score([50], [40])
    assert late > early > 0
    assert late == pytest.approx(np.e - 1)


def test_bootstrap_ci_contains_estimate():
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    p = y + rng.normal(scale=2, size=200)
    lo, hi = bootstrap_ci(y, p, rmse, n_boot=300)
    assert lo < rmse(y, p) < hi


def test_decision_costs():
    cm = CostModel(preventive=1, failure=10, waste_per_cycle=0.01)
    c = policy_cost(np.array([50, 50]), np.array([40, 60]), cm)
    np.testing.assert_allclose(c, [1.1, 10.0])
    true = np.full(100, 50.0)
    pred = true + np.random.default_rng(0).normal(0, 10, 100)
    m, costs = tune_margin(pred, true, np.arange(0, 60.0), cm)
    assert m > 0  # biased-safe margin beats trusting the point estimate
    assert costs[int(m)] <= costs[0]
    s = summarise(true, pred - m, cm)
    assert 0 <= s["failure_rate"] <= 1


def test_end_to_end_synthetic(synth, tmp_path):
    from turbofan.pipeline import Config, run_all

    cfg = Config(subsets=("FD001",), epochs=2, n_estimators=40, n_boot=50, window=20,
                 seeds=(0, 1), nn_lrs=(1e-3, 3e-3), nn_widths=(8,))
    payload = run_all(synth, tmp_path, cfg, log=lambda *_: None)
    m = json.loads((tmp_path / "metrics.json").read_text())
    res = m["subsets"]["FD001"]
    assert set(res["models"]) == {"LightGBM", "XGBoost", "1D-CNN", "LSTM", "Weibull AFT"}
    assert 0 <= res["intervals"]["CQR-LightGBM"]["coverage_mean"] <= 1
    assert len(res["models"]["LSTM"]["rmse_seeds"]) == 2
    assert len(res["nn_tuning"]["1D-CNN"]) == 2  # one entry per grid point
    for sel in res["selection"]:  # early stopping picked a finite number of rounds / epochs
        assert 1 <= sel["LightGBM_rounds"] <= 40 and sel["LSTM_epochs"] >= 1
    assert (tmp_path / "coverage_width.png").exists()
    assert payload["subsets"]["FD001"]["n_test_engines"] == 8
