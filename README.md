# turbofan-rul-conformal

[![CI](https://github.com/rkesseku/turbofan-rul-conformal/actions/workflows/ci.yml/badge.svg)](https://github.com/rkesseku/turbofan-rul-conformal/actions/workflows/ci.yml)

**Uncertainty-aware predictive maintenance on NASA C-MAPSS.** The project predicts how many cycles a turbofan engine has left (its remaining useful life, RUL), wraps every prediction in a conformal interval with a finite-sample coverage guarantee, and turns predictions into maintenance decisions under an explicit cost model.

## Why this exists

Most RUL projects stop at an RMSE table. A maintenance planner needs two more things:

1. **How sure is the model?** A point estimate of "40 cycles left" is unusable without a range.
2. **When should I act?** Acting at the point estimate fails the engine roughly half the time, because half of all unbiased predictions are late.

This repo answers both questions on the standard benchmark, using the official test sets, and keeps the evaluation honest. Engines are split at the engine level, calibration engines are never used for training, every test engine contributes exactly one prediction, and each RMSE comes with a bootstrap confidence interval.

## Data

NASA's *Turbofan Engine Degradation Simulation Data Set* (C-MAPSS; Saxena, Goebel, Simon and Eklund, 2008) contains four subsets:

| Subset | Train engines | Test engines | Operating conditions | Fault modes |
|---|---|---|---|---|
| FD001 | 100 | 100 | 1 | 1 |
| FD002 | 260 | 259 | 6 | 1 |
| FD003 | 100 | 100 | 1 | 2 |
| FD004 | 249 | 248 | 6 | 2 |

Training engines run to failure. Test engines stop at some earlier point, and `RUL_FD00x.txt` gives their true remaining life. The data comes from the NASA Prognostics Center of Excellence data repository. It is not redistributed here: download it and unzip `CMAPSSData.zip` into `data/raw/cmapss/`.

## Method

- **Targets.** RUL is capped at 125 cycles. This is the usual piecewise-linear convention, because early-life degradation is not observable. Test RMSE is computed against the capped truth. Maintenance costs use the *uncapped* truth.
- **Split.** Training engines are split 75/25 at the engine level into *fit* and *calibration* sets (seed 0). No engine appears in both.
- **Preprocessing.**
  - The three operating settings are clustered with k-means (6 clusters for FD002/FD004, 1 otherwise), and every sensor is standardised within its cluster.
  - Sensors are dropped if they are near-constant or have |Spearman ρ| < 0.05 with RUL on the fit engines.
  - Inputs are 30-cycle sliding windows plus a scaled cycle counter. Test engines shorter than 30 cycles are front-padded.
- **Models.** All are trained on the fit engines only:
  - LightGBM and XGBoost on per-channel window summaries (last, mean, std, min, max, slope, last-5 mean).
  - A 1D-CNN and a 2-layer LSTM in PyTorch on raw windows (CPU, early stopping, 40 epochs at most).
  - A Weibull accelerated-failure-time survival baseline (`lifelines`) on PCA-reduced window features. Each (engine, cycle) snapshot is a subject whose duration is its remaining life. The predicted median is the point estimate.
- **Uncertainty** (target 90% coverage, α = 0.1):
  - *Split conformal* on every model: the interval is prediction ± q, where q is the ⌈(n+1)(1−α)⌉-th smallest absolute residual on the calibration windows.
  - *Conformalized quantile regression* (CQR; Romano, Patterson and Candès, 2019) on LightGBM's 5% and 95% quantile models.
  - The raw quantile interval is also reported, to show what calibration fixes.
- **Decision layer.** At inspection, maintenance is scheduled `trigger` cycles ahead.
  - If the engine fails first (trigger ≥ true RUL), the cost is 10.
  - Otherwise the cost is 1 (a planned visit) plus 0.01 per cycle of useful life given up.
  - Four policies are compared on the test engines, all using LightGBM: trust the point estimate; use the split-conformal lower bound; use the CQR lower bound; use the point estimate minus a margin *m* tuned on the calibration engines.

## Results

All numbers come from one run of `turbofan run-all` (seed 0) and are stored in [`docs/results/metrics.json`](docs/results/metrics.json).

**Point accuracy: test RMSE in cycles (95% bootstrap CI over test engines)**

| Model | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| LightGBM | **12.04** [9.62, 14.29] | **12.94** [11.47, 14.34] | **13.52** [10.88, 15.87] | **14.11** [12.17, 16.06] |
| XGBoost | 12.11 [9.55, 14.42] | 12.96 [11.57, 14.39] | 13.53 [11.12, 15.88] | 14.36 [12.42, 16.38] |
| 1D-CNN | 16.27 [13.58, 18.71] | 18.60 [16.68, 20.51] | 19.16 [15.91, 22.22] | 19.22 [17.14, 21.24] |
| LSTM | 15.84 [13.00, 18.50] | 18.28 [16.26, 20.16] | 18.07 [14.91, 21.10] | 17.35 [14.88, 19.68] |
| Weibull AFT | 17.03 [13.66, 20.11] | 16.22 [14.02, 18.23] | 19.35 [15.33, 23.04] | 23.28 [21.00, 25.34] |

NASA asymmetric score for LightGBM: 225 / 737 / 358 / 972 on FD001–FD004. With the same features and no deep-learning tuning, gradient boosting beat both neural networks on every subset. The neural networks here are untuned baselines (one architecture, one seed, CPU-sized), so this is not a claim about what tuned sequence models can reach.

**Uncertainty: empirical coverage of 90% intervals on the test engines (mean width in cycles)**

| Interval | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| LightGBM 5–95% quantiles, uncalibrated | 0.710 (32.6) | 0.795 (34.3) | 0.650 (27.9) | 0.690 (32.3) |
| CQR on LightGBM | 0.940 (45.0) | 0.907 (39.5) | 0.930 (47.2) | 0.887 (43.3) |
| Split conformal, LightGBM | 0.910 (42.4) | 0.915 (46.9) | 0.890 (45.8) | 0.923 (48.4) |
| Split conformal, LSTM | 0.870 (49.8) | 0.915 (62.9) | 0.840 (54.6) | 0.895 (56.3) |

The raw quantile model under-covers badly (65–80% where 90% was asked for). Both conformal methods bring LightGBM to 0.89–0.94 on every subset. CQR gives narrower intervals than split conformal on FD002 and FD004, the subsets with six operating conditions (on FD004 its coverage of 0.887 sits just below target).

**Decisions: mean cost per test engine (failure rate)**

| Policy (LightGBM) | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| Act at the point estimate | 6.00 (55.0%) | 4.57 (38.2%) | 6.18 (57.0%) | 5.60 (49.6%) |
| Act at the split-conformal lower bound | 1.57 (4.0%) | 1.73 (4.6%) | 1.94 (8.0%) | 1.68 (4.0%) |
| Act at the CQR lower bound | 1.48 (3.0%) | 1.66 (4.2%) | 1.65 (5.0%) | 1.90 (6.9%) |
| Margin tuned on calibration engines | **1.34** (0.0%, m=39) | **1.46** (0.4%, m=41) | **1.58** (3.0%, m=35) | **1.49** (0.4%, m=43) |

Trusting the point estimate costs 2.6–4.5 times as much as the uncertainty-aware policies. A conformal lower bound needs no cost model and already removes most failures. When the cost ratio is known, tuning the margin on held-out engines does best on all four subsets, at the price of giving up more useful life (31–46 cycles per engine on average, against 20–32 for the lower bounds).

![Coverage and width](docs/results/coverage_width.png)
![Cost vs margin](docs/results/cost_vs_margin.png)

A short summary is in [`docs/brief/brief.pdf`](docs/brief/brief.pdf).

## Limitations

- **Single seed, single split.** The confidence intervals cover test-engine sampling, not training randomness.
- **Coverage is approximate.** Calibration windows from one engine are correlated, and test engines contribute one truncated snapshot each, so the exchangeability behind conformal guarantees holds only approximately. Empirical coverage ranged from 0.84 to 0.94.
- **Marginal coverage only.** Coverage is not guaranteed within RUL ranges or per engine.
- **Capped truth.** Coverage and RMSE use the capped (125-cycle) truth.
- **Illustrative costs.** The cost model (failure = 10× a planned visit, 0.01 per wasted cycle) is illustrative. Real fleets would plug in their own numbers.
- **Simulated data.** C-MAPSS is simulated. Results do not transfer to real engines without re-validation.

## How to run

```bash
pip install -e ".[dev]"          # Python 3.10-3.12
pytest                           # synthetic data only, about 15 s
ruff check src tests
turbofan smoke                   # tiny end-to-end run on synthetic engines
turbofan run-all --data data/raw/cmapss --out docs/results   # full run (CPU; about 2.7 h on a shared 14-core machine)
python docs/brief/make_figures.py
quarto render docs/brief/brief.qmd --to typst
```

Package layout: `data.py` (loading and RUL targets), `preprocess.py` (normalisation, windows, features), `models.py`, `conformal.py`, `metrics.py`, `decision.py`, `pipeline.py`, `plots.py`, `synthetic.py`, `cli.py`.

## References

- A. Saxena, K. Goebel, D. Simon, N. Eklund. *Damage propagation modeling for aircraft engine run-to-failure simulation.* International Conference on Prognostics and Health Management, 2008.
- Y. Romano, E. Patterson, E. Candès. *Conformalized quantile regression.* NeurIPS, 2019.

## License

MIT. See [LICENSE](LICENSE).
