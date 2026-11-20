# Does tuning help? Optuna search for the LightGBM point model

Search: TPE (seed 0); trial 0 = default recipe; no pruning; objective = validation RMSE (cycles, capped truth) of a model trained on the other fit engines, trees early-stopped on the validation engines (patience 100, cap 2000). Data: fit engines only; validation engines = the 15% of fit engines held out for early stopping by training seed 0; calibration and test engines never read during the search. Final comparison: run-all protocol for the LightGBM point model: seeds [0, 1, 2], each with its own early-stopping engines; split conformal (alpha 0.1) on the calibration engines; decisions from the point prediction; one evaluation on the official test engines. Bootstrap: 2000 paired resamples of test engines (seed-mean difference, tuned - default); validation: resamples of validation engines. Threads: 3 per run, 4 subsets run in parallel. Total wall-clock time 26.7 min.

## Validation (search split, seed 0)

| Subset | Trials | Search time (s) | Default val RMSE | Tuned val RMSE | Tuned - default [95% CI] | Best trial | Trees (default / tuned) |
|---|---|---|---|---|---|---|---|
| FD001 | 100 | 562 | 15.82 | 15.01 | -0.81 [-1.38, -0.07] | 80 | 203 / 566 |
| FD002 | 100 | 858 | 14.90 | 14.66 | -0.25 [-0.45, -0.02] | 38 | 170 / 257 |
| FD003 | 100 | 744 | 14.68 | 14.09 | -0.58 [-1.29, +0.13] | 92 | 146 / 1124 |
| FD004 | 83 | 1505 | 11.57 | 11.29 | -0.28 [-0.59, +0.02] | 81 | 618 / 709 |

## Tuned parameters

| Subset | num_leaves | learning_rate | min_child_samples | subsample | colsample_bytree | reg_lambda |
|---|---|---|---|---|---|---|
| default | 31 | 0.03 | 40 | 0.8 | 0.8 | 0 |
| FD001 | 10 | 0.01518 | 23 | 0.9723 | 0.751 | 11.04 |
| FD002 | 13 | 0.02579 | 5 | 0.5777 | 0.8955 | 1.882 |
| FD003 | 7 | 0.01217 | 41 | 0.9571 | 0.905 | 8.854 |
| FD004 | 86 | 0.01034 | 33 | 0.9013 | 0.4307 | 9.707 |

## Test (mean over seeds)

| Subset | Recipe | RMSE (± SD) | NASA score | SC coverage | SC width | Cost, SC lower bound | Cost, tuned margin | Trees |
|---|---|---|---|---|---|---|---|---|
| FD001 | default | 12.33 ± 0.91 | 230 | 0.900 | 42.5 | 1.60 | 1.35 | 203, 227, 175 |
| FD001 | tuned | 12.45 ± 0.85 | 239 | 0.893 | 40.3 | 1.62 | 1.35 | 566, 1218, 490 |
| FD002 | default | 13.12 ± 0.06 | 736 | 0.920 | 46.5 | 1.72 | 1.46 | 170, 227, 265 |
| FD002 | tuned | 13.07 ± 0.11 | 712 | 0.919 | 46.0 | 1.72 | 1.45 | 257, 309, 450 |
| FD003 | default | 13.89 ± 0.70 | 373 | 0.903 | 47.6 | 1.83 | 1.40 | 146, 223, 304 |
| FD003 | tuned | 12.80 ± 0.51 | 282 | 0.893 | 44.5 | 1.87 | 1.38 | 1124, 731, 1229 |
| FD004 | default | 14.71 ± 0.29 | 1002 | 0.917 | 48.5 | 1.70 | 1.50 | 618, 169, 162 |
| FD004 | tuned | 14.38 ± 0.20 | 973 | 0.910 | 48.7 | 1.70 | 1.49 | 709, 554, 513 |

Tuned minus default on test, 95% paired bootstrap interval over test engines:

| Subset | RMSE | NASA score | SC coverage | SC width | Cost, SC lower bound | Cost, tuned margin |
|---|---|---|---|---|---|---|
| FD001 | +0.12 [-0.44, +0.62] | +10 [-9, +29] | -0.007 [-0.030, +0.017] | -2.1 [-2.2, -2.0] | +0.02 [-0.07, +0.14] | -0.00 [-0.00, +0.00] |
| FD002 | -0.05 [-0.37, +0.27] | -25 [-79, +24] | -0.001 [-0.012, +0.008] | -0.5 [-0.6, -0.5] | -0.00 [-0.05, +0.04] | -0.02 [-0.05, +0.02] |
| FD003 | -1.09 [-2.00, -0.12] | -91 [-192, -12] | -0.010 [-0.040, +0.020] | -3.1 [-3.2, -3.0] | +0.05 [-0.07, +0.17] | -0.02 [-0.17, +0.09] |
| FD004 | -0.33 [-0.65, -0.02] | -29 [-86, +34] | -0.007 [-0.022, +0.008] | +0.3 [+0.2, +0.3] | +0.00 [-0.06, +0.06] | -0.01 [-0.01, -0.00] |
