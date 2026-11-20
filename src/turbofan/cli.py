"""Command-line entry point: ``turbofan run-all``, ``turbofan tune`` and ``turbofan smoke``."""

from __future__ import annotations

import argparse
import tempfile

from .data import SUBSETS
from .pipeline import Config, run_all


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="turbofan")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ra = sub.add_parser("run-all", help="run every model on the C-MAPSS subsets")
    ra.add_argument("--data", default="data/raw/cmapss", help="folder with train_/test_/RUL_ files")
    ra.add_argument("--out", default="docs/results")
    ra.add_argument("--subsets", nargs="+", default=list(SUBSETS))
    ra.add_argument("--epochs", type=int, default=40)
    ra.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="training seeds (model init and early-stopping engines)")
    ra.add_argument("--alpha", type=float, default=0.10)
    ra.add_argument("--cache", default=None,
                    help="folder for per-subset results, reused if the run is restarted")
    tu = sub.add_parser("tune", help="Optuna search for LightGBM, then tuned vs default on test")
    tu.add_argument("--data", default="data/raw/cmapss", help="folder with train_/test_/RUL_ files")
    tu.add_argument("--out", default="docs/results")
    tu.add_argument("--subsets", nargs="+", default=list(SUBSETS))
    tu.add_argument("--trials", type=int, default=100, help="Optuna trials per subset")
    tu.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="training seeds for the final tuned-vs-default comparison")
    tu.add_argument("--n-jobs", type=int, default=13, help="LightGBM threads")
    tu.add_argument("--timeout", type=float, default=None, help="search seconds per subset")
    tu.add_argument("--max-rounds", type=int, default=2000, help="cap on boosting rounds")
    tu.add_argument("--n-boot", type=int, default=2000)
    tu.add_argument("--window", type=int, default=30)
    tu.add_argument("--merge", nargs="+", default=None,
                    help="only merge tuning.json files from these folders into --out")
    sm = sub.add_parser("smoke", help="tiny run on synthetic data")
    sm.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    if args.cmd == "run-all":
        cfg = Config(subsets=tuple(args.subsets), epochs=args.epochs, seeds=tuple(args.seeds),
                     alpha=args.alpha)
        run_all(args.data, args.out, cfg, cache_dir=args.cache)
    elif args.cmd == "tune":
        from .tuning import TuneConfig, merge_tuning, run_tuning

        if args.merge:
            merge_tuning(args.merge, args.out)
            return
        cfg = Config(subsets=tuple(args.subsets), seeds=tuple(args.seeds),
                     n_estimators=args.max_rounds, n_boot=args.n_boot, window=args.window)
        tcfg = TuneConfig(run=cfg, n_trials=args.trials, n_jobs=args.n_jobs,
                          timeout=args.timeout, subsets=tuple(args.subsets))
        run_tuning(args.data, args.out, tcfg)
    else:
        from .synthetic import write_subset

        with tempfile.TemporaryDirectory() as tmp:
            write_subset(tmp, "FD001")
            cfg = Config(subsets=("FD001",), epochs=2, n_estimators=50, n_boot=100,
                         seeds=(0, 1), nn_lrs=(1e-3,), nn_widths=(16,))
            run_all(tmp, args.out or tmp, cfg)


if __name__ == "__main__":
    main()
