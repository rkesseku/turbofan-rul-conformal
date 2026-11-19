"""Command-line entry point: ``turbofan run-all`` and ``turbofan smoke``."""

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
    sm = sub.add_parser("smoke", help="tiny run on synthetic data")
    sm.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    if args.cmd == "run-all":
        cfg = Config(subsets=tuple(args.subsets), epochs=args.epochs, seeds=tuple(args.seeds),
                     alpha=args.alpha)
        run_all(args.data, args.out, cfg, cache_dir=args.cache)
    else:
        from .synthetic import write_subset

        with tempfile.TemporaryDirectory() as tmp:
            write_subset(tmp, "FD001")
            cfg = Config(subsets=("FD001",), epochs=2, n_estimators=50, n_boot=100,
                         seeds=(0, 1), nn_lrs=(1e-3,), nn_widths=(16,))
            run_all(tmp, args.out or tmp, cfg)


if __name__ == "__main__":
    main()
