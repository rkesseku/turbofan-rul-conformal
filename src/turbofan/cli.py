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
    ra.add_argument("--seed", type=int, default=0)
    ra.add_argument("--alpha", type=float, default=0.10)
    sm = sub.add_parser("smoke", help="tiny run on synthetic data")
    sm.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    if args.cmd == "run-all":
        cfg = Config(subsets=tuple(args.subsets), epochs=args.epochs, seed=args.seed,
                     alpha=args.alpha)
        run_all(args.data, args.out, cfg)
    else:
        from .synthetic import write_subset

        with tempfile.TemporaryDirectory() as tmp:
            write_subset(tmp, "FD001")
            cfg = Config(subsets=("FD001",), epochs=2, n_estimators=50, n_boot=100)
            run_all(tmp, args.out or tmp, cfg)


if __name__ == "__main__":
    main()
