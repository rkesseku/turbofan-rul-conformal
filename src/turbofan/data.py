"""Loading NASA C-MAPSS files and attaching remaining-useful-life (RUL) targets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

OS_COLS = ["os1", "os2", "os3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
COLS = ["unit", "cycle", *OS_COLS, *SENSOR_COLS]
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
# Number of operating conditions per subset (Saxena et al., 2008).
N_CONDITIONS = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}


@dataclass
class Subset:
    """One C-MAPSS subset: run-to-failure training engines and truncated test engines."""

    name: str
    train: pd.DataFrame
    test: pd.DataFrame


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a whitespace-separated C-MAPSS table (26 columns, no header)."""
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, : len(COLS)]
    df.columns = COLS
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df


def add_train_rul(df: pd.DataFrame, cap: float) -> pd.DataFrame:
    """Training engines run to failure, so true RUL = last cycle - current cycle."""
    df = df.copy()
    life = df.groupby("unit")["cycle"].transform("max")
    df["rul_true"] = (life - df["cycle"]).astype(float)
    df["rul"] = df["rul_true"].clip(upper=cap)
    return df


def add_test_rul(df: pd.DataFrame, rul_last: np.ndarray, cap: float) -> pd.DataFrame:
    """Test engines stop before failure; RUL_FDxxx.txt gives the true RUL at the last cycle."""
    df = df.copy()
    units = np.sort(df["unit"].unique())
    if len(units) != len(rul_last):
        raise ValueError(f"{len(units)} test units but {len(rul_last)} RUL values")
    last = dict(zip(units, np.asarray(rul_last, dtype=float), strict=True))
    maxc = df.groupby("unit")["cycle"].transform("max")
    df["rul_true"] = df["unit"].map(last) + (maxc - df["cycle"])
    df["rul"] = df["rul_true"].clip(upper=cap)
    return df


def load_subset(root: str | Path, name: str, cap: float = 125.0) -> Subset:
    """Load train/test/RUL files for one subset and attach capped and uncapped RUL."""
    root = Path(root)
    train = add_train_rul(read_table(root / f"train_{name}.txt"), cap)
    rul = np.atleast_1d(np.loadtxt(root / f"RUL_{name}.txt"))
    test = add_test_rul(read_table(root / f"test_{name}.txt"), rul, cap)
    return Subset(name=name, train=train, test=test)


def split_units(units: np.ndarray, cal_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Engine-level split into fit and calibration units; no engine appears in both."""
    units = np.sort(np.unique(units))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(units)
    n_cal = max(1, int(round(cal_frac * len(units))))
    return np.sort(perm[n_cal:]), np.sort(perm[:n_cal])
