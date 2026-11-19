"""Result figures written next to metrics.json."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, GREY, GREEN, RED = "#1f4e9e", "#d97706", "#6b7280", "#15803d", "#b91c1c"
MODEL_COLORS = {"LightGBM": BLUE, "XGBoost": "#60a5fa", "1D-CNN": GREEN, "LSTM": ORANGE,
                "Weibull AFT": GREY}


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)


def fig_rmse(payload: dict, path: Path) -> None:
    subs = list(payload["subsets"])
    models = list(next(iter(payload["subsets"].values()))["models"])
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    width = 0.8 / len(models)
    x = np.arange(len(subs))
    for i, m in enumerate(models):
        vals = [payload["subsets"][s]["models"][m]["rmse_mean"] for s in subs]
        err = [payload["subsets"][s]["models"][m]["rmse_sd"] for s in subs]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, yerr=err, capsize=2, label=m,
               color=MODEL_COLORS.get(m, GREY))
    ax.set_xticks(x, subs)
    ax.set_ylabel("Test RMSE (cycles), mean and SD")
    ax.legend(ncol=5, fontsize=7.5, frameon=False, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_coverage(payload: dict, path: Path, alpha: float) -> None:
    subs = list(payload["subsets"])
    methods = ["QR-LightGBM (uncalibrated)", "CQR-LightGBM", "SC-LightGBM", "SC-LSTM"]
    colors = [GREY, BLUE, "#60a5fa", ORANGE]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0))
    width = 0.8 / len(methods)
    x = np.arange(len(subs))
    for i, (m, c) in enumerate(zip(methods, colors, strict=True)):
        cov = [payload["subsets"][s]["intervals"][m]["coverage_mean"] for s in subs]
        wid = [payload["subsets"][s]["intervals"][m]["width_mean"] for s in subs]
        off = x + i * width - 0.4 + width / 2
        axes[0].bar(off, cov, width, color=c, label=m)
        axes[1].bar(off, wid, width, color=c, label=m)
    axes[0].axhline(1 - alpha, color=RED, lw=1, ls="--")
    axes[0].set_ylim(0.5, 1.02)
    axes[0].set_ylabel("Empirical coverage")
    axes[1].set_ylabel("Mean interval width (cycles)")
    for ax in axes:
        ax.set_xticks(x, subs)
        _style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, frameon=False, loc="upper center", ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_intervals(arrays: dict, path: Path, name: str) -> None:
    y = arrays["y_test"]
    order = np.argsort(y)
    p = arrays["preds_test"]["LightGBM"]
    lo, hi = arrays["bounds"]["CQR-LightGBM"]
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    xs = np.arange(len(y))
    ax.fill_between(xs, lo[order], hi[order], color=BLUE, alpha=0.18, label="CQR 90% interval")
    ax.plot(xs, p[order], ".", color=BLUE, ms=4, label="LightGBM prediction")
    ax.plot(xs, y[order], "-", color="black", lw=1.2, label="True RUL (capped at 125)")
    ax.set_xlabel(f"{name} test engines, sorted by true RUL")
    ax.set_ylabel("RUL (cycles)")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_cost(all_arrays: dict, payload: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    palette = [BLUE, ORANGE, GREEN, RED]
    for c, (name, arr) in zip(palette, all_arrays.items(), strict=False):
        ax.plot(arr["margins"], arr["test_costs"], color=c, label=f"{name} test")
        ax.plot(arr["margins"], arr["cal_costs"], color=c, ls=":", lw=1)
        m = arr["tuned_margin"]
        ax.axvline(m, color=c, lw=0.6, alpha=0.5)
    ax.set_xlabel("Safety margin m (maintain at predicted RUL - m)")
    ax.set_ylabel("Mean cost per engine")
    ax.legend(fontsize=7.5, frameon=False, ncol=2)
    ax.set_title("Seed 0. Solid: test engines; dotted: calibration engines; vertical: tuned m",
                 fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def make_all(all_arrays: dict, payload: dict, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    fig_rmse(payload, out_dir / "rmse_by_model.png")
    fig_coverage(payload, out_dir / "coverage_width.png", payload["config"]["alpha"])
    first = next(iter(all_arrays))
    fig_intervals(all_arrays[first], out_dir / f"intervals_{first}.png", first)
    fig_cost(all_arrays, payload, out_dir / "cost_vs_margin.png")
