"""Build brief figures: an architecture diagram plus copies of the result figures."""

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

HERE = Path(__file__).parent
FIG = HERE / "figures"
RES = HERE.parent / "results"
BLUE, LIGHT, GREY = "#1f4e9e", "#eef3fb", "#374151"


def box(ax, x, y, w, h, title, body, fc=LIGHT):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=BLUE, lw=1.2))
    ax.text(x + w / 2, y + h - 0.17, title, ha="center", va="top", fontsize=8.5,
            weight="bold", color=BLUE)
    ax.text(x + w / 2, y + h - 0.48, body, ha="center", va="top", fontsize=6.8, color=GREY,
            linespacing=1.35)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", color=GREY, lw=1.1))


def architecture():
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    box(ax, 0.1, 2.5, 2.1, 1.9, "C-MAPSS data",
        "FD001-FD004\nrun-to-failure train\ntruncated test + RUL\nRUL capped at 125")
    box(ax, 2.6, 2.5, 2.2, 1.9, "Preprocess",
        "engine-level split\n75% fit / 25% calibrate\n(15% of fit engines\nfor early stopping)\nper-condition z-score")
    box(ax, 5.2, 2.5, 2.2, 1.9, "Models",
        "LightGBM, XGBoost\n(window features)\n1D-CNN, LSTM\n(3 training seeds)\nWeibull AFT survival")
    box(ax, 7.8, 2.5, 2.1, 1.9, "Conformal",
        "split conformal\nCQR on LightGBM\nquantiles (5%, 95%)\ntarget: 90% coverage")
    box(ax, 2.6, 0.15, 3.2, 1.5, "Evaluate (official test sets)",
        "RMSE + bootstrap CI, NASA score\n"
        "coverage and width per subset\none prediction per test engine")
    box(ax, 6.3, 0.15, 3.6, 1.5, "Maintenance decision",
        "schedule at predicted RUL - margin\nfailure = 10x planned visit\n"
        "margin tuned on calibration engines\nvs. conformal lower bound")
    arrow(ax, 2.2, 3.45, 2.6, 3.45)
    arrow(ax, 4.8, 3.45, 5.2, 3.45)
    arrow(ax, 7.4, 3.45, 7.8, 3.45)
    # Point predictions feed both evaluation and the margin policy, and conformal intervals
    # feed both coverage/width evaluation and the lower-bound policies (pipeline._evaluate_seed).
    arrow(ax, 5.5, 2.5, 5.5, 1.67)
    arrow(ax, 6.9, 2.5, 6.9, 1.67)
    arrow(ax, 8.1, 2.5, 5.75, 1.67)
    arrow(ax, 8.85, 2.5, 8.85, 1.67)
    fig.tight_layout()
    fig.savefig(FIG / "architecture.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    architecture()
    for p in RES.glob("*.png"):
        shutil.copy(p, FIG / p.name)
