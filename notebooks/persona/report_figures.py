"""report_figures.py - Person A's methodology / data figures for the report.

Run with:
    python -m notebooks.persona.report_figures

Person B owns the performance charts (cumulative returns, drawdowns).
This script produces the *data + methodology* figures that justify the
pipeline itself:

  1. sanity_3panel.png   - Random / Oracle / Uniform cumulative curves.
       The Project Framework §4.6 gate, made visual: random ~ flat,
       oracle ~ explosive, uniform ~ flat at 1.0. Shows the engine has no
       look-ahead and trades on the prediction sign.
  2. universe_coverage.png - S&P 500 constituent count per month, proving
       point-in-time membership (no survivorship bias) over 2003-2025.
  3. walkforward_scheme.png - schematic of the expanding train / fixed
       test-block walk-forward (train_window=120, test_window=12).
  4. splice_timeline.png - CRSP vs yfinance source mix per month, showing
       the 2022-12 handoff in the spliced price panel.

Figures land in results/persona_figures/ (committable; the report links them).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import load_prices_spliced, load_sp500_membership
from src.backtest import run_walk_forward_backtest
from src.sanity import RandomModel, OracleModel, UniformModel

BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "results" / "persona_figures"

FIG_START, FIG_END = "2010-01-01", "2024-12-31"


def _build_panel():
    """Spliced returns (wide) + a trivial feature panel for the sanity run."""
    spliced = load_prices_spliced(start=FIG_START, end=FIG_END)
    returns = spliced["ret"].unstack("ticker").sort_index()
    # Restrict to the 200 most-present names so the sanity backtest is quick.
    keep = returns.notna().sum().sort_values(ascending=False).head(200).index
    returns = returns[keep]
    rng = np.random.default_rng(0)
    idx = pd.MultiIndex.from_product(
        [returns.index, returns.columns], names=["date", "asset"]
    )
    features = pd.DataFrame(
        rng.standard_normal((len(idx), 3)), index=idx, columns=["f1", "f2", "f3"]
    )
    return returns, features, spliced


def fig_sanity_3panel(returns, features):
    models = {
        "Random\n(must be flat)": RandomModel(),
        "Oracle\n(must explode up)": OracleModel(returns=returns),
        "Uniform\n(must be flat at 1.0)": UniformModel(),
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (name, model) in zip(axes, models.items()):
        res = run_walk_forward_backtest(
            returns=returns, features=features, model=model,
            train_window=24, test_window=12, transaction_cost_bps=0.0,
        )
        curve = (1.0 + res.gross_returns).cumprod()
        ax.plot(curve.index, curve.values, lw=1.6)
        ax.axhline(1.0, color="gray", ls="--", lw=0.8)
        ax.set_title(name, fontsize=11)
        ax.set_ylabel("cumulative gross NAV")
        if "Oracle" in name:
            ax.set_yscale("log")
    fig.suptitle("Backtest engine sanity gate (Project Framework §4.6)", fontsize=13)
    fig.tight_layout()
    _save(fig, "sanity_3panel.png")


def fig_universe_coverage():
    months = pd.date_range(FIG_START, FIG_END, freq="ME")
    counts = [len(load_sp500_membership(m.strftime("%Y-%m-%d"))) for m in months]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(months, counts, lw=1.5)
    ax.set_title("Point-in-time S&P 500 membership count (no survivorship bias)")
    ax.set_ylabel("# constituents")
    ax.set_ylim(0, max(counts) * 1.1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "universe_coverage.png")


def fig_walkforward_scheme():
    train_window, test_window, n_folds = 120, 12, 6
    fig, ax = plt.subplots(figsize=(11, 4))
    for f in range(n_folds):
        start = f * test_window
        ax.barh(f, train_window, left=start, color="#60a5fa", edgecolor="white")
        ax.barh(f, test_window, left=start + train_window, color="#f59e0b",
                edgecolor="white")
    ax.set_yticks(range(n_folds))
    ax.set_yticklabels([f"fold {f+1}" for f in range(n_folds)])
    ax.invert_yaxis()
    ax.set_xlabel("rebalance period (months)")
    ax.set_title("Walk-forward: expanding-origin train (120m) + frozen test block (12m)")
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color="#60a5fa", label="train (refit here)"),
        plt.Rectangle((0, 0), 1, 1, color="#f59e0b", label="test (frozen model)"),
    ], loc="lower right")
    fig.tight_layout()
    _save(fig, "walkforward_scheme.png")


def fig_splice_timeline(spliced):
    src = (
        spliced.reset_index()
        .groupby([pd.Grouper(key="date", freq="ME"), "source"]).size()
        .unstack("source", fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.stackplot(
        src.index,
        [src.get("crsp", 0), src.get("yfinance", 0)],
        labels=["CRSP (<= 2022-12)", "yfinance (>= 2023-01)"],
        colors=["#34d399", "#a78bfa"],
    )
    ax.axvline(pd.Timestamp("2022-12-30"), color="black", ls="--", lw=1)
    ax.set_title("Spliced price panel: CRSP -> yfinance source mix per month")
    ax.set_ylabel("# stocks with a return")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, "splice_timeline.png")


def _save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(BASE_DIR)}")


def main() -> int:
    print("Building Person A methodology figures...")
    returns, features, spliced = _build_panel()
    fig_sanity_3panel(returns, features)
    fig_universe_coverage()
    fig_walkforward_scheme()
    fig_splice_timeline(spliced)
    print(f"\nDone. 4 figures in {OUT_DIR.relative_to(BASE_DIR)}/")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
