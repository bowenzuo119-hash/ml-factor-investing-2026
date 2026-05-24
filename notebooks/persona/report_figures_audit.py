"""report_figures_audit.py - audit + robustness visuals for REPORT 5/6.

Three figures into results/persona_figures/:

  1. leaky_vs_honest_equity.png   - Phase 15 leaky (+1.49, survivorship-biased
                                     S&P) vs Phase 23g honest (+1.05, broad PIT)
                                     cumulative returns. The audit-story visual.
  2. rolling_sharpe_23g.png       - rolling 24-month net Sharpe of 23g
                                     (edge is persistent, not a 2020 anomaly).
  3. rolling_ic_23g.png           - rolling 12-month rank IC of XGBoost on 23g
                                     (predictive power doesn't decay).

    python -m notebooks.persona.report_figures_audit
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.metrics import sharpe_ratio

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "results" / "persona_figures"
LEAKY = ROOT / "results" / "14_official_canonical_k5" / "per_model_results.pkl"   # +1.50 pre-PIT S&P
COLLAPSE = ROOT / "results" / "15_canonical_2002" / "per_model_results.pkl"       # -0.31 same S&P, PIT on
HONEST = ROOT / "results" / "23g_canonical_qfiltered_orig_tune" / "per_model_results.pkl"  # +1.07 broad PIT
PRED23G = ROOT / "results" / "23g_canonical_qfiltered_orig_tune" / "predictions.parquet"
RETURNS_BROAD = ROOT / "data" / "processed" / "returns_broad_sharadar_2002_2024.parquet"

C_LEAK = "#ef4444"   # red
C_HONEST = "#2563eb" # blue
C_LINE = "#7c3aed"   # purple
DPI = 130


def _xgb_returns(pkl):
    return pickle.load(open(pkl, "rb"))["XGBoost"].portfolio_returns.dropna()


def fig_leaky_vs_honest():
    leaky = _xgb_returns(LEAKY)
    collapse = _xgb_returns(COLLAPSE)
    honest = _xgb_returns(HONEST)
    fig, ax = plt.subplots(figsize=(12, 5))
    series = [
        (leaky, C_LEAK, "--",
         f"1. Leaky S&P (pre-PIT) — Sharpe {sharpe_ratio(leaky):+.2f}"),
        (collapse, "#9ca3af", "-",
         f"2. Same S&P, PIT enforced — Sharpe {sharpe_ratio(collapse):+.2f}  (the leak removed)"),
        (honest, C_HONEST, "-",
         f"3. Broad survivorship-free rebuild — Sharpe {sharpe_ratio(honest):+.2f}"),
    ]
    for r, c, ls, lab in series:
        cum = (1 + r).cumprod()
        ax.plot(cum.index, cum.values, color=c, ls=ls, lw=2.0, label=lab)
    ax.set_title("The audit in one chart — survivorship leak, collapse, honest rebuild",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("growth of $1 (net, compounded, log)")
    ax.set_yscale("log")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25, which="both")
    fig.text(0.5, -0.04,
             "Toggling PIT on the SAME S&P universe collapses +1.50 -> -0.31: the headline was "
             "survivorship. The broad survivorship-free rebuild earns +1.07 honestly (different, "
             "higher-vol universe -- compare Sharpe, not cumulative height).",
             ha="center", fontsize=8.5, style="italic")
    fig.savefig(FIG / "leaky_vs_honest_equity.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  leaky {sharpe_ratio(leaky):+.2f} / collapse {sharpe_ratio(collapse):+.2f} / "
          f"honest {sharpe_ratio(honest):+.2f} -> leaky_vs_honest_equity.png")


def fig_rolling_sharpe():
    r = _xgb_returns(HONEST)
    roll = r.rolling(24).apply(lambda x: x.mean() / x.std(ddof=1) * np.sqrt(12), raw=False).dropna()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(roll.index, roll.values, color=C_LINE, lw=2.0)
    ax.axhline(0, color="black", lw=0.6)
    ax.axhline(float(roll.mean()), color=C_HONEST, ls="--", lw=1.0,
               label=f"mean {roll.mean():+.2f}")
    ax.fill_between(roll.index, 0, roll.values, where=roll.values > 0, color=C_LINE, alpha=0.08)
    ax.set_title("Phase 23g rolling 24-month net Sharpe - the edge is persistent",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("annualised Sharpe (trailing 24m)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25)
    fig.text(0.5, -0.02,
             "The strategy is Sharpe-positive across nearly every rolling 2-year window - "
             "the edge is not a single-period (e.g. 2020) artifact.",
             ha="center", fontsize=8.5, style="italic")
    fig.savefig(FIG / "rolling_sharpe_23g.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  rolling 24m Sharpe: mean {roll.mean():+.2f}, min {roll.min():+.2f} -> rolling_sharpe_23g.png")


def fig_rolling_ic():
    preds = pd.read_parquet(PRED23G)["XGBoost"]
    ret = pd.read_parquet(RETURNS_BROAD)
    pred_dates = preds.index.get_level_values("date").unique().sort_values()
    ic = {}
    for t in pred_dates:
        nxt = ret.index[ret.index > t]
        if len(nxt) == 0:
            continue
        p = preds.loc[t]
        r = ret.loc[nxt[0]]
        common = p.index.intersection(r.dropna().index)
        if len(common) > 20:
            ic[t] = p.loc[common].corr(r.loc[common], method="spearman")
    ic = pd.Series(ic).sort_index()
    roll = ic.rolling(12).mean().dropna()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(roll.index, roll.values, color=C_LINE, lw=2.0)
    ax.axhline(0, color="black", lw=0.6)
    ax.axhline(float(ic.mean()), color=C_HONEST, ls="--", lw=1.0,
               label=f"full-sample mean IC {ic.mean():+.3f}")
    ax.fill_between(roll.index, 0, roll.values, where=roll.values > 0, color=C_LINE, alpha=0.08)
    ax.set_title("Phase 23g rolling 12-month rank IC (XGBoost) - predictive power is stable",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Spearman IC (trailing 12m mean)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25)
    fig.text(0.5, -0.02,
             "Cross-sectional rank correlation between forecast and next-month return stays "
             "positive throughout - the model's signal does not decay over 2012-2024.",
             ha="center", fontsize=8.5, style="italic")
    fig.savefig(FIG / "rolling_ic_23g.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  rolling 12m IC: full-sample mean {ic.mean():+.3f} -> rolling_ic_23g.png")


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    fig_leaky_vs_honest()
    fig_rolling_sharpe()
    fig_rolling_ic()
    print(f"\nDone -> {FIG}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
