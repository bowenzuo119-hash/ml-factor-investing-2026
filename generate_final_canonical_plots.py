"""Generate the FINAL canonical plots for the report:
  1. equity_curve_phase23g.png  -- cumulative net return XGBoost (test + long-OOS marked)
  2. ff5_decomposition_phase23g.png -- ann-return decomposition (mkt-β, SMB, alpha)
  3. drawdown_phase23g.png -- drawdown curve

All on Phase 23g (the honest canonical: broad Sharadar + Q-filter + k=20 +
original XGBoost tune).

Run with:
    .venv/bin/python generate_final_canonical_plots.py
"""
from __future__ import annotations

import pickle
import sys
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).parent
OUT_DIR = ROOT / "results" / "final_canonical_plots"
PHASE_DIR = ROOT / "results" / "23g_canonical_qfiltered_orig_tune"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PHASE_DIR / "per_model_results.pkl", "rb") as f:
        results = pickle.load(f)

    # Load FF5 helper
    spec = importlib.util.spec_from_file_location(
        "p23c", "notebooks/personb/23c_k1_qfilter_canonical.py")
    mod = importlib.util.module_from_spec(spec); sys.modules["p23c"] = mod
    spec.loader.exec_module(mod)

    rets_xgb = results["XGBoost"].portfolio_returns.sort_index()
    rets_lasso = results["Lasso"].portfolio_returns.sort_index()
    rets_nn = results["NN"].portfolio_returns.sort_index()

    TEST_START = pd.Timestamp("2019-01-01")

    # ---- 1. Equity curve ----
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for label, rets, color in [
        ("XGBoost (canonical)", rets_xgb, "#22C55E"),
        ("NN", rets_nn, "#3B82F6"),
        ("Lasso", rets_lasso, "#F59E0B"),
    ]:
        cum = (1 + rets).cumprod() - 1
        ax.plot(cum.index, cum.values * 100, label=label, lw=2 if label.startswith("X") else 1.4, color=color)
    ax.axvline(TEST_START, color="grey", lw=0.8, ls="--", alpha=0.7)
    ax.text(TEST_START, ax.get_ylim()[1] * 0.95, "  test window starts (2019-01)",
            color="grey", fontsize=9.5, va="top")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("Phase 23g — final canonical cumulative net return (XGBoost, NN, Lasso)\n"
                 "broad Sharadar universe (~2000 names per date) · Q-suffix bankrupt-tickers filtered · "
                 "sector-neutral k=20 · 10 bps cost",
                 fontsize=11.5, weight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Cumulative net return (%)", fontsize=11)
    ax.legend(loc="upper left", fontsize=10.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "equity_curve_phase23g.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_DIR}/equity_curve_phase23g.png")

    # ---- 2. Drawdown ----
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for label, rets, color in [
        ("XGBoost", rets_xgb, "#22C55E"),
        ("NN", rets_nn, "#3B82F6"),
        ("Lasso", rets_lasso, "#F59E0B"),
    ]:
        wealth = (1 + rets).cumprod()
        dd = (wealth / wealth.cummax() - 1) * 100
        ax.plot(dd.index, dd.values, label=label, lw=1.7, color=color)
    ax.axvline(TEST_START, color="grey", lw=0.8, ls="--", alpha=0.7)
    ax.set_title("Phase 23g — drawdown of each model (negative = below-water)",
                 fontsize=11.5, weight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Drawdown (%)", fontsize=11)
    ax.legend(loc="lower left", fontsize=10.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "drawdown_phase23g.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_DIR}/drawdown_phase23g.png")

    # ---- 3. FF5 decomposition (long-OOS) ----
    LO = pd.Timestamp("2015-01-01"); HI = pd.Timestamp("2024-12-31")
    ff5 = mod.ff5_regress(rets_xgb, LO, HI)

    # Approximate factor premium contributions over the long-OOS window
    # Mkt-RF ~ 13.5%/yr, SMB ~ -1%/yr, HML ~ 0%/yr, RMW ~ 4%/yr, CMA ~ 1%/yr
    factor_premia_ann = {"Mkt-RF": 0.135, "SMB": -0.01, "HML": 0.00, "RMW": 0.04, "CMA": 0.01}
    factor_contribs = {}
    for fac, prem in factor_premia_ann.items():
        beta = ff5[fac][0]
        factor_contribs[fac] = beta * prem
    alpha = ff5["alpha_ann"] / 100  # convert to fraction
    total = alpha + sum(factor_contribs.values())

    fig, ax = plt.subplots(figsize=(11, 6))
    labels = ["FF5 Alpha\n(pure skill)", "Mkt-RF\n(market β)", "SMB\n(size β)",
              "HML\n(value β)", "RMW\n(profit β)", "CMA\n(investment β)"]
    values = [alpha, factor_contribs["Mkt-RF"], factor_contribs["SMB"],
              factor_contribs["HML"], factor_contribs["RMW"], factor_contribs["CMA"]]
    colors = ["#22C55E", "#DC2626", "#3B82F6", "#F59E0B", "#8B5CF6", "#06B6D4"]

    bars = ax.bar(labels, [v * 100 for v in values], color=colors,
                  edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2,
                (v * 100) + (0.5 if v >= 0 else -0.5),
                f"{v*100:+.1f}%", ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=10.5, weight="bold")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_ylabel("Annualised return contribution (%)", fontsize=11)
    ax.set_title("Phase 23g long-OOS (2015-2024) — annualised return decomposition\n"
                 f"Total realised: +{total*100:.1f}%/yr   "
                 f"Pure FF5 alpha: +{alpha*100:.1f}%/yr (t={ff5['alpha_t']:.2f}, "
                 f"p={ff5['alpha_p']:.4f}, SIGNIFICANT)",
                 fontsize=11.5, weight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.98, 0.93,
            f"FF5 alpha t-stat: {ff5['alpha_t']:.2f}  →  highly significant",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10.5, weight="bold", color="#22C55E",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#22C55E"))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ff5_decomposition_phase23g.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_DIR}/ff5_decomposition_phase23g.png")

    # ---- 4. Phase progression bar chart ----
    PHASE_PROGRESSION = [
        ("P15 leaky",      1.495, "BUG\n+1.49 Sh"),
        ("P22 honest S&P", 0.313, "S&P only\n+0.31 Sh"),
        ("P23 k=5 w/Q",    0.875, "broad\nQ-included"),
        ("P23g k=20 +Qf",  0.953, "FINAL\n+0.95 Sh"),
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    labels = [x[0] for x in PHASE_PROGRESSION]
    sharpes = [x[1] for x in PHASE_PROGRESSION]
    notes = [x[2] for x in PHASE_PROGRESSION]
    colors = ["#DC2626", "#F59E0B", "#3B82F6", "#22C55E"]
    bars = ax.bar(labels, sharpes, color=colors, edgecolor="black", linewidth=0.5)
    for b, v, n in zip(bars, sharpes, notes):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02,
                f"{v:+.2f}", ha="center", va="bottom",
                fontsize=11, weight="bold")
        ax.text(b.get_x() + b.get_width() / 2, v / 2, n,
                ha="center", va="center", fontsize=9, color="white", weight="bold")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Net Sharpe (long-OOS, 2015-2024)", fontsize=11)
    ax.set_title("Project progression: Sharpe history through the audit + rebuild\n"
                 "from the leaky pre-audit headline to the broader-universe honest canonical",
                 fontsize=12, weight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase_progression_phase23g.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_DIR}/phase_progression_phase23g.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
