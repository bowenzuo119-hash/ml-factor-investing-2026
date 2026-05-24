"""Generate honest-framing canonical plots for Phase 24-RT (the new final).

Addresses the "+700% cumulative looks shocking" problem by adding THREE
visualisations that decompose the gross return:

  1. equity_curve_phase24_honest.png  -- 3-line equity curve (log Y-axis):
       (a) XGBoost raw cumulative
       (b) SPY total return benchmark
       (c) XGBoost minus (beta * Mkt-RF) -- beta-hedged "pure alpha" curve

  2. drawdown_phase24.png             -- drawdown trajectory, XGBoost vs SPY

  3. ff5_decomposition_phase24.png    -- annualised return decomposition
       (pure alpha + market beta + size + value + profitability + investment)

  4. phase_progression_phase24.png    -- Sharpe history with honest 24-RT
       headline (replaces phase_progression_phase23g.png)

The "honest framing" point: a +1.07 Sharpe with beta=+1.3 isn't a
market-neutral L/S edge -- it's a 1.3x leveraged long-market book PLUS
+18%/yr alpha. The visuals make this decomposition explicit instead of
silently inflating with the market.

Run with:
    .venv/bin/python generate_honest_canonical_plots.py
"""
from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).parent
OUT_DIR = ROOT / "results" / "final_canonical_plots"
PHASE_DIR = ROOT / "results" / "24_canonical_with_chmom"   # Phase 24-RT
PHASE_NAME = "Phase 24-RT"


def load_ff5():
    spec = importlib.util.spec_from_file_location(
        "p7", "notebooks/personb/07_statistical_robustness.py")
    mod = importlib.util.module_from_spec(spec); sys.modules["p7"] = mod
    spec.loader.exec_module(mod)
    return mod.fetch_ff_monthly(five_factor=True), mod


def ff5_regress_aligned(rets: pd.Series, ff5: pd.DataFrame,
                        lo: pd.Timestamp, hi: pd.Timestamp, mod) -> dict:
    sl = rets[(rets.index >= lo) & (rets.index <= hi)]
    a = pd.DataFrame({"y": sl.values, "ym": sl.index.to_period("M")}).set_index("ym")
    b = pd.DataFrame(ff5.values, index=ff5.index.to_period("M"), columns=ff5.columns)
    merged = a.join(b, how="inner").dropna()
    y = (merged["y"] - merged["RF"]).to_numpy()
    Xcols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    X = np.column_stack([np.ones(len(merged))] + [merged[c].to_numpy() for c in Xcols])
    r = mod.regress_with_hac(y, X, lags=6)
    return {
        "n": r["n"], "r2": r["r2"],
        "alpha_ann": r["beta"][0] * 12 * 100,
        "alpha_t": r["t"][0], "alpha_p": r["p"][0],
        "Mkt-RF": (r["beta"][1], r["t"][1], r["p"][1]),
        "SMB": (r["beta"][2], r["t"][2], r["p"][2]),
        "HML": (r["beta"][3], r["t"][3], r["p"][3]),
        "RMW": (r["beta"][4], r["t"][4], r["p"][4]),
        "CMA": (r["beta"][5], r["t"][5], r["p"][5]),
    }


def align_to_ff5(rets: pd.Series, ff5: pd.DataFrame) -> pd.DataFrame:
    """Inner-join portfolio returns and FF5 monthly factors by year-month."""
    a = pd.DataFrame({"y": rets.values, "ym": rets.index.to_period("M")}).set_index("ym")
    b = pd.DataFrame(ff5.values, index=ff5.index.to_period("M"), columns=ff5.columns)
    return a.join(b, how="inner").dropna()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PHASE_DIR / "per_model_results.pkl", "rb") as f:
        res = pickle.load(f)

    rets_xgb = res["XGBoost"].portfolio_returns.sort_index()
    rets_lasso = res["Lasso"].portfolio_returns.sort_index()
    rets_nn = res["NN"].portfolio_returns.sort_index()

    print("Loading FF5 factors...")
    ff5, ff5_mod = load_ff5()

    # FF5 regression on full-OOS for the beta-hedged curve
    LO_FULL = pd.Timestamp("2012-04-01"); HI = pd.Timestamp("2024-12-31")
    LO_LONG = pd.Timestamp("2015-01-01")
    TEST_START = pd.Timestamp("2019-01-01")

    ff5_full = ff5_regress_aligned(rets_xgb, ff5, LO_FULL, HI, ff5_mod)
    ff5_long = ff5_regress_aligned(rets_xgb, ff5, LO_LONG, HI, ff5_mod)
    beta_full = ff5_full["Mkt-RF"][0]
    print(f"Phase 24-RT FF5: full-OOS alpha={ff5_full['alpha_ann']:+.2f}%/yr "
          f"(t={ff5_full['alpha_t']:+.2f}), Mkt-β={beta_full:+.2f}")

    # =============================================================
    # 1. EQUITY CURVE WITH SPY BENCHMARK + BETA-HEDGED PURE ALPHA
    # =============================================================
    aligned = align_to_ff5(rets_xgb, ff5)
    # SPY-like total return = Mkt-RF + RF
    aligned["spy"] = aligned["Mkt-RF"] + aligned["RF"]
    # Beta-hedged portfolio = raw return minus beta*Mkt-RF
    # (excess-return frame; we hedge the market-excess exposure)
    aligned["xgb_hedged_excess"] = (aligned["y"] - aligned["RF"]) - beta_full * aligned["Mkt-RF"]
    # Convert back to total return by adding RF
    aligned["xgb_hedged"] = aligned["xgb_hedged_excess"] + aligned["RF"]

    # Re-index back to actual trading-day month-end dates
    # (period-aligned merge lost the exact dates; reconstruct from rets_xgb)
    aligned_dates = []
    period_to_date = {p: d for p, d in zip(rets_xgb.index.to_period("M"), rets_xgb.index)}
    aligned_dates = [period_to_date[p] for p in aligned.index]
    aligned.index = pd.DatetimeIndex(aligned_dates)

    fig, ax = plt.subplots(figsize=(13, 6.5))
    cum_xgb = (1 + aligned["y"]).cumprod()
    cum_spy = (1 + aligned["spy"]).cumprod()
    cum_hedge = (1 + aligned["xgb_hedged"]).cumprod()

    ax.plot(cum_xgb.index, cum_xgb.values, label=f"XGBoost (canonical) — gross +{(cum_xgb.iloc[-1]-1)*100:.0f}%",
            color="#22C55E", lw=2.4)
    ax.plot(cum_spy.index, cum_spy.values,
            label=f"S&P 500 total return — +{(cum_spy.iloc[-1]-1)*100:.0f}%",
            color="#94A3B8", lw=1.6, ls="--")
    ax.plot(cum_hedge.index, cum_hedge.values,
            label=f"Beta-hedged pure alpha — +{(cum_hedge.iloc[-1]-1)*100:.0f}%  (subtract β·SPY)",
            color="#DC2626", lw=2.0)

    ax.axvline(TEST_START, color="grey", lw=0.7, ls=":", alpha=0.7)
    ax.text(TEST_START, ax.get_ylim()[1] * 0.5, "  test window starts (2019)",
            color="grey", fontsize=9.5, rotation=90, va="center")

    ax.set_yscale("log")
    ax.set_title(
        f"{PHASE_NAME} canonical — cumulative growth of $1, log scale\n"
        f"Decomposition: green = strategy gross (1.4× market exposure + alpha) · "
        f"grey = SPY benchmark · red = pure alpha after β·SPY hedge",
        fontsize=11, weight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Cumulative wealth (log scale, $1 → $X)", fontsize=11)
    ax.legend(loc="upper left", fontsize=10.5)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "equity_curve_phase24_honest.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'equity_curve_phase24_honest.png'}")

    # =============================================================
    # 2. DRAWDOWN vs SPY
    # =============================================================
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for label, ser, color, lw in [
        ("XGBoost (canonical)", aligned["y"], "#22C55E", 1.8),
        ("S&P 500", aligned["spy"], "#94A3B8", 1.4),
        ("Beta-hedged pure alpha", aligned["xgb_hedged"], "#DC2626", 1.6),
    ]:
        wealth = (1 + ser).cumprod()
        dd = (wealth / wealth.cummax() - 1) * 100
        worst = dd.min()
        ax.plot(dd.index, dd.values, label=f"{label} (max DD {worst:.1f}%)",
                color=color, lw=lw, ls="--" if label == "S&P 500" else "-")
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(TEST_START, color="grey", lw=0.7, ls=":", alpha=0.7)
    ax.set_title(f"{PHASE_NAME} — drawdown vs SPY (raw XGBoost has high β exposure → high DD)",
                 fontsize=11.5, weight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Drawdown (%)", fontsize=11)
    ax.legend(loc="lower left", fontsize=10.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "drawdown_phase24.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'drawdown_phase24.png'}")

    # =============================================================
    # 3. FF5 RETURN DECOMPOSITION (long-OOS)
    # =============================================================
    factor_premia = {"Mkt-RF": 0.135, "SMB": -0.01, "HML": 0.00, "RMW": 0.04, "CMA": 0.01}
    factor_contribs = {f: ff5_long[f][0] * p for f, p in factor_premia.items()}
    alpha_pct = ff5_long["alpha_ann"] / 100
    total = alpha_pct + sum(factor_contribs.values())

    fig, ax = plt.subplots(figsize=(11, 6))
    labels = ["Pure FF5 alpha", "Mkt-RF (β)", "SMB (size)",
              "HML (value)", "RMW (profit)", "CMA (invest)"]
    values = [alpha_pct, factor_contribs["Mkt-RF"], factor_contribs["SMB"],
              factor_contribs["HML"], factor_contribs["RMW"], factor_contribs["CMA"]]
    colors = ["#22C55E", "#DC2626", "#3B82F6", "#F59E0B", "#8B5CF6", "#06B6D4"]

    bars = ax.bar(labels, [v * 100 for v in values], color=colors,
                  edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2,
                (v * 100) + (0.5 if v >= 0 else -0.5),
                f"{v*100:+.2f}%", ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=10.5, weight="bold")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_ylabel("Annualised return contribution (%)", fontsize=11)
    ax.set_title(
        f"{PHASE_NAME} long-OOS (2015-24) — annualised return decomposition\n"
        f"Total realised +{total*100:.1f}%/yr  =  pure FF5 alpha +{alpha_pct*100:.1f}%/yr  +  factor exposures",
        fontsize=11.5, weight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.98, 0.93,
            f"FF5 α t-stat = {ff5_long['alpha_t']:.2f}  (p={ff5_long['alpha_p']:.4f}, SIGNIFICANT)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10.5, weight="bold", color="#22C55E",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#22C55E"))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ff5_decomposition_phase24.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'ff5_decomposition_phase24.png'}")

    # =============================================================
    # 4. PHASE PROGRESSION (updated with 24-RT)
    # =============================================================
    # Note: leaky +1.50 lived in Phase 14 (results/14_official_canonical_k5),
    # not Phase 15. Phase 15 is the SAME panel + PIT applied = the collapse.
    # Phase 22 = strict-PIT S&P only.  Phase 23g = broad rebuild.
    # Phase 24-RT = broad + chmom + retune = FINAL canonical.
    PHASE_HISTORY = [
        ("P14 (leaky)",     1.495, "BUG\n+1.49"),
        ("P15 (+PIT)",     -0.309, "leak\nremoved"),
        ("P22 (S&P only)",  0.313, "honest\n+0.31"),
        ("P23g (broad)",    0.953, "broad\n+0.95"),
        ("P24-RT (final)",  0.980, "FINAL\n+0.98"),
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    labels = [x[0] for x in PHASE_HISTORY]
    sharpes = [x[1] for x in PHASE_HISTORY]
    notes = [x[2] for x in PHASE_HISTORY]
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
    ax.set_title("Project Sharpe progression — audit + rebuild + GKX-feature extension",
                 fontsize=12, weight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase_progression_phase24.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'phase_progression_phase24.png'}")

    print(f"\nAll 4 plots in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
