"""Long-leg vs short-leg P&L decomposition for the Phase 24-RT canonical.

Sanity-check + report visual: where does the +1.08 Sharpe / +34%/yr ann
return actually come from?

Output (per leg over the full-OOS window):
  * Long-leg mean monthly + annualised return + Sharpe
  * Short-leg mean monthly + annualised return + Sharpe
  * Combined L/S monthly + annualised return + Sharpe
  * Cumulative growth chart with three lines (saved to results/...)

The finding (as of Phase 24-RT): the LONG leg makes ~+36.6%/yr (Sharpe
+1.15) while the SHORT leg contributes ~-1.9%/yr (Sharpe -0.47) — so
the strategy is essentially LONG-LEG DOMINATED, with the short leg
acting as a near-zero-P&L market-neutralizing hedge. Important for the
report's honest framing: this is "long high-conviction stocks + token
short hedge," NOT a symmetric long-short stock-picker.

Run with:
    .venv/bin/python -m notebooks.personb.long_short_decomp
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PHASE_DIR = (
    Path(__file__).resolve().parents[2]
    / "results" / "24_canonical_with_chmom"
)
RETURNS_FILE = (
    Path(__file__).resolve().parents[2] / "data" / "processed"
    / "returns_broad_sharadar_2002_2024.parquet"
)
OUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "results" / "long_short_decomp"
)


def decompose(weights: pd.DataFrame, next_returns: pd.DataFrame
              ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """For each rebalance date, return (long_pnl, short_pnl, combined_pnl).

    Long P&L = Σ over long positions of (long_weight × next_return).
    Short P&L = Σ over short positions of (short_weight × next_return),
        where short_weight is NEGATIVE — so this is the P&L from being
        short those names (positive when shorted stocks decline).
    """
    long_records = []
    short_records = []
    for t in weights.index:
        if t not in next_returns.index:
            continue
        w_t = weights.loc[t]
        rets = next_returns.loc[t].reindex(w_t.index)
        valid = w_t.index.intersection(rets.dropna().index)
        long_w = w_t.loc[valid][w_t > 0]
        short_w = w_t.loc[valid][w_t < 0]
        long_pnl = float((long_w * rets.loc[long_w.index]).sum())
        short_pnl = float((short_w * rets.loc[short_w.index]).sum())
        long_records.append((t, long_pnl))
        short_records.append((t, short_pnl))
    long_pnl = pd.Series(dict(long_records)).sort_index()
    short_pnl = pd.Series(dict(short_records)).sort_index()
    combined = (long_pnl + short_pnl).rename("combined")
    return long_pnl.rename("long"), short_pnl.rename("short"), combined


def report_stats(name: str, ser: pd.Series) -> None:
    n = len(ser)
    mean_m = ser.mean()
    std_m = ser.std()
    sharpe = mean_m / std_m * np.sqrt(12) if std_m > 0 else float("nan")
    cum = (1 + ser).prod() - 1
    cagr = (1 + cum) ** (12 / n) - 1 if n > 0 else float("nan")
    print(f"  {name:<12}  mean monthly = {mean_m*100:>+7.3f}%  "
          f"ann (CAGR) = {cagr*100:>+7.2f}%  "
          f"Sharpe = {sharpe:>+5.2f}  cum = {cum*100:>+8.1f}%")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("Phase 24-RT XGBoost canonical — long-leg vs short-leg decomposition")
    print("=" * 78)

    with open(PHASE_DIR / "per_model_results.pkl", "rb") as f:
        res = pickle.load(f)
    xgb = res["XGBoost"]
    weights = xgb.weights
    returns_wide = pd.read_parquet(RETURNS_FILE)
    next_returns = returns_wide.shift(-1)

    long_pnl, short_pnl, combined = decompose(weights, next_returns)
    # Align to the engine's portfolio_returns (which labels by realization date)
    # decompose() returns indexed by TRADE date; shift to next rebal to match engine
    # Easier: re-index by engine's portfolio_returns dates if you want exact match
    engine_pr = xgb.portfolio_returns.sort_index()

    print(f"\nFull-OOS ({len(long_pnl)} months, {long_pnl.index.min().date()} -> "
          f"{long_pnl.index.max().date()})")
    report_stats("Long leg", long_pnl)
    report_stats("Short leg", short_pnl)
    report_stats("Combined", combined)

    # Cumulative-return chart
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for label, ser, color, lw in [
        ("Long-leg cumulative",  long_pnl,  "#22C55E", 2.4),
        ("Short-leg cumulative", short_pnl, "#DC2626", 1.8),
        ("Combined L/S (canonical)", combined, "#1F3864", 2.6),
    ]:
        cum = (1 + ser).cumprod()
        final = (cum.iloc[-1] - 1) * 100
        ax.plot(cum.index, cum.values, label=f"{label}  ({final:+.0f}%)",
                color=color, lw=lw)
    ax.axhline(1.0, color="grey", lw=0.5)
    ax.set_yscale("log")
    ax.set_title("Phase 24-RT — long-leg vs short-leg P&L decomposition\n"
                 "Long leg does the work; short leg is a near-flat market-neutralizing hedge",
                 fontsize=11.5, weight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Cumulative growth of $1 (log scale)", fontsize=11)
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = OUT_DIR / "long_short_decomp_phase24.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out}")

    # Also: monthly-return scatter (long vs short) to show how often each leg
    # was the winner per month
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(short_pnl.values * 100, long_pnl.values * 100,
               alpha=0.55, s=30, color="#1F3864", edgecolor="white", lw=0.4)
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    # 45-degree line for reference
    lim = max(abs(long_pnl.min()), abs(long_pnl.max()),
              abs(short_pnl.min()), abs(short_pnl.max())) * 100
    ax.plot([-lim, lim], [-lim, lim], color="grey", lw=0.6, ls="--", alpha=0.7)
    ax.set_xlabel("Short-leg monthly P&L (%)", fontsize=11)
    ax.set_ylabel("Long-leg monthly P&L (%)", fontsize=11)
    ax.set_title("Phase 24-RT — monthly long-leg vs short-leg P&L scatter\n"
                 "Long-leg returns dominate; short-leg clusters near zero",
                 fontsize=11, weight="bold")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out2 = OUT_DIR / "long_short_scatter_phase24.png"
    fig.savefig(out2, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out2}")

    # Save raw decomposed series for the report
    decomp_df = pd.DataFrame({
        "long_pnl": long_pnl,
        "short_pnl": short_pnl,
        "combined": combined,
    })
    decomp_df.to_parquet(OUT_DIR / "decomp_series.parquet")
    print(f"Wrote {OUT_DIR / 'decomp_series.parquet'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
