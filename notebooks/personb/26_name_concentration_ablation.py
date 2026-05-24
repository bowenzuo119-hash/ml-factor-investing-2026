"""26_name_concentration_ablation.py — single-name P&L concentration check.

Closes the §6 name-fragility loop with a concrete number, not just the
IONQ/NDAQ pair from Bowen's q-filter decomposition. The question: in the
committed Phase 24-RT canonical pkl (buggy-filter baseline; IONQ + NDAQ
not present), are there OTHER single names whose removal would move the
Sharpe by >0.05? If yes, the strategy is name-fragile in a way the
§6 capacity caveat understates. If no, the IONQ contribution is the
worst-case single-name fragility we've measured.

Method: post-process the committed pkl's weights (no model re-run).
  1. Compute per-name lifetime P&L contribution =
     sum over months of (weight_{t,name} * next-period-return_{name})
  2. Identify the top-10 contributors by abs(lifetime P&L).
  3. For each top contributor, drop it from the weights matrix entirely,
     renormalise the long and short legs separately per rebalance date,
     recompute net returns, and report the resulting Sharpe / FF5 alpha.
  4. Print a leave-one-out table.

Output: results/26_name_concentration/leave_one_out.csv
        results/26_name_concentration/top_contributors.csv

Run with:
    .venv/bin/python -m notebooks.personb.26_name_concentration_ablation
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import sharpe_ratio


ROOT = Path(__file__).resolve().parents[2]
CANON = ROOT / "results" / "24_canonical_with_chmom" / "per_model_results.pkl"
RETURNS_FILE = (
    ROOT / "data" / "processed" / "returns_broad_sharadar_2002_2024.parquet"
)
OUT_DIR = ROOT / "results" / "26_name_concentration"
TOP_N = 10
COST_BPS = 10.0


def renormalised_returns(
    weights: pd.DataFrame,
    next_returns: pd.DataFrame,
    drop_ticker: str | None,
) -> pd.Series:
    """Recompute monthly net returns after optionally dropping one ticker.

    Renormalises long and short legs separately on each rebalance date so
    the gross book remains 1.0× (matching the engine's pre-leverage
    convention). Charges 10 bps/side on L1 turnover.
    """
    w = weights.copy()
    if drop_ticker is not None and drop_ticker in w.columns:
        w = w.drop(columns=[drop_ticker])
    # Renormalise each leg per rebalance (long sum = 1, short sum = -1)
    longs = w.where(w > 0, 0.0)
    shorts = w.where(w < 0, 0.0)
    long_sum = longs.sum(axis=1).replace(0, np.nan)
    short_sum = shorts.sum(axis=1).abs().replace(0, np.nan)
    longs = longs.div(long_sum, axis=0).fillna(0.0)
    shorts = shorts.div(short_sum, axis=0).fillna(0.0)
    w_norm = longs + shorts

    # Realise next-month return per rebalance date
    rets = pd.Series(index=w_norm.index, dtype=float)
    prev_w = pd.Series(0.0, index=w_norm.columns)
    cost_rate = COST_BPS / 1e4
    for t in w_norm.index:
        if t not in next_returns.index:
            continue
        wt = w_norm.loc[t]
        rt = next_returns.loc[t].reindex(wt.index)
        valid = rt.dropna().index
        gross = float((wt.loc[valid] * rt.loc[valid]).sum())
        turnover = float((wt - prev_w.reindex(wt.index, fill_value=0.0)).abs().sum())
        rets.loc[t] = gross - cost_rate * turnover
        prev_w = wt
    return rets.dropna()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("Phase 26: name-concentration ablation on Phase 24-RT canonical")
    print("=" * 70)

    with open(CANON, "rb") as f:
        res = pickle.load(f)
    xgb = res["XGBoost"]
    weights = xgb.weights
    returns_wide = pd.read_parquet(RETURNS_FILE)
    next_returns = returns_wide.shift(-1)

    print(f"\n[1/3] Loaded canonical pkl + returns panel")
    print(f"  weights shape: {weights.shape}")
    print(f"  date range: {weights.index.min().date()} -> {weights.index.max().date()}")
    print(f"  tickers in weights matrix: {len(weights.columns)}")
    print(f"  NDAQ present? {'NDAQ' in weights.columns}")
    print(f"  IONQ present? {'IONQ' in weights.columns}")

    # Per-name lifetime P&L contribution
    print(f"\n[2/3] Computing per-name lifetime P&L contribution...")
    aligned_w = weights.copy()
    aligned_r = next_returns.reindex(index=aligned_w.index, columns=aligned_w.columns)
    monthly_contrib = (aligned_w * aligned_r).fillna(0.0)
    lifetime_pnl = monthly_contrib.sum(axis=0)
    n_months_active = (aligned_w != 0).sum(axis=0)
    contrib_df = pd.DataFrame({
        "lifetime_pnl_pct": lifetime_pnl * 100,
        "abs_lifetime_pnl_pct": lifetime_pnl.abs() * 100,
        "n_months_active": n_months_active,
    }).sort_values("abs_lifetime_pnl_pct", ascending=False)
    print(f"  Top-{TOP_N} contributors by |lifetime P&L|:")
    print(contrib_df.head(TOP_N).to_string())
    contrib_df.head(50).to_csv(OUT_DIR / "top_contributors.csv")

    # Baseline metrics (no name dropped)
    print(f"\n[3/3] Leave-one-out: drop each top contributor, recompute Sharpe...")
    base_rets = xgb.portfolio_returns.dropna()
    base_sharpe = sharpe_ratio(base_rets)
    base_ann = float(base_rets.mean() * 12)
    print(f"  Baseline: Sharpe = {base_sharpe:+.4f}, ann mean = {base_ann*100:+.2f}%/yr")
    print(f"  (Note: this baseline is the buggy-filter pkl; corrected canonical is +0.116 higher)")
    print()

    rows = [{
        "dropped": "(none) baseline",
        "sharpe": base_sharpe,
        "delta_sharpe": 0.0,
        "ann_mean_pct": base_ann * 100,
        "lifetime_pnl_pct_of_dropped": 0.0,
    }]
    top_tickers = contrib_df.head(TOP_N).index.tolist()
    for ticker in top_tickers:
        new_rets = renormalised_returns(weights, next_returns, drop_ticker=ticker)
        new_sharpe = sharpe_ratio(new_rets)
        new_ann = float(new_rets.mean() * 12)
        delta = new_sharpe - base_sharpe
        rows.append({
            "dropped": ticker,
            "sharpe": round(new_sharpe, 4),
            "delta_sharpe": round(delta, 4),
            "ann_mean_pct": round(new_ann * 100, 2),
            "lifetime_pnl_pct_of_dropped": round(float(contrib_df.loc[ticker, "lifetime_pnl_pct"]), 3),
        })
        print(f"  drop {ticker:8s}: Sharpe {new_sharpe:+.4f}  (Δ {delta:+.4f})  "
              f"name lifetime P&L = {contrib_df.loc[ticker, 'lifetime_pnl_pct']:+.2f}%")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_DIR / "leave_one_out.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'leave_one_out.csv'}")

    # Headline: max single-name fragility
    drop_only = out_df.iloc[1:]
    worst_drop = drop_only.loc[drop_only["delta_sharpe"].abs().idxmax()]
    print()
    print(f"HEADLINE: dropping the single highest-P&L name "
          f"({worst_drop['dropped']!r}) shifts Sharpe by "
          f"{worst_drop['delta_sharpe']:+.4f}.")
    print(f"For comparison, the Q-fix decomposition reported +0.042 Sharpe "
          f"attributable to IONQ alone.")
    if abs(worst_drop['delta_sharpe']) > 0.05:
        print(f"  -> Material single-name fragility detected "
              f"({worst_drop['dropped']!r} >0.05 Sharpe).")
    else:
        print(f"  -> No single name in the committed pkl shifts Sharpe by "
              f">0.05. IONQ's +0.042 contribution (from corrected filter) is "
              f"close to the upper bound on single-name fragility.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
