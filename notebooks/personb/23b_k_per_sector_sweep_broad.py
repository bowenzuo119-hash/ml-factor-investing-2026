"""Phase 23b: k_per_sector sensitivity sweep on Phase 23's predictions.

Re-derives portfolios from Phase 23's saved predictions at different k values
(3, 5, 7, 10, 15, 20) -- no re-fitting needed. Tells us the empirical
optimum k on the broader Sharadar universe. Phase 13's analog was on
the smaller S&P-500 panel; this re-runs on the broader Russell-1500-like
universe where 11 sectors × 5-20 names per leg is the relevant range.

Uses the same `build_portfolio_returns` logic as Phase 13.

Run after Phase 23 has produced predictions.parquet.

    .venv/bin/python -m notebooks.personb.23b_k_per_sector_sweep_broad
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.metrics import summary_stats


PHASE_DIR = (
    Path(__file__).resolve().parents[2] / "results" / "23_canonical_broad_sharadar"
)
RESULTS_DIR = (
    Path(__file__).resolve().parents[2] / "results" / "23b_k_per_sector_sweep_broad"
)
RETURNS_FILE = (
    Path(__file__).resolve().parents[2] / "data" / "processed"
    / "returns_broad_sharadar_2002_2024.parquet"
)
FEATURES_FILE = (
    Path(__file__).resolve().parents[2] / "data" / "processed"
    / "features_broad_sharadar_2002_2024.parquet"
)

TEST_START = pd.Timestamp("2019-01-01")
TEST_END = pd.Timestamp("2024-12-31")

K_VALUES = (1, 2, 3, 5, 7, 10, 15, 20, 30)
# Broader universe (~180 names per GICS sector) lets us go BOTH ways:
# * Smaller k (1-2): concentrated bets on model's top-conviction pick per
#   sector. Highest exposure to model skill if model is accurate, highest
#   single-name risk if not.
# * Larger k (15-30): diversified, market-beta-leaning, lower idiosyncratic.
# At Phase 13's S&P-500 scale (~45/sector) only k=3-20 was meaningful;
# here k=1/2 is interesting (top-pick-per-sector strategy) and k=50 isn't
# (would burn 1/4 of available alpha into the middling 1/4 of each sector).
COST_BPS = 10.0
MODELS = ("Lasso", "XGBoost", "NN")


def build_portfolio_returns(preds, returns_wide, sector_map, k, cost_bps=COST_BPS):
    next_returns = returns_wide.shift(-1)
    rebal_dates = sorted(preds.index.get_level_values("date").unique())
    cost_rate = cost_bps / 10_000.0
    prev_weights = pd.Series(dtype=float)
    records = []

    for t in rebal_dates:
        try:
            cs = preds.xs(t, level="date")
        except KeyError:
            continue
        cs_df = pd.DataFrame({
            "score": cs.values,
            "sector": [sector_map.get(str(tk).upper(), "UNKNOWN") for tk in cs.index],
        }, index=cs.index).dropna(subset=["score"])

        longs, shorts = [], []
        for sec, grp in cs_df.groupby("sector", sort=False):
            ranked = grp["score"].sort_values(ascending=False)
            longs.extend(ranked.head(k).index.tolist())
            shorts.extend(ranked.tail(k).index.tolist())
        if not longs or not shorts:
            prev_weights = pd.Series(dtype=float)
            continue

        weights = pd.Series(0.0, index=cs.index)
        weights.loc[longs] = 1.0 / len(longs)
        weights.loc[shorts] = -1.0 / len(shorts)

        if t not in next_returns.index:
            prev_weights = weights
            continue
        rets_t = next_returns.loc[t].reindex(weights.index)
        valid = weights.index.intersection(rets_t.dropna().index)
        gross = float((weights.loc[valid] * rets_t.loc[valid]).sum())

        union = weights.index.union(prev_weights.index)
        w_now = weights.reindex(union, fill_value=0.0)
        w_prev = prev_weights.reindex(union, fill_value=0.0)
        turnover = float((w_now - w_prev).abs().sum())
        cost = cost_rate * turnover
        records.append((t, gross - cost))
        prev_weights = weights

    return pd.Series(dict(records)).sort_index()


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("Phase 23b: k_per_sector sensitivity on broad Sharadar predictions")
    print("=" * 72)
    print(f"  k values: {K_VALUES}")

    preds_wide = pd.read_parquet(PHASE_DIR / "predictions.parquet")
    returns_wide = pd.read_parquet(RETURNS_FILE)
    features = pd.read_parquet(FEATURES_FILE)
    # sector_map from features (SIC fallback applied)
    sector_map = (
        features.reset_index().groupby("ticker")["sector"].first().to_dict()
    )
    print(f"  predictions: {preds_wide.shape}, returns: {returns_wide.shape}")
    print(f"  sector_map size: {len(sector_map)}")

    rows = []
    for model in MODELS:
        if model not in preds_wide.columns:
            continue
        preds = preds_wide[model].dropna()
        for k in K_VALUES:
            rets = build_portfolio_returns(preds, returns_wide, sector_map, k)
            for win, mask in [
                ("full_oos", pd.Series(True, index=rets.index)),
                ("test_only", (rets.index >= TEST_START) & (rets.index <= TEST_END)),
            ]:
                slc = rets[mask]
                if len(slc) < 2:
                    continue
                stats = summary_stats(slc)
                rows.append({
                    "model": model, "k": k, "window": win,
                    "n_months": len(slc),
                    "sharpe": stats["sharpe_ratio"],
                    "ann_return": stats["annualised_return"],
                    "max_dd": stats["max_drawdown"],
                    "vol": stats["annualised_volatility"],
                })
            print(f"  {model:8s} k={k:2d}: test Sh={rows[-1]['sharpe']:+.3f}  "
                  f"long Sh={rows[-2]['sharpe']:+.3f}")

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_parquet(RESULTS_DIR / "sweep_metrics.parquet")

    print("\nSummary -- test window 2019-2024")
    print("=" * 72)
    test = metrics_df[metrics_df["window"] == "test_only"].copy()
    pivot = test.pivot_table(index="k", columns="model", values="sharpe")
    print(pivot.round(3).to_string())
    print()
    print("Summary -- long-OOS 2012-2024")
    print("=" * 72)
    long = metrics_df[metrics_df["window"] == "full_oos"].copy()
    pivot = long.pivot_table(index="k", columns="model", values="sharpe")
    print(pivot.round(3).to_string())

    # Plot Sharpe vs k per model
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, win in zip(axes, ("test_only", "full_oos")):
        sub = metrics_df[metrics_df["window"] == win]
        for model in MODELS:
            mm = sub[sub["model"] == model]
            ax.plot(mm["k"], mm["sharpe"], "o-", label=model, lw=1.7)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(f"Sharpe vs k_per_sector ({win})")
        ax.set_xlabel("k_per_sector")
        ax.set_ylabel("Net Sharpe")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "sharpe_vs_k.png", dpi=180)
    print(f"\nWrote {RESULTS_DIR}/sweep_metrics.parquet + sharpe_vs_k.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
