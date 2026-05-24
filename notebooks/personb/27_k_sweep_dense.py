"""27_k_sweep_dense.py - denser k-per-sector sweep on Phase 24-RT predictions.

Phase 24c earlier swept k in {1, 2, 3, 5, 7, 10, 15, 20, 30, 50} and locked
k=20. This re-runs the sweep on a much denser grid (every value 1..30 plus
selected larger values) so the optimum is read off a curve, not a 10-point
sample.

Method: post-process the canonical's PREDICTIONS (not weights), so each k
gets fresh top-/bottom-k picks per sector per rebalance. No model re-runs;
all values share the same trained model + features + universe + cost model.

Run with:
    .venv/bin/python -m notebooks.personb.27_k_sweep_dense
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import sharpe_ratio, max_drawdown, annualised_return


ROOT = Path(__file__).resolve().parents[2]
CANON = ROOT / "results" / "24_canonical_with_chmom" / "per_model_results.pkl"
PREDS = ROOT / "results" / "24_canonical_with_chmom" / "predictions.parquet"
RETURNS_FILE = ROOT / "data" / "processed" / "returns_broad_sharadar_2002_2024.parquet"
FEATURES_FILE = ROOT / "data" / "processed" / "features_broad_sharadar_with_chmom_maxret.parquet"
OUT_DIR = ROOT / "results" / "27_k_sweep_dense"

# Dense k grid: every value 1..30, then coarser steps to 100
K_GRID = list(range(1, 31)) + [35, 40, 45, 50, 60, 75, 100]

COST_BPS = 10.0
TEST_START = pd.Timestamp("2019-01-01")
LONG_START = pd.Timestamp("2015-01-01")


def build_weights_for_k(
    preds: pd.Series,
    sector_map: dict[str, str],
    eligible_at_date: dict[pd.Timestamp, set[str]],
    k: int,
) -> pd.DataFrame:
    """Pick top-k / bottom-k by sector per rebalance, dollar-neutral.

    Replicates the engine's sector-neutral construction:
    - longs.sum = +1.0, shorts.sum = -1.0 BEFORE leverage
    - per-sector cap: k = min(k, len(sector_names) // 2)
    """
    weights_records = []
    dates = sorted(preds.index.get_level_values("date").unique())
    for t in dates:
        scores_t = preds.loc[t].dropna()
        # Restrict to eligible universe at t
        elig = eligible_at_date.get(t, set())
        if elig:
            scores_t = scores_t[scores_t.index.isin(elig)]
        if scores_t.empty:
            continue
        # Group by sector, pick top-k / bottom-k each
        long_list, short_list = [], []
        sectors = pd.Series(
            {tk: sector_map.get(tk, "UNKNOWN") for tk in scores_t.index},
            name="sector",
        )
        for _sec, grp in scores_t.groupby(sectors):
            ranked = grp.sort_values(ascending=False)
            k_eff = min(int(k), len(ranked) // 2)
            if k_eff < 1:
                continue
            long_list.extend(ranked.head(k_eff).index.tolist())
            short_list.extend(ranked.tail(k_eff).index.tolist())
        w = pd.Series(0.0, index=scores_t.index)
        if long_list:
            w.loc[long_list] = 1.0 / len(long_list)
        if short_list:
            w.loc[short_list] = -1.0 / len(short_list)
        weights_records.append((t, w))
    # Pivot to wide
    all_tickers = sorted({tk for _, w in weights_records for tk in w.index})
    weights = pd.DataFrame(0.0, index=[t for t, _ in weights_records], columns=all_tickers)
    for t, w in weights_records:
        weights.loc[t, w.index] = w.values
    return weights


def portfolio_returns(
    weights: pd.DataFrame, next_returns: pd.DataFrame, cost_rate: float
) -> pd.Series:
    """Compute monthly net returns with 10 bps/side cost on L1 turnover."""
    rets = pd.Series(index=weights.index, dtype=float)
    prev_w = pd.Series(0.0, index=weights.columns)
    for t in weights.index:
        if t not in next_returns.index:
            continue
        wt = weights.loc[t]
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
    print("Phase 27: dense k-per-sector sweep on Phase 24-RT canonical predictions")
    print("=" * 70)

    print("\n[1/4] Loading predictions + returns + sector map...")
    # Predictions parquet is already Q-filtered (Bowen's corrected pkl); we
    # use the predictions' (date, ticker) index as the trusted universe and
    # skip re-applying is_bankruptcy_ticker (which would require the raw
    # Sharadar tickers.parquet that lives only on Bowen's machine).
    preds_all = pd.read_parquet(PREDS)
    preds_xgb = preds_all["XGBoost"].copy()
    preds_xgb.index = preds_xgb.index.set_names(["date", "ticker"])
    valid_tickers = set(preds_xgb.index.get_level_values("ticker"))
    print(f"  predictions: {len(preds_xgb):,} (date, ticker) rows, "
          f"{len(valid_tickers):,} unique tickers")

    returns_wide = pd.read_parquet(RETURNS_FILE)
    keep_cols = [c for c in returns_wide.columns if c in valid_tickers]
    returns_wide = returns_wide[keep_cols]
    next_returns = returns_wide.shift(-1)
    print(f"  returns: {returns_wide.shape} after restricting to prediction-universe")

    features = pd.read_parquet(FEATURES_FILE)
    ft_all = features.index.get_level_values("ticker")
    features = features.loc[ft_all.isin(valid_tickers)]
    sector_map = features.reset_index().groupby("ticker")["sector"].first().to_dict()
    print(f"  sector map: {len(sector_map):,} tickers, "
          f"{pd.Series(list(sector_map.values())).nunique()} sectors")

    fd = features.index.get_level_values("date")
    ft = features.index.get_level_values("ticker")
    elig = {d: set(ft[fd == d].unique()) for d in fd.unique()}
    print(f"  eligible universe map built for {len(elig)} rebalance dates")

    print(f"\n[2/4] Running k-sweep across {len(K_GRID)} values...")
    cost_rate = COST_BPS / 1e4
    rows = []
    for i, k in enumerate(K_GRID):
        weights = build_weights_for_k(preds_xgb, sector_map, elig, k)
        net = portfolio_returns(weights, next_returns, cost_rate)
        # 3 windows
        full = net
        long_oos = net[net.index >= LONG_START]
        test = net[net.index >= TEST_START]
        row = {
            "k": k,
            "n_pos_per_rebal": int((weights != 0).sum(axis=1).median()),
            "sharpe_full": sharpe_ratio(full),
            "sharpe_long": sharpe_ratio(long_oos),
            "sharpe_test": sharpe_ratio(test),
            "ann_full": annualised_return(full),
            "mdd_full": max_drawdown(full),
        }
        rows.append(row)
        print(f"  [{i+1:2d}/{len(K_GRID)}] k={k:3d}  pos/rebal={row['n_pos_per_rebal']:4d}  "
              f"Sh(full)={row['sharpe_full']:+.3f}  Sh(long)={row['sharpe_long']:+.3f}  "
              f"Sh(test)={row['sharpe_test']:+.3f}  ann={row['ann_full']*100:+5.1f}%  "
              f"MDD={row['mdd_full']*100:+5.1f}%", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "sweep_metrics.csv", index=False)
    print(f"\n[3/4] Wrote {OUT_DIR / 'sweep_metrics.csv'}")

    # Plot
    print("\n[4/4] Generating sweep figure...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for col, label, color in [
        ("sharpe_full", "Full-OOS 2012-2024", "#1F3864"),
        ("sharpe_long", "Long-OOS 2015-2024", "#22C55E"),
        ("sharpe_test", "Test-OOS 2019-2024", "#DC2626"),
    ]:
        ax.plot(df["k"], df[col], "o-", label=label, color=color, lw=1.7,
                markersize=4.5)
        peak_k = int(df.loc[df[col].idxmax(), "k"])
        peak_sh = float(df[col].max())
        ax.annotate(f"k*={peak_k}, Sh={peak_sh:+.2f}",
                    xy=(peak_k, peak_sh),
                    xytext=(peak_k + 2, peak_sh + 0.02),
                    fontsize=8, color=color)
    ax.axvline(20, color="grey", ls="--", lw=0.7, alpha=0.6, label="k=20 (canonical lock)")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("k (long/short picks per GICS sector)", fontsize=11)
    ax.set_ylabel("Sharpe ratio (10 bps/side)", fontsize=11)
    ax.set_title("Phase 27 — dense k-per-sector sweep on Phase 24-RT canonical\n"
                 "Three OOS windows: full / long-OOS / test", fontsize=11.5, weight="bold")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "k_sweep_dense.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'k_sweep_dense.png'}")

    # Headline
    print()
    print("=" * 70)
    print("HEADLINE: per-window optimal k")
    print("=" * 70)
    for col, label in [
        ("sharpe_full", "Full-OOS"),
        ("sharpe_long", "Long-OOS"),
        ("sharpe_test", "Test-OOS"),
    ]:
        peak_row = df.loc[df[col].idxmax()]
        k_canon = df.loc[df["k"] == 20].iloc[0]
        print(f"  {label:8s} optimal k = {int(peak_row['k']):3d}  Sh={peak_row[col]:+.3f}  "
              f"(vs k=20 canonical: Sh={k_canon[col]:+.3f}, "
              f"Δ={k_canon[col]-peak_row[col]:+.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
