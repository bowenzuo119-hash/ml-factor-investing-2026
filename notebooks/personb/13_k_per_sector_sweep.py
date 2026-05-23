"""Phase 13: sensitivity sweep on k_per_sector for the canonical Phase 12 model.

Re-derives portfolios from Phase 12's saved predictions at different
k_per_sector values (3, 5, 7, 10, 15, 20) -- no re-fitting needed. Plots
Sharpe / annualised return / max drawdown vs k so the report can defend
the chosen k=10 with an empirical sensitivity curve rather than an
arbitrary pick.

For each k, the portfolio construction is:
  * At each rebalance, group stocks by GICS sector (from load_sector_map())
  * Within each sector, pick top-k by score (longs) and bottom-k by score (shorts)
  * Equal-weight within each leg, dollar-neutral overall
  * 10 bps transaction cost charged on L1 turnover at each rebalance
  * Monthly returns realised at t+1 using the spliced 2003-2024 returns panel

Run with:
    .venv/bin/python -m notebooks.personb.13_k_per_sector_sweep
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.factors import load_sector_map
from src.metrics import summary_stats


PHASE_DIR = (
    Path(__file__).resolve().parents[2] / "results" / "12_official_canonical"
)
RESULTS_DIR = (
    Path(__file__).resolve().parents[2] / "results" / "13_k_per_sector_sweep"
)
PANEL_FILE = (
    Path(__file__).resolve().parents[2] / "data" / "processed"
    / "returns_spliced_2003_2024.parquet"
)

# Test window per Framework section 7.2 (and Phase 12's canonical sample)
TEST_START = pd.Timestamp("2019-01-01")
TEST_END = pd.Timestamp("2024-12-31")

K_VALUES = (3, 5, 7, 10, 15, 20)
COST_BPS = 10.0
MODELS = ("Lasso", "XGBoost", "NN")


def build_portfolio_returns(
    preds: pd.Series,
    returns_wide: pd.DataFrame,
    sector_map: dict[str, str],
    k: int,
    cost_bps: float = COST_BPS,
) -> pd.Series:
    """For one model's predictions and a chosen k, return the net monthly
    portfolio return series (date-indexed).

    Construction at each rebalance date t:
      1. Group t's predictions by sector_map[ticker].
      2. Within each sector, pick top-k by score (longs) and bottom-k (shorts).
      3. Equal-weight within each leg: long weight = 1 / n_longs,
         short weight = -1 / n_shorts.
      4. Realise return: sum(weights * next-month return). 10 bps cost on
         the L1 turnover vs the previous rebalance.
    """
    next_returns = returns_wide.shift(-1)
    rebal_dates = sorted(
        preds.index.get_level_values("date").unique()
    )
    cost_rate = cost_bps / 10_000.0

    prev_weights = pd.Series(dtype=float)
    records: list[tuple[pd.Timestamp, float]] = []

    for t in rebal_dates:
        try:
            cs = preds.xs(t, level="date")
        except KeyError:
            continue
        # Map each ticker to its sector; group and pick top-k / bottom-k.
        cs_df = pd.DataFrame({
            "score": cs.values,
            "sector": [sector_map.get(str(tk).upper(), "UNKNOWN") for tk in cs.index],
        }, index=cs.index)
        cs_df = cs_df.dropna(subset=["score"])

        longs: list[str] = []
        shorts: list[str] = []
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

        # Realise next-period return
        if t not in next_returns.index:
            prev_weights = weights
            continue
        rets_t = next_returns.loc[t].reindex(weights.index)
        valid = weights.index.intersection(rets_t.dropna().index)
        gross = float((weights.loc[valid] * rets_t.loc[valid]).sum())

        # Transaction cost on L1 turnover
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
    print("Phase 13: k_per_sector sensitivity sweep on Phase 12 predictions")
    print("=" * 72)
    print(f"  source: {PHASE_DIR.name}")
    print(f"  k values: {K_VALUES}")
    print(f"  cost: {COST_BPS} bps per L1 unit of turnover")
    print(f"  test window: {TEST_START.date()} -> {TEST_END.date()}")

    print("\n[1/3] Loading predictions + returns + sector map...")
    preds_wide = pd.read_parquet(PHASE_DIR / "predictions.parquet")
    returns_wide = pd.read_parquet(PANEL_FILE)
    sector_map = load_sector_map()
    print(f"  predictions: {preds_wide.shape}, returns: {returns_wide.shape}")

    print(f"\n[2/3] Reconstructing portfolios at {len(K_VALUES)} k values "
          f"× {len(MODELS)} models = {len(K_VALUES) * len(MODELS)} runs...")

    rows = []
    by_model_k: dict[tuple[str, int], pd.Series] = {}
    for model in MODELS:
        if model not in preds_wide.columns:
            continue
        preds = preds_wide[model].dropna()
        # Some models predict NaN for some (date, ticker) -- preds is now
        # only the populated rows.
        for k in K_VALUES:
            rets = build_portfolio_returns(preds, returns_wide, sector_map, k)
            by_model_k[(model, k)] = rets

            # Metrics on both windows
            for window_name, mask in [
                ("full_oos", pd.Series(True, index=rets.index)),
                ("test_only",
                 (rets.index >= TEST_START) & (rets.index <= TEST_END)),
            ]:
                slc = rets[mask]
                if len(slc) < 2:
                    continue
                stats = summary_stats(slc)
                rows.append({
                    "model": model,
                    "k": k,
                    "window": window_name,
                    "n_months": len(slc),
                    "sharpe": stats["sharpe_ratio"],
                    "ann_return": stats["annualised_return"],
                    "max_dd": stats["max_drawdown"],
                    "vol": stats["annualised_volatility"],
                })
            print(f"  {model:8s} k={k:2d}: "
                  f"test Sharpe={rows[-1]['sharpe']:+.3f}, "
                  f"long Sharpe={rows[-2]['sharpe']:+.3f}")

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_parquet(RESULTS_DIR / "sweep_metrics.parquet")

    # Print summary table
    print("\n[3/3] Summary - test window 2019-2024")
    print("=" * 72)
    test = metrics_df[metrics_df["window"] == "test_only"].copy()
    pivot = test.pivot_table(index="k", columns="model", values="sharpe")
    print(pivot.round(3).to_string())
    print()
    pivot_dd = test.pivot_table(index="k", columns="model", values="max_dd")
    print("Max drawdown (test):")
    print(pivot_dd.round(3).to_string())

    # Plot: Sharpe vs k for each model
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"Lasso": "#9CA3AF", "XGBoost": "#DC2626", "NN": "#3B82F6"}
    for ax, window in zip(axes, ("test_only", "full_oos")):
        df = metrics_df[metrics_df["window"] == window]
        for model in MODELS:
            sub = df[df["model"] == model].sort_values("k")
            ax.plot(sub["k"], sub["sharpe"], "-o",
                    color=colors.get(model, "black"),
                    label=model, linewidth=2, markersize=8)
        ax.set_xlabel("k (stocks per sector per leg)")
        ax.set_ylabel("Net Sharpe ratio")
        ax.set_title(
            "Test window 2019-2024" if window == "test_only" else "Long-OOS"
        )
        ax.grid(alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.legend(loc="best")
        ax.axvline(10, color="grey", linestyle=":", alpha=0.5,
                   label="current canonical")
    fig.suptitle(
        "k_per_sector sensitivity (Phase 12 predictions)",
        fontsize=12, weight="bold",
    )
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "sharpe_vs_k.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {RESULTS_DIR.name}/sweep_metrics.parquet + sharpe_vs_k.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
