"""regime_ablation_check.py - Does Person C's regime overlay help, and why not?

Run with:
    python -m notebooks.persona.regime_ablation_check

Holds the canonical Phase-15 setup fixed (2002-04 panel, 13 features, tuned
XGBoost, Layer-2 sector-relative target) and decomposes the regime overlay
into its two levers so we can attribute the effect:

  A. canonical        : k=5 always, leverage 1.0          (no overlay)
  B. full overlay (C) : k 5/2 by regime, leverage 1.0/0.4
  C. leverage-only    : k=5 always, leverage 1.0/0.4      (isolates de-levering)
  D. k-only           : k 5/2 by regime, leverage 1.0     (isolates concentration)

Plus diagnostics: the strategy's mean net return in calm vs crisis months,
and the regime label at the max-drawdown trough — these explain *why* the
overlay does what it does.

Regime lookup is period-matched (year-month) so the overlay fires on 100%
of months.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import run_walk_forward_backtest
from src.factors import build_feature_panel, load_sector_map
from src.models import XGBoostModel
from src.metrics import sharpe_ratio, max_drawdown, calmar_ratio, annualised_return

START, END = "2002-04-01", "2024-12-31"
TRAIN_WINDOW, TEST_WINDOW = 120, 12
LONG_Q, SHORT_Q = 0.8, 0.2
TC_BPS = 10.0
TARGET_KIND = "sector_relative"
INCLUDE = ("mom", "rev", "mvol", "ivol", "log_mktcap",
           "bm", "ep", "dvol", "roe", "roa", "de", "asset_growth", "accruals")
PANEL_FILE = "data/processed/returns_spliced_2002_2024.parquet"
OVERLAY = "results/regime_overlay_rules.csv"
TEST_START = "2019-01-01"


def _load_regime_by_period():
    """period -> (regime_str, leverage, k_per_sector) from the overlay CSV."""
    df = pd.read_csv(OVERLAY, parse_dates=["month_end"])
    out = {}
    for _, r in df.iterrows():
        out[pd.Period(pd.Timestamp(r["month_end"]), "M")] = (
            r["regime"], float(r["leverage"]), int(r["k_per_sector"])
        )
    return out


def main() -> int:
    print("Loading panel + features (cached)...")
    returns_wide = pd.read_parquet(PANEL_FILE)
    features = build_feature_panel(start=START, end=END, include=INCLUDE, sector_rank=True)
    sector_map = load_sector_map()
    reg = _load_regime_by_period()

    def _p(date):
        return reg.get(pd.Period(pd.Timestamp(date), "M"), ("calm", 1.0, 5))

    # 4 regime functions decomposing the overlay's two levers.
    fns = {
        "A. canonical (no overlay)": lambda d: {"k_per_sector": 5},
        "B. full overlay (C)":       lambda d: {"k_per_sector": _p(d)[2], "leverage": _p(d)[1]},
        "C. leverage-only":          lambda d: {"k_per_sector": 5, "leverage": _p(d)[1]},
        "D. k-only (concentration)": lambda d: {"k_per_sector": _p(d)[2]},
    }

    common = dict(
        returns=returns_wide, features=features,
        train_window=TRAIN_WINDOW, test_window=TEST_WINDOW,
        long_quantile=LONG_Q, short_quantile=SHORT_Q,
        transaction_cost_bps=TC_BPS, sector_map=sector_map,
    )

    rows, ret_A = [], None
    for label, rfn in fns.items():
        print(f"\nRunning XGBoost — {label} ...")
        res = run_walk_forward_backtest(model=XGBoostModel(target_kind=TARGET_KIND),
                                        regime_fn=rfn, **common)
        net = res.portfolio_returns
        if label.startswith("A"):
            ret_A = net
        rows.append({
            "config": label,
            "sharpe_full": round(sharpe_ratio(net), 3),
            "sharpe_test": round(sharpe_ratio(net[net.index >= TEST_START]), 3),
            "ann_ret": round(annualised_return(net), 4),
            "max_dd": round(max_drawdown(net), 4),
            "calmar": round(calmar_ratio(net), 3),
            "avg_lev": round(res.metadata.get("avg_leverage", 1.0), 3),
        })

    print("\n" + "=" * 84)
    print("REGIME ABLATION — XGBoost, identical except regime_fn")
    print("=" * 84)
    print(pd.DataFrame(rows).to_string(index=False))

    # --- Diagnostics on the canonical (no-overlay) return stream ---
    periods = ret_A.index.to_period("M")
    regimes = pd.Series([reg.get(p, ("calm", 1.0, 5))[0] for p in periods], index=ret_A.index)
    n_crisis = int((regimes == "crisis").sum())
    print("\n--- Why? Diagnostics on the canonical strategy's months ---")
    print(f"OOS months: {len(ret_A)}  (crisis-labelled: {n_crisis}, {100*n_crisis/len(ret_A):.0f}%)")
    print("Mean NET monthly return by regime (canonical, full leverage):")
    print(f"  calm  : {ret_A[regimes=='calm'].mean()*100:+.3f}%  ({(regimes=='calm').sum()} months)")
    print(f"  crisis: {ret_A[regimes=='crisis'].mean()*100:+.3f}%  ({n_crisis} months)")
    # max-drawdown trough month + its regime
    wealth = (1 + ret_A).cumprod()
    dd = wealth / wealth.cummax() - 1
    trough = dd.idxmin()
    print(f"Max-drawdown trough: {trough.date()} (dd={dd.min()*100:.1f}%), "
          f"regime there = {regimes.loc[trough]}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
