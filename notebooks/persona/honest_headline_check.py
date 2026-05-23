"""honest_headline_check.py - What is the survivorship-honest headline Sharpe?

Run with:
    python -m notebooks.persona.honest_headline_check

Runs the Phase-15 canonical recipe (2002 panel, 13 features, tuned XGBoost,
Layer-2 target, k=5) under three universe settings to pin down the honest
headline number that the survivorship fix exposed:

  A. no PIT (the old biased +1.49)        : eligible_universe_fn=None
  B. full PIT (train + trade)             : apply_pit_to_training=True
  C. train-full / trade-PIT (GKX-style)   : apply_pit_to_training=False

A vs B = total survivorship cost. B vs C = how much of the collapse is the
training-data restriction vs the trading restriction. C is not look-ahead
(training on past returns of any stock is fine; only *trading* non-members
was the leak), so if C holds up it is a defensible honest canonical.

(The hyperparameter-retune track is separate and slow; Person B runs it.)
"""

from __future__ import annotations

import pandas as pd

from src.backtest import run_walk_forward_backtest
from src.factors import build_feature_panel, load_sector_map
from src.models import XGBoostModel
from src.data_loader import load_sp500_membership
from src.metrics import sharpe_ratio, max_drawdown, annualised_return

START, END = "2002-04-01", "2024-12-31"
TRAIN_WINDOW, TEST_WINDOW = 120, 12
LONG_Q, SHORT_Q, TC_BPS, K = 0.8, 0.2, 10.0, 5
TARGET_KIND = "sector_relative"
INCLUDE = ("mom", "rev", "mvol", "ivol", "log_mktcap",
           "bm", "ep", "dvol", "roe", "roa", "de", "asset_growth", "accruals")
PANEL_FILE = "data/processed/returns_spliced_2002_2024.parquet"
TEST_START = "2019-01-01"

_UCACHE: dict = {}
def universe_at(date):
    key = pd.Timestamp(date).strftime("%Y-%m-%d")
    if key not in _UCACHE:
        _UCACHE[key] = set(load_sp500_membership(asof=key))
    return _UCACHE[key]


def _summary(res, label):
    net = res.portfolio_returns
    return {
        "config": label,
        "sharpe_full": round(sharpe_ratio(net), 3),
        "sharpe_test": round(sharpe_ratio(net[net.index >= TEST_START]), 3),
        "ann_ret_full": round(annualised_return(net), 4),
        "max_dd_full": round(max_drawdown(net), 4),
    }


def main() -> int:
    print("Loading panel + features (cached)...")
    returns_wide = pd.read_parquet(PANEL_FILE)
    features = build_feature_panel(start=START, end=END, include=INCLUDE, sector_rank=True)
    sector_map = load_sector_map()

    common = dict(
        returns=returns_wide, features=features,
        train_window=TRAIN_WINDOW, test_window=TEST_WINDOW,
        long_quantile=LONG_Q, short_quantile=SHORT_Q,
        transaction_cost_bps=TC_BPS, sector_map=sector_map,
        regime_fn=lambda d: {"k_per_sector": K},
    )
    configs = [
        ("A. no PIT (old biased)",        dict(eligible_universe_fn=None)),
        ("B. full PIT (train+trade)",     dict(eligible_universe_fn=universe_at, apply_pit_to_training=True)),
        ("C. train-full / trade-PIT",     dict(eligible_universe_fn=universe_at, apply_pit_to_training=False)),
    ]
    rows = []
    for label, kw in configs:
        print(f"\nRunning XGBoost — {label} ...")
        res = run_walk_forward_backtest(model=XGBoostModel(target_kind=TARGET_KIND), **common, **kw)
        rows.append(_summary(res, label))

    print("\n" + "=" * 80)
    print("HONEST HEADLINE — XGBoost, Phase-15 recipe, varying only the PIT setting")
    print("=" * 80)
    print(pd.DataFrame(rows).to_string(index=False))
    a, b, c = rows
    print("\nReadout:")
    print(f"  Survivorship cost (A->B full PIT): "
          f"Sharpe {a['sharpe_full']:+.2f} -> {b['sharpe_full']:+.2f} (full-OOS)")
    print(f"  Training-restriction effect (C vs B): "
          f"train-full {c['sharpe_full']:+.2f} vs train-PIT {b['sharpe_full']:+.2f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
