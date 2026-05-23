"""regime_overlay_pit_check.py - Does C's leverage-only overlay help in the PIT world?

The +1.50 -> +1.56 leverage win was measured on the OLD (no-PIT, biased)
canonical. This re-checks the leverage-only overlay
(``results/regime_overlay_rules.csv``: calm lev=1.0 / crisis lev=0.4, k=5 fixed)
on the survivorship-honest canonical (train-full / trade-PIT,
``apply_pit_to_training=False``) to see whether it still buys a drawdown
improvement once the look-ahead is removed.

Both runs are identical except for the regime overlay:
  baseline : regime_fn = lambda d: {"k_per_sector": 5}   (leverage 1.0 always)
  overlay  : regime_fn = leverage from CSV + k=5          (leverage 0.4 in crisis)

The CSV's long/short_quantile columns are intentionally ignored: with a
sector_map + k_per_sector the engine takes the sector-neutral top-k/bottom-k
path (backtest.py), so quantiles are inert. We pass only leverage + k=5 to
isolate the one lever C's overlay actually moves.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE python -m notebooks.persona.regime_overlay_pit_check
"""

from __future__ import annotations

import pandas as pd

from src.backtest import run_walk_forward_backtest
from src.factors import build_feature_panel, load_sector_map
from src.models import XGBoostModel
from src.data_loader import load_sp500_membership
from src.regime import make_regime_fn
from src.metrics import sharpe_ratio, max_drawdown, annualised_return

START, END = "2002-04-01", "2024-12-31"
TRAIN_WINDOW, TEST_WINDOW = 120, 12
LONG_Q, SHORT_Q, TC_BPS, K = 0.8, 0.2, 10.0, 5
TARGET_KIND = "sector_relative"
INCLUDE = ("mom", "rev", "mvol", "ivol", "log_mktcap",
           "bm", "ep", "dvol", "roe", "roa", "de", "asset_growth", "accruals")
PANEL_FILE = "data/processed/returns_spliced_2002_2024.parquet"
OVERLAY_CSV = "results/regime_overlay_rules.csv"
TEST_START = "2019-01-01"

_UCACHE: dict = {}
def universe_at(date):
    key = pd.Timestamp(date).strftime("%Y-%m-%d")
    if key not in _UCACHE:
        _UCACHE[key] = set(load_sp500_membership(asof=key))
    return _UCACHE[key]


def _summary(res, label):
    net = res.portfolio_returns
    net_test = net[net.index >= TEST_START]
    lev = res.leverage
    return {
        "config": label,
        "sharpe_full": round(sharpe_ratio(net), 3),
        "sharpe_test": round(sharpe_ratio(net_test), 3),
        "ann_ret_full": round(annualised_return(net), 4),
        "maxdd_full": round(max_drawdown(net), 4),
        "maxdd_test": round(max_drawdown(net_test), 4),
        "mean_lev": round(float(lev.mean()), 3),
    }


def main() -> int:
    print("Loading panel + features (cached)...")
    returns_wide = pd.read_parquet(PANEL_FILE)
    features = build_feature_panel(start=START, end=END, include=INCLUDE, sector_rank=True)
    sector_map = load_sector_map()

    base_overlay = make_regime_fn(OVERLAY_CSV)

    def no_overlay_fn(d):
        return {"k_per_sector": K}  # leverage defaults to 1.0 every month

    def leverage_only_fn(d):
        # Only leverage varies by regime; breadth (k) held fixed at K.
        return {"leverage": float(base_overlay(d).get("leverage", 1.0)),
                "k_per_sector": K}

    common = dict(
        returns=returns_wide, features=features,
        train_window=TRAIN_WINDOW, test_window=TEST_WINDOW,
        long_quantile=LONG_Q, short_quantile=SHORT_Q,
        transaction_cost_bps=TC_BPS, sector_map=sector_map,
        eligible_universe_fn=universe_at, apply_pit_to_training=False,  # honest canonical
    )
    configs = [
        ("PIT canonical, NO overlay (lev=1.0)", no_overlay_fn),
        ("PIT canonical + leverage-only overlay", leverage_only_fn),
    ]
    rows = []
    for label, rfn in configs:
        print(f"\nRunning XGBoost — {label} ...")
        res = run_walk_forward_backtest(model=XGBoostModel(target_kind=TARGET_KIND),
                                        regime_fn=rfn, **common)
        rows.append(_summary(res, label))

    print("\n" + "=" * 82)
    print("REGIME OVERLAY IN THE PIT WORLD — XGBoost, train-full/trade-PIT, only overlay varies")
    print("=" * 82)
    print(pd.DataFrame(rows).to_string(index=False))
    base, ov = rows
    print("\nReadout:")
    print(f"  Sharpe (full):  no-overlay {base['sharpe_full']:+.2f} -> overlay {ov['sharpe_full']:+.2f}")
    print(f"  Sharpe (test):  no-overlay {base['sharpe_test']:+.2f} -> overlay {ov['sharpe_test']:+.2f}")
    print(f"  Max DD (full):  no-overlay {base['maxdd_full']:+.1%} -> overlay {ov['maxdd_full']:+.1%}")
    print(f"  Max DD (test):  no-overlay {base['maxdd_test']:+.1%} -> overlay {ov['maxdd_test']:+.1%}")
    print(f"  Mean leverage:  no-overlay {base['mean_lev']:.2f} -> overlay {ov['mean_lev']:.2f} "
          f"(overlay < 1.0 confirms crisis de-risking engaged)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
