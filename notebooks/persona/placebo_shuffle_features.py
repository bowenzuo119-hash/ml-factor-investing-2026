"""placebo_shuffle_features.py - is the +1.15 alpha real, or a leakage artifact?

The definitive feature-leakage test. Runs the EXACT 14-feature canonical recipe
(q-filter, PIT, k=20, 24a params) but PERMUTES the feature vectors across
tickers WITHIN each rebalance date -- ticker i is handed a random other ticker's
features that month. Sector + the (date,ticker) index stay real, so the engine,
universe, target, and cost machinery are untouched; only the feature->ticker
mapping is destroyed.

If the strategy still makes money on scrambled features, the "signal" is an
artifact (engine/target leakage or feature look-ahead). If Sharpe collapses to
~0, the genuine +1.15 comes from real cross-sectional predictive content. Run a
few seeds to be sure.

    KMP_DUPLICATE_LIB_OK=TRUE python -m notebooks.persona.placebo_shuffle_features
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import run_walk_forward_backtest
from src.models import XGBoostModel
from src.data_loader import is_bankruptcy_ticker
from src.metrics import sharpe_ratio, max_drawdown, annualised_return

ROOT = Path(__file__).resolve().parents[2]
RETURNS_FILE = ROOT / "data" / "processed" / "returns_broad_sharadar_2002_2024.parquet"
FEATURES_FILE = ROOT / "data" / "processed" / "features_broad_sharadar_with_chmom_maxret.parquet"
P24A = ROOT / "results" / "24a_retune_xgb_with_chmom" / "best_params.json"

TRAIN_WINDOW, TEST_WINDOW = 120, 12
LONG_Q, SHORT_Q, TC_BPS, K = 0.8, 0.2, 10.0, 20
TARGET_KIND = "sector_relative"
INCLUDE = ["mom", "rev", "mvol", "ivol", "log_mktcap", "bm", "ep", "dvol",
           "roe", "roa", "de", "asset_growth", "accruals", "chmom"]
SEEDS = [0, 1]


def xgb():
    p = json.load(open(P24A))["best_params"]
    return XGBoostModel(
        target_kind=TARGET_KIND,
        n_estimators=p.get("n_estimators", 200), max_depth=p.get("max_depth", 3),
        learning_rate=p.get("learning_rate", 0.0115), subsample=p.get("subsample", 0.717),
        colsample_bytree=p.get("colsample_bytree", 0.890),
        min_child_weight=p.get("min_child_weight", 11),
        reg_alpha=p.get("reg_alpha", 0.794), reg_lambda=p.get("reg_lambda", 2.305),
    )


def shuffle_features_within_date(feats: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Permute the feature-vector rows across tickers within each date."""
    rng = np.random.default_rng(seed)
    out = feats.copy()
    vals = out[INCLUDE].to_numpy()
    dates = out.index.get_level_values("date")
    for d in dates.unique():
        pos = np.where(dates == d)[0]
        vals[pos] = vals[pos][rng.permutation(len(pos))]
    out[INCLUDE] = vals
    return out


def run(feats, returns, sector_map, umap):
    res = run_walk_forward_backtest(
        returns=returns, features=feats, model=xgb(),
        train_window=TRAIN_WINDOW, test_window=TEST_WINDOW,
        long_quantile=LONG_Q, short_quantile=SHORT_Q, transaction_cost_bps=TC_BPS,
        regime_fn=lambda d: {"k_per_sector": K},
        sector_map=sector_map, eligible_universe_fn=lambda d: umap.get(pd.Timestamp(d), set()),
    )
    net = res.portfolio_returns.dropna()
    return sharpe_ratio(net), annualised_return(net), max_drawdown(net)


def main() -> int:
    returns = pd.read_parquet(RETURNS_FILE)
    features = pd.read_parquet(FEATURES_FILE)[INCLUDE + ["sector"]]
    tk = features.index.get_level_values("ticker")
    features = features.loc[~pd.Series([is_bankruptcy_ticker(t) for t in tk], index=features.index)]
    returns = returns.drop(columns=[c for c in returns.columns if is_bankruptcy_ticker(c)])
    sector_map = features.reset_index().groupby("ticker")["sector"].first().to_dict()
    fd = features.index.get_level_values("date"); ft = features.index.get_level_values("ticker")
    umap = {d: set(ft[fd == d].unique()) for d in fd.unique()}

    print("Baseline (real features):", flush=True)
    s0, a0, d0 = run(features, returns, sector_map, umap)
    print(f"  REAL features    -> Sharpe {s0:+.3f}, ann {a0:+.1%}, DD {d0:+.0%}")

    print("\nPlacebo (features shuffled across tickers within each date):", flush=True)
    rows = []
    for seed in SEEDS:
        fs = shuffle_features_within_date(features, seed)
        s, a, dd = run(fs, returns, sector_map, umap)
        rows.append(s)
        print(f"  shuffled seed={seed} -> Sharpe {s:+.3f}, ann {a:+.1%}, DD {dd:+.0%}")

    print("\n" + "=" * 60)
    print(f"  REAL features Sharpe : {s0:+.3f}")
    print(f"  shuffled mean Sharpe : {np.mean(rows):+.3f}  (range {min(rows):+.3f}..{max(rows):+.3f})")
    # Only a POSITIVE placebo Sharpe signals leakage. A placebo that collapses
    # to ~0 or goes negative (no signal + cost drag) is exactly what a genuine
    # edge looks like -- the profit requires the real feature->ticker mapping.
    verdict = ("SUSPICIOUS -- placebo still profits (leakage)" if np.mean(rows) > 0.3
               else "REAL signal -- placebo does NOT profit; the +1.15 needs genuine features")
    print(f"  verdict: {verdict}")
    print(f"  REAL minus placebo = {s0 - np.mean(rows):+.2f} Sharpe of genuine feature content")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
