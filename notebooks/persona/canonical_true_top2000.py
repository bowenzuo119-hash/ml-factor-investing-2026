"""canonical_true_top2000.py - does the headline survive the REAL top-2000 PIT?

Audit finding: the committed canonical (Phase 24-RT) trades the feature panel's
per-month coverage (~4,400 names median, the alive survivorship-free union),
NOT the "top-2000 by market cap" the report claims. `load_universe_at` (the
true per-month top-2000 PIT filter) was only used to build the panel COLUMNS,
never as the trading universe.

This re-runs the 24-RT recipe (14 feats, q-filter, k=20, retuned 24a params)
UNCHANGED except eligible_universe_fn = the real top-2000-by-mcap at each date.
If Sharpe/alpha hold, the broad-vs-top-2000 distinction is cosmetic; if they
move materially, the report must either re-baseline or relabel the universe.

    KMP_DUPLICATE_LIB_OK=TRUE python -m notebooks.persona.canonical_true_top2000
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import run_walk_forward_backtest
from src.models import XGBoostModel
from src.data_loader import load_universe_at
from src.metrics import sharpe_ratio, max_drawdown, annualised_return
from notebooks.persona.verify_phase23_headline import fetch_ff5, nw_ols

ROOT = Path(__file__).resolve().parents[2]
RETURNS_FILE = ROOT / "data" / "processed" / "returns_broad_sharadar_2002_2024.parquet"
FEATURES_FILE = ROOT / "data" / "processed" / "features_broad_sharadar_with_chmom_maxret.parquet"
P24A = ROOT / "results" / "24a_retune_xgb_with_chmom" / "best_params.json"
P23A = ROOT / "results" / "23a_retune_broad_sharadar" / "best_params.json"

TRAIN_WINDOW, TEST_WINDOW = 120, 12
LONG_Q, SHORT_Q, TC_BPS, K = 0.8, 0.2, 10.0, 20
TARGET_KIND = "sector_relative"
TEST_START = pd.Timestamp("2019-01-01")
INCLUDE = ["mom", "rev", "mvol", "ivol", "log_mktcap", "bm", "ep", "dvol",
           "roe", "roa", "de", "asset_growth", "accruals", "chmom"]  # 24-RT's 14


def is_q(t):
    t = str(t).upper().strip()
    return len(t) >= 4 and t.endswith("Q")


def xgb():
    if P24A.exists():
        p = json.load(open(P24A))["best_params"]
    else:
        p = json.load(open(P23A))["by_model"]["xgboost"]["best_params"]
    return XGBoostModel(
        target_kind=TARGET_KIND,
        n_estimators=p.get("n_estimators", 200), max_depth=p.get("max_depth", 3),
        learning_rate=p.get("learning_rate", 0.0115), subsample=p.get("subsample", 0.717),
        colsample_bytree=p.get("colsample_bytree", 0.890),
        min_child_weight=p.get("min_child_weight", 11),
        reg_alpha=p.get("reg_alpha", 0.794), reg_lambda=p.get("reg_lambda", 2.305),
    )


def ff5_alpha(net):
    ff = fetch_ff5(); ff.index = ff.index.to_period("M")
    r = net.copy(); r.index = r.index.to_period("M")
    common = r.index.intersection(ff.index)
    r, f = r.loc[common], ff.loc[common]
    y = r.values - f["RF"].values
    X = np.column_stack([np.ones(len(y))] + [f[c].values for c in ["Mkt-RF","SMB","HML","RMW","CMA"]])
    b, _, t = nw_ols(y, X)
    return b[0]*12, t[0], b[1], b[2]  # alpha_ann, t, mkt_beta, smb_beta


_UC = {}
def top2000_at(d):
    key = pd.Timestamp(d).strftime("%Y-%m")
    if key not in _UC:
        _UC[key] = set(load_universe_at(d, top_n_by_marketcap=2000)["ticker"])
    return _UC[key]


def main() -> int:
    returns = pd.read_parquet(RETURNS_FILE)
    features = pd.read_parquet(FEATURES_FILE)[INCLUDE + ["sector"]]
    tk = features.index.get_level_values("ticker")
    features = features.loc[~pd.Series([is_q(t) for t in tk], index=features.index)]
    returns = returns.drop(columns=[c for c in returns.columns if is_q(c)])
    sector_map = features.reset_index().groupby("ticker")["sector"].first().to_dict()

    # diagnostic: how many names does top-2000 leave per month vs the panel
    fd = features.index.get_level_values("date")
    sizes = []
    for d in sorted(set(fd))[::24]:
        avail = set(features.loc[d].index)
        sizes.append((str(pd.Timestamp(d).date()), len(avail), len(top2000_at(d) & avail)))
    print("date | panel-coverage | top2000∩panel")
    for s in sizes:
        print(f"  {s[0]}: {s[1]} -> {s[2]}")

    print("\nRunning XGBoost walk-forward with TRUE top-2000 PIT universe...", flush=True)
    res = run_walk_forward_backtest(
        returns=returns, features=features, model=xgb(),
        train_window=TRAIN_WINDOW, test_window=TEST_WINDOW,
        long_quantile=LONG_Q, short_quantile=SHORT_Q, transaction_cost_bps=TC_BPS,
        regime_fn=lambda d: {"k_per_sector": K},
        sector_map=sector_map, eligible_universe_fn=top2000_at,
    )
    net = res.portfolio_returns.dropna()
    net_t = net[net.index >= TEST_START]
    a_ann, a_t, mb, sb = ff5_alpha(net)
    print("\n" + "=" * 66)
    print("TRUE top-2000 PIT vs committed 24-RT (broad ~4400/mo)")
    print("=" * 66)
    print(f"  {'metric':<16}{'TRUE top-2000':>16}{'24-RT (committed)':>20}")
    print(f"  {'Sharpe full':<16}{sharpe_ratio(net):>+16.3f}{'+1.08':>20}")
    print(f"  {'Sharpe test':<16}{sharpe_ratio(net_t):>+16.3f}{'+1.06':>20}")
    print(f"  {'ann ret / DD':<16}{annualised_return(net):>+15.1%}/{max_drawdown(net):>.0%}{'+33%/-35%':>20}")
    print(f"  {'FF5 alpha/yr':<16}{a_ann:>+15.1%}{'+18.2%':>20}")
    print(f"  {'FF5 alpha t':<16}{a_t:>+16.2f}{'+5.7':>20}")
    print(f"  {'Mkt-beta':<16}{mb:>+16.2f}{'+1.3':>20}")
    print(f"  {'SMB-beta':<16}{sb:>+16.2f}{'+1.1':>20}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
