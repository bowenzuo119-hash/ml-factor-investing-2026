"""canonical_broad_16feat.py - 24b-equivalent: all 16 features, original tune.

Person A's un-blocked stand-in for B's Phase 24b (which retunes XGBoost on the
16-feature panel and is running on B's machine, not yet pushed). This runs the
SAME 23g broad recipe (Q-filter, PIT, k=20, walk-forward) but with the FULL
16-feature set (13 base + chmom + maxret + mom36m) and the ORIGINAL 23a tune --
no Optuna re-tune (that's B's slow part; we don't duplicate it). So it's the
"orig-tune" A/B: does adding the 3 new GKX features move the headline, and is
the result still cost-robust?

Reports net Sharpe vs the 13-feature 23g baseline (1.068), FF5 alpha, and a
cost grid for the 30-bps headline.

    KMP_DUPLICATE_LIB_OK=TRUE python -m notebooks.persona.canonical_broad_16feat
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import run_walk_forward_backtest
from src.models import XGBoostModel
from src.metrics import sharpe_ratio, max_drawdown, annualised_return
from notebooks.persona.verify_phase23_headline import fetch_ff5, nw_ols

ROOT = Path(__file__).resolve().parents[2]
RETURNS_FILE = ROOT / "data" / "processed" / "returns_broad_sharadar_2002_2024.parquet"
FEATURES_FILE = ROOT / "data" / "processed" / "features_broad_sharadar_with_chmom_maxret.parquet"
PARAMS_FILE = ROOT / "results" / "23a_retune_broad_sharadar" / "best_params.json"

TRAIN_WINDOW, TEST_WINDOW = 120, 12
LONG_Q, SHORT_Q, TC_BPS, K = 0.8, 0.2, 10.0, 20
TARGET_KIND = "sector_relative"
TEST_START = pd.Timestamp("2019-01-01")
BASE_13 = ["mom", "rev", "mvol", "ivol", "log_mktcap", "bm", "ep", "dvol",
           "roe", "roa", "de", "asset_growth", "accruals"]
NEW = ["chmom", "maxret", "mom36m"]
INCLUDE = BASE_13 + NEW   # 16 features
BPS_GRID = [10, 20, 30, 50]
SHARPE_13F = 1.068  # 23g 13-feature baseline (from the ablation)


def is_q(t):  # delegates to the centralized, isdelisted-gated filter
    from src.data_loader import is_bankruptcy_ticker
    return is_bankruptcy_ticker(t)


def xgb():
    p = json.load(open(PARAMS_FILE))["by_model"]["xgboost"]["best_params"]
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
    X = np.column_stack([np.ones(len(y))] + [f[c].values
                         for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]])
    beta, _, t = nw_ols(y, X)
    return beta[0] * 12, t[0]


def main() -> int:
    returns = pd.read_parquet(RETURNS_FILE)
    features = pd.read_parquet(FEATURES_FILE)[INCLUDE + ["sector"]]
    tk = features.index.get_level_values("ticker")
    features = features.loc[~pd.Series([is_q(t) for t in tk], index=features.index)]
    returns = returns.drop(columns=[c for c in returns.columns if is_q(c)])
    print(f"features {features.shape} ({len(INCLUDE)} feats), returns {returns.shape}")

    sector_map = features.reset_index().groupby("ticker")["sector"].first().to_dict()
    fd = features.index.get_level_values("date")
    ft = features.index.get_level_values("ticker")
    umap = {d: set(ft[fd == d].unique()) for d in fd.unique()}

    print("Running XGBoost walk-forward (16 features, orig 23a tune)...", flush=True)
    res = run_walk_forward_backtest(
        returns=returns, features=features, model=xgb(),
        train_window=TRAIN_WINDOW, test_window=TEST_WINDOW,
        long_quantile=LONG_Q, short_quantile=SHORT_Q, transaction_cost_bps=TC_BPS,
        regime_fn=lambda d: {"k_per_sector": K},
        sector_map=sector_map,
        eligible_universe_fn=lambda d: umap.get(pd.Timestamp(d), set()),
    )
    net = res.portfolio_returns.dropna()
    gross = res.gross_returns.dropna()
    idx = net.index.intersection(gross.index)
    net, gross = net.loc[idx], gross.loc[idx]
    cost10 = gross - net
    net_t = net[net.index >= TEST_START]

    a_ann, a_t = ff5_alpha(net)
    print("\n" + "=" * 70)
    print("24b-EQUIVALENT (16 features, ORIG 23a tune) vs 13-feature 23g")
    print("=" * 70)
    print(f"  Sharpe full : {sharpe_ratio(net):+.3f}   (13-feat 23g: {SHARPE_13F:+.3f})")
    print(f"  Sharpe test : {sharpe_ratio(net_t):+.3f}")
    print(f"  ann ret/DD  : {annualised_return(net):+.1%} / {max_drawdown(net):+.1%}")
    print(f"  FF5 alpha   : {a_ann:+.1%}/yr  t={a_t:+.2f}")

    print("\n  Cost grid (FF5 alpha t):")
    ff = fetch_ff5(); ff.index = ff.index.to_period("M")
    print(f"  {'bps':>5} {'Sharpe':>8} {'FF5 a/yr':>10} {'t(a)':>7}")
    for bps in BPS_GRID:
        n = gross - (bps / 10.0) * cost10
        r = n.copy(); r.index = r.index.to_period("M")
        common = r.index.intersection(ff.index)
        rr, f = r.loc[common], ff.loc[common]
        y = rr.values - f["RF"].values
        X = np.column_stack([np.ones(len(y))] + [f[c].values
                             for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]])
        b, _, t = nw_ols(y, X)
        print(f"  {bps:>5} {sharpe_ratio(n):>+8.2f} {b[0]*12:>+9.1%} {t[0]:>+7.2f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
