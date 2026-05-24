"""canonical_qfix_validate.py - does the q-filter fix move the headline?

Re-runs the Phase 24-RT broad canonical (14 feats, 24a params, k=20, broad
survivorship-free coverage universe) with the CORRECTED bankruptcy filter
(`is_bankruptcy_ticker`, which un-drops NDAQ + IONQ) and compares to the
committed 24-RT (+1.08 full / FF5 alpha +18.2%/yr t=5.74, old symbol-only
filter). Expect a negligible move (2 names in ~4,400).

    KMP_DUPLICATE_LIB_OK=TRUE python -m notebooks.persona.canonical_qfix_validate
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
           "roe", "roa", "de", "asset_growth", "accruals", "chmom"]


def xgb():
    p = json.load(open(P24A))["best_params"] if P24A.exists() \
        else json.load(open(P23A))["by_model"]["xgboost"]["best_params"]
    return XGBoostModel(
        target_kind=TARGET_KIND,
        n_estimators=p.get("n_estimators", 200), max_depth=p.get("max_depth", 3),
        learning_rate=p.get("learning_rate", 0.0115), subsample=p.get("subsample", 0.717),
        colsample_bytree=p.get("colsample_bytree", 0.890),
        min_child_weight=p.get("min_child_weight", 11),
        reg_alpha=p.get("reg_alpha", 0.794), reg_lambda=p.get("reg_lambda", 2.305),
    )


def ff5(net):
    ff = fetch_ff5(); ff.index = ff.index.to_period("M")
    r = net.copy(); r.index = r.index.to_period("M")
    c = r.index.intersection(ff.index); r, f = r.loc[c], ff.loc[c]
    y = r.values - f["RF"].values
    X = np.column_stack([np.ones(len(y))] + [f[col].values for col in ["Mkt-RF","SMB","HML","RMW","CMA"]])
    b, _, t = nw_ols(y, X)
    return b[0]*12, t[0]


def _old_filter(t):  # the buggy symbol-only rule (drops NDAQ/IONQ)
    t = str(t).upper().strip()
    return len(t) >= 4 and t.endswith("Q")


def run_arm(filter_fn, label):
    returns = pd.read_parquet(RETURNS_FILE)
    features = pd.read_parquet(FEATURES_FILE)[INCLUDE + ["sector"]]
    tk = features.index.get_level_values("ticker")
    features = features.loc[~pd.Series([filter_fn(t) for t in tk], index=features.index)]
    returns = returns.drop(columns=[c for c in returns.columns if filter_fn(c)])
    sector_map = features.reset_index().groupby("ticker")["sector"].first().to_dict()
    fd = features.index.get_level_values("date"); ft = features.index.get_level_values("ticker")
    umap = {d: set(ft[fd == d].unique()) for d in fd.unique()}
    print(f"[{label}] returns {returns.shape[1]} cols; running...", flush=True)
    res = run_walk_forward_backtest(
        returns=returns, features=features, model=xgb(),
        train_window=TRAIN_WINDOW, test_window=TEST_WINDOW,
        long_quantile=LONG_Q, short_quantile=SHORT_Q, transaction_cost_bps=TC_BPS,
        regime_fn=lambda d: {"k_per_sector": K},
        sector_map=sector_map, eligible_universe_fn=lambda d: umap.get(pd.Timestamp(d), set()),
    )
    net = res.portfolio_returns.dropna()
    a, t = ff5(net)
    return {"sharpe": sharpe_ratio(net), "sharpe_t": sharpe_ratio(net[net.index >= TEST_START]),
            "alpha": a, "alpha_t": t, "ncols": returns.shape[1]}


def main() -> int:
    # Same recipe, only the q-filter differs -> the delta isolates the fix
    # (avoids confounding with my-recipe-vs-committed replication noise).
    old = run_arm(_old_filter, "OLD symbol-only")
    new = run_arm(is_bankruptcy_ticker, "NEW isdelisted-gated (+NDAQ/IONQ)")
    print("\n" + "=" * 64)
    print("Q-FILTER FIX -- same recipe, old vs corrected (delta = pure fix)")
    print("=" * 64)
    print(f"  {'metric':<14}{'OLD':>12}{'NEW(+2)':>12}{'delta':>10}")
    print(f"  {'Sharpe full':<14}{old['sharpe']:>+12.3f}{new['sharpe']:>+12.3f}{new['sharpe']-old['sharpe']:>+10.3f}")
    print(f"  {'Sharpe test':<14}{old['sharpe_t']:>+12.3f}{new['sharpe_t']:>+12.3f}{new['sharpe_t']-old['sharpe_t']:>+10.3f}")
    print(f"  {'FF5 alpha/yr':<14}{old['alpha']:>+12.1%}{new['alpha']:>+12.1%}{new['alpha']-old['alpha']:>+10.1%}")
    print(f"  {'FF5 alpha t':<14}{old['alpha_t']:>+12.2f}{new['alpha_t']:>+12.2f}{new['alpha_t']-old['alpha_t']:>+10.2f}")
    print(f"  cols: {old['ncols']} -> {new['ncols']} (+{new['ncols']-old['ncols']})")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
