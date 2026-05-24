"""out_of_time_test.py - Bonus robustness check: strict out-of-time split.

Trains XGBoost ONCE on 2002-2018 (no walk-forward refit) and trades the frozen
model over 2019-2024, on the same Phase 23g broad recipe (13 feats, Q-filter,
PIT, retuned 23a params, k=20). If the OOT test Sharpe is within ~0.1 of the
walk-forward test Sharpe (~1.0), the walk-forward isn't manufacturing the edge
via repeated refits -> bulletproofs REPORT 6 against "walk-forward leakage".

Single fit is achieved by setting train_window = #months before 2019-01 and
test_window huge (the engine refits only at i==train_window, never again).

    KMP_DUPLICATE_LIB_OK=TRUE python -m notebooks.persona.out_of_time_test
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.backtest import run_walk_forward_backtest
from src.models import XGBoostModel
from src.metrics import sharpe_ratio, max_drawdown, annualised_return

ROOT = Path(__file__).resolve().parents[2]
RETURNS_FILE = ROOT / "data" / "processed" / "returns_broad_sharadar_2002_2024.parquet"
FEATURES_FILE = ROOT / "data" / "processed" / "features_broad_sharadar_with_chmom_maxret.parquet"
PARAMS_FILE = ROOT / "results" / "23a_retune_broad_sharadar" / "best_params.json"

LONG_Q, SHORT_Q, TC_BPS, K = 0.8, 0.2, 10.0, 20
TARGET_KIND = "sector_relative"
OOS_START = pd.Timestamp("2019-01-01")
INCLUDE = ["mom", "rev", "mvol", "ivol", "log_mktcap", "bm", "ep", "dvol",
           "roe", "roa", "de", "asset_growth", "accruals"]
WF_TEST_SHARPE = 0.996  # 23g walk-forward test-only (from the ablation no-overlay arm)


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


def main() -> int:
    returns = pd.read_parquet(RETURNS_FILE)
    features = pd.read_parquet(FEATURES_FILE)[INCLUDE + ["sector"]]
    tk = features.index.get_level_values("ticker")
    features = features.loc[~pd.Series([is_q(t) for t in tk], index=features.index)]
    returns = returns.drop(columns=[c for c in returns.columns if is_q(c)])

    sector_map = features.reset_index().groupby("ticker")["sector"].first().to_dict()
    fd = features.index.get_level_values("date")
    ft = features.index.get_level_values("ticker")
    umap = {d: set(ft[fd == d].unique()) for d in fd.unique()}
    universe_at = lambda d: umap.get(pd.Timestamp(d), set())

    # train_window = #months before OOS_START -> single fit on 2002..2018, no refit
    panel_dates = returns.index
    train_window = int((panel_dates < OOS_START).sum())
    test_window = len(panel_dates) + 1  # never refit again
    print(f"OOT split: train on {panel_dates[0].date()}..{panel_dates[train_window-1].date()} "
          f"({train_window} months), trade {panel_dates[train_window].date()}.. "
          f"(frozen model, no refit)")

    res = run_walk_forward_backtest(
        returns=returns, features=features, model=xgb(),
        train_window=train_window, test_window=test_window,
        long_quantile=LONG_Q, short_quantile=SHORT_Q, transaction_cost_bps=TC_BPS,
        regime_fn=lambda d: {"k_per_sector": K},
        sector_map=sector_map, eligible_universe_fn=universe_at,
    )
    net = res.portfolio_returns.dropna()
    oos = net[net.index >= OOS_START]
    n_refits = res.metadata.get("n_refits", "?")

    print("\n" + "=" * 64)
    print("OUT-OF-TIME TEST - XGBoost trained once on 2002-18, traded 2019-24")
    print("=" * 64)
    print(f"  refits during OOS: {n_refits} (should be 1 = single fit)")
    print(f"  OOT test Sharpe (2019-24)   : {sharpe_ratio(oos):+.3f}")
    print(f"  walk-forward test Sharpe    : {WF_TEST_SHARPE:+.3f}")
    print(f"  difference                  : {sharpe_ratio(oos)-WF_TEST_SHARPE:+.3f}")
    print(f"  OOT ann return / max DD     : {annualised_return(oos):+.1%} / {max_drawdown(oos):+.1%}")
    diff = abs(sharpe_ratio(oos) - WF_TEST_SHARPE)
    print(f"\n  {'PASS' if diff < 0.15 else 'CHECK'}: walk-forward "
          f"{'is not manufacturing the edge' if diff < 0.15 else 'differs materially -- investigate'} "
          f"(|delta| {diff:.2f})")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
