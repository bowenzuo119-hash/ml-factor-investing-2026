"""regime_overlay_ablation_broad.py - Priority 3: regime overlay on Phase 23g.

Runs the Phase 23g broad canonical recipe (XGBoost, 13 features, Q-filter, PIT,
retuned 23a params, k=20) TWICE -- identical except the regime overlay -- so
the with-vs-without delta is internally consistent (the §5 ablation table):

  no-overlay : regime_fn = {"k_per_sector": 20}            (leverage 1.0 always)
  overlay    : regime_fn = leverage from C's CSV + k=20    (crisis de-levers 0.4)

Leverage-only by construction: we take ONLY `leverage` from the overlay CSV and
force k=20 (the 23g canonical k), so the k/quantile columns of the CSV are
ignored -- the ablation (Phase A) showed the leverage lever helps and the
k-concentration lever hurts.

    KMP_DUPLICATE_LIB_OK=TRUE python -m notebooks.persona.regime_overlay_ablation_broad
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.backtest import run_walk_forward_backtest
from src.models import XGBoostModel
from src.regime import make_regime_fn
from src.metrics import sharpe_ratio, max_drawdown, annualised_return

ROOT = Path(__file__).resolve().parents[2]
RETURNS_FILE = ROOT / "data" / "processed" / "returns_broad_sharadar_2002_2024.parquet"
FEATURES_FILE = ROOT / "data" / "processed" / "features_broad_sharadar_with_chmom_maxret.parquet"
PARAMS_FILE = ROOT / "results" / "23a_retune_broad_sharadar" / "best_params.json"
OVERLAY_CSV = ROOT / "results" / "regime_overlay_rules.csv"

TRAIN_WINDOW, TEST_WINDOW = 120, 12
LONG_Q, SHORT_Q, TC_BPS, K = 0.8, 0.2, 10.0, 20
TARGET_KIND = "sector_relative"
TEST_START = pd.Timestamp("2019-01-01")
INCLUDE = ["mom", "rev", "mvol", "ivol", "log_mktcap", "bm", "ep", "dvol",
           "roe", "roa", "de", "asset_growth", "accruals"]  # 23g's original 13


def is_q_bankruptcy(t: str) -> bool:  # delegates to the centralized filter
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


def summarise(res, label):
    net = res.portfolio_returns.dropna()
    net_t = net[net.index >= TEST_START]
    return {
        "config": label,
        "sharpe_full": round(sharpe_ratio(net), 3),
        "sharpe_test": round(sharpe_ratio(net_t), 3),
        "ann_ret_full": round(annualised_return(net), 4),
        "maxdd_full": round(max_drawdown(net), 4),
        "maxdd_test": round(max_drawdown(net_t), 4),
        "mean_lev": round(float(res.leverage.mean()), 3),
    }


def main() -> int:
    print("Loading broad panels + Q-filter ...")
    returns = pd.read_parquet(RETURNS_FILE)
    features = pd.read_parquet(FEATURES_FILE)[INCLUDE + ["sector"]]

    # Q-filter: drop bankrupt (Q-suffix) tickers from features rows + returns cols
    tk = features.index.get_level_values("ticker")
    features = features.loc[~pd.Series([is_q_bankruptcy(t) for t in tk], index=features.index)]
    qcols = [c for c in returns.columns if is_q_bankruptcy(c)]
    returns = returns.drop(columns=qcols)
    print(f"  features {features.shape}, returns {returns.shape} "
          f"(dropped {len(qcols)} Q cols)")

    sector_map = features.reset_index().groupby("ticker")["sector"].first().to_dict()
    fd = features.index.get_level_values("date")
    ft = features.index.get_level_values("ticker")
    umap = {d: set(ft[fd == d].unique()) for d in fd.unique()}
    universe_at = lambda d: umap.get(pd.Timestamp(d), set())

    lev_fn = make_regime_fn(OVERLAY_CSV)

    def no_overlay(d):
        return {"k_per_sector": K}

    def overlay(d):
        return {"leverage": float(lev_fn(d).get("leverage", 1.0)), "k_per_sector": K}

    common = dict(
        returns=returns, features=features,
        train_window=TRAIN_WINDOW, test_window=TEST_WINDOW,
        long_quantile=LONG_Q, short_quantile=SHORT_Q, transaction_cost_bps=TC_BPS,
        sector_map=sector_map, eligible_universe_fn=universe_at,
    )
    rows = []
    for label, rfn in [("23g no-overlay (k=20)", no_overlay),
                       ("23g + leverage-only overlay", overlay)]:
        print(f"\nRunning XGBoost -- {label} ...", flush=True)
        res = run_walk_forward_backtest(model=xgb(), regime_fn=rfn, **common)
        rows.append(summarise(res, label))

    print("\n" + "=" * 84)
    print("REGIME OVERLAY ABLATION -- Phase 23g broad canonical (XGBoost)")
    print("=" * 84)
    print(pd.DataFrame(rows).to_string(index=False))
    base, ov = rows
    print("\nReadout (overlay effect):")
    print(f"  Sharpe full : {base['sharpe_full']:+.2f} -> {ov['sharpe_full']:+.2f}")
    print(f"  Sharpe test : {base['sharpe_test']:+.2f} -> {ov['sharpe_test']:+.2f}")
    print(f"  Max DD full : {base['maxdd_full']:+.1%} -> {ov['maxdd_full']:+.1%}")
    print(f"  Max DD test : {base['maxdd_test']:+.1%} -> {ov['maxdd_test']:+.1%}")
    print(f"  mean leverage: {base['mean_lev']:.2f} -> {ov['mean_lev']:.2f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
