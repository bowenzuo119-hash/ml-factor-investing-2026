"""Phase 23f: ensemble XGBoost + Lasso predictions.

Both XGBoost and Lasso showed significant FF5 alpha at k=20 + Q-filter
in Phase 23c. Averaging the predictions often diversifies model-specific
noise, lifting Sharpe modestly.

Three ensemble flavours tested:
  * mean: equal-weight average
  * rank: average of cross-sectional ranks (more robust to outliers)
  * z: average of cross-sectional z-scores

Each then run through k=20 + Q-filter + FF5 per Phase 23c.

Run with:
    .venv/bin/python -m notebooks.personb.23f_ensemble_xgb_lasso
"""
from __future__ import annotations

import pickle
import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import summary_stats


PHASE_DIR = (
    Path(__file__).resolve().parents[2] / "results" / "23_canonical_broad_sharadar"
)
RESULTS_DIR = (
    Path(__file__).resolve().parents[2] / "results" / "23f_ensemble"
)
RETURNS_FILE = (
    Path(__file__).resolve().parents[2] / "data" / "processed"
    / "returns_broad_sharadar_2002_2024.parquet"
)
FEATURES_FILE = (
    Path(__file__).resolve().parents[2] / "data" / "processed"
    / "features_broad_sharadar_2002_2024.parquet"
)

# Reuse Phase 23c's portfolio construction and FF5 regression
_spec = importlib.util.spec_from_file_location(
    "p23c", "notebooks/personb/23c_k1_qfilter_canonical.py")
p23c = importlib.util.module_from_spec(_spec); sys.modules["p23c"] = p23c
_spec.loader.exec_module(p23c)

K = 20
TEST_START = pd.Timestamp("2019-01-01")
TEST_END = pd.Timestamp("2024-12-31")
LONG_OOS_START = pd.Timestamp("2015-01-01")
FULL_OOS_START = pd.Timestamp("2012-04-01")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print(f"Phase 23f: ensemble XGBoost + Lasso (k={K} + Q-filter + FF5)")
    print("=" * 72)

    preds_wide = pd.read_parquet(PHASE_DIR / "predictions.parquet")
    returns_wide = pd.read_parquet(RETURNS_FILE)
    features = pd.read_parquet(FEATURES_FILE)
    sector_map = (
        features.reset_index().groupby("ticker")["sector"].first().to_dict()
    )

    xgb = preds_wide["XGBoost"].dropna()
    las = preds_wide["Lasso"].dropna()
    # Align on common (date, ticker) index
    common = xgb.index.intersection(las.index)
    xgb_c = xgb.loc[common]
    las_c = las.loc[common]

    # Three ensembles
    ensembles = {}

    # 1. mean of raw scores
    ensembles["mean_xgb_lasso"] = 0.5 * xgb_c + 0.5 * las_c

    # 2. mean of cross-sectional ranks (per date)
    def to_xs_rank(s):
        return s.groupby(level="date").rank(pct=True)
    ensembles["rank_xgb_lasso"] = 0.5 * to_xs_rank(xgb_c) + 0.5 * to_xs_rank(las_c)

    # 3. mean of cross-sectional z-scores (per date)
    def to_xs_z(s):
        m = s.groupby(level="date").transform("mean")
        sd = s.groupby(level="date").transform("std").replace(0, np.nan)
        return (s - m) / sd
    ensembles["z_xgb_lasso"] = 0.5 * to_xs_z(xgb_c) + 0.5 * to_xs_z(las_c)

    rows = []
    rets_by = {}
    for name, preds in ensembles.items():
        print(f"\n[{name}] reconstructing k={K} portfolio with Q-filter...")
        rets = p23c.build_portfolio_returns(
            preds, returns_wide, sector_map, k=K, q_filter=True,
        )
        rets_by[name] = rets

        for win, lo, hi in [
            ("test", TEST_START, TEST_END),
            ("long-OOS", LONG_OOS_START, TEST_END),
            ("full-OOS", FULL_OOS_START, TEST_END),
        ]:
            sl = rets[(rets.index >= lo) & (rets.index <= hi)]
            if len(sl) < 12:
                continue
            stats = summary_stats(sl)
            ff5 = p23c.ff5_regress(rets, lo, hi)
            sig = "*** SIG" if ff5["alpha_p"] < 0.05 else ""
            row = {
                "ensemble": name, "window": win, "n_months": len(sl),
                "sharpe": stats["sharpe_ratio"],
                "ann_return": stats["annualised_return"],
                "max_dd": stats["max_drawdown"],
                "vol": stats["annualised_volatility"],
                "ff5_alpha_ann_pct": ff5["alpha_ann"],
                "ff5_alpha_t": ff5["alpha_t"],
                "ff5_alpha_p": ff5["alpha_p"],
                "mkt_beta": ff5["Mkt-RF"][0],
            }
            rows.append(row)
            print(f"  {win:10s}  Sharpe={stats['sharpe_ratio']:+.3f}  "
                  f"FF5α={ff5['alpha_ann']:+5.2f}%/yr (t={ff5['alpha_t']:+.2f}, "
                  f"p={ff5['alpha_p']:.3f}) {sig}  Mkt-β={ff5['Mkt-RF'][0]:+.2f}")

    df = pd.DataFrame(rows)
    df.to_parquet(RESULTS_DIR / "metrics.parquet")
    with open(RESULTS_DIR / "portfolio_returns.pkl", "wb") as f:
        pickle.dump(rets_by, f)

    print()
    print("=" * 72)
    print("SUMMARY -- ensemble Sharpes by window")
    print("=" * 72)
    print(df.pivot_table(index="ensemble", columns="window",
                         values="sharpe").round(3).to_string())
    print()
    print("Compare to Phase 23c (XGBoost alone, k=20 + Q-filter):")
    print("  test=+0.59  long-OOS=+0.71  full-OOS=+0.89  FF5α=+26.5%/yr t=4.17")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
