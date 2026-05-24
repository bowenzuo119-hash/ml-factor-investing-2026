"""Phase 23c: cleaned canonical -- k=1 + Q-suffix bankrupt-ticker filter.

Reuses Phase 23's saved predictions (no model retraining). At each
rebalance:
  1. Drop all tickers whose symbol contains 'Q' as the last character
     (Sharadar convention: 'Q' suffix = bankruptcy proceedings, e.g.
     INTEQ, LKSDQ, FREDQ, LEHMQ, ENRNQ, etc.). These are not legitimately
     tradable and their reported returns are often artifacts of post-
     bankruptcy auction prints.
  2. Select top-1 / bottom-1 by prediction within each GICS sector
     (k=1, the empirical optimum from Phase 23b's k-sweep).
  3. Equal-weight, dollar-neutral, 10 bps cost on L1 turnover.
  4. Compute Sharpe, max DD, IC, FF5 alpha (Newey-West HAC SE).

The goal: get a CLEAN headline number that excludes the data-anomaly
tail risk we surfaced in Phase 23 (the April 2020 +144% month was
dominated by bankrupt + penny-stock outliers like NNDM +218%, LSCG +285%).

Run with:
    .venv/bin/python -m notebooks.personb.23c_k1_qfilter_canonical
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import is_bankruptcy_ticker
from src.metrics import summary_stats, information_coefficient


PHASE_DIR = (
    Path(__file__).resolve().parents[2] / "results" / "23_canonical_broad_sharadar"
)
RESULTS_DIR = (
    Path(__file__).resolve().parents[2] / "results" / "23c_k1_qfilter_canonical"
)
RETURNS_FILE = (
    Path(__file__).resolve().parents[2] / "data" / "processed"
    / "returns_broad_sharadar_2002_2024.parquet"
)
FEATURES_FILE = (
    Path(__file__).resolve().parents[2] / "data" / "processed"
    / "features_broad_sharadar_2002_2024.parquet"
)

TEST_START = pd.Timestamp("2019-01-01")
TEST_END = pd.Timestamp("2024-12-31")
LONG_OOS_START = pd.Timestamp("2015-01-01")

K = 20  # k-sweep with Q-filter showed k=15-30 is the robust plateau;
        # k=20 picked for symmetry (~440 positions, ~0.45% per position).
COST_BPS = 10.0
MODELS = ("Lasso", "XGBoost", "NN")


def build_portfolio_returns(preds: pd.Series, returns_wide: pd.DataFrame,
                            sector_map: dict[str, str], k: int = 1,
                            cost_bps: float = COST_BPS,
                            q_filter: bool = True) -> pd.Series:
    next_returns = returns_wide.shift(-1)
    rebal_dates = sorted(preds.index.get_level_values("date").unique())
    cost_rate = cost_bps / 10_000.0
    prev_weights = pd.Series(dtype=float)
    records = []
    q_dropped_per_rebal = []

    for t in rebal_dates:
        try:
            cs = preds.xs(t, level="date")
        except KeyError:
            continue
        cs_df = pd.DataFrame({
            "score": cs.values,
            "sector": [sector_map.get(str(tk).upper(), "UNKNOWN") for tk in cs.index],
        }, index=cs.index).dropna(subset=["score"])
        before = len(cs_df)
        if q_filter:
            cs_df = cs_df[~cs_df.index.map(is_bankruptcy_ticker)]
        q_dropped_per_rebal.append(before - len(cs_df))

        longs, shorts = [], []
        for sec, grp in cs_df.groupby("sector", sort=False):
            ranked = grp["score"].sort_values(ascending=False)
            longs.extend(ranked.head(k).index.tolist())
            shorts.extend(ranked.tail(k).index.tolist())
        if not longs or not shorts:
            prev_weights = pd.Series(dtype=float)
            continue

        weights = pd.Series(0.0, index=cs.index)
        weights.loc[longs] = 1.0 / len(longs)
        weights.loc[shorts] = -1.0 / len(shorts)

        if t not in next_returns.index:
            prev_weights = weights
            continue
        rets_t = next_returns.loc[t].reindex(weights.index)
        valid = weights.index.intersection(rets_t.dropna().index)
        gross = float((weights.loc[valid] * rets_t.loc[valid]).sum())

        union = weights.index.union(prev_weights.index)
        w_now = weights.reindex(union, fill_value=0.0)
        w_prev = prev_weights.reindex(union, fill_value=0.0)
        turnover = float((w_now - w_prev).abs().sum())
        cost = cost_rate * turnover
        records.append((t, gross - cost))
        prev_weights = weights

    print(f"  q-filtered dropped per rebal: mean={np.mean(q_dropped_per_rebal):.0f}, "
          f"max={max(q_dropped_per_rebal)}")
    return pd.Series(dict(records)).sort_index()


def ff5_regress(rets: pd.Series, lo: pd.Timestamp, hi: pd.Timestamp) -> dict:
    """FF5 regression with Newey-West HAC SE. Returns dict of stats."""
    import sys, importlib.util
    spec = importlib.util.spec_from_file_location(
        "phase7", "notebooks/personb/07_statistical_robustness.py")
    mod = importlib.util.module_from_spec(spec); sys.modules["phase7"] = mod
    spec.loader.exec_module(mod)
    ff5 = mod.fetch_ff_monthly(five_factor=True)

    sl = rets[(rets.index >= lo) & (rets.index <= hi)]
    a = pd.DataFrame({"y": sl.values, "ym": sl.index.to_period("M")}).set_index("ym")
    b = pd.DataFrame(ff5.values, index=ff5.index.to_period("M"), columns=ff5.columns)
    merged = a.join(b, how="inner").dropna()
    y = (merged["y"] - merged["RF"]).to_numpy()
    Xcols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    X = np.column_stack([np.ones(len(merged))] + [merged[c].to_numpy() for c in Xcols])
    r = mod.regress_with_hac(y, X, lags=6)
    return {
        "n": r["n"], "r2": r["r2"],
        "alpha_ann": r["beta"][0] * 12 * 100, "alpha_t": r["t"][0], "alpha_p": r["p"][0],
        "Mkt-RF": (r["beta"][1], r["t"][1], r["p"][1]),
        "SMB":    (r["beta"][2], r["t"][2], r["p"][2]),
        "HML":    (r["beta"][3], r["t"][3], r["p"][3]),
        "RMW":    (r["beta"][4], r["t"][4], r["p"][4]),
        "CMA":    (r["beta"][5], r["t"][5], r["p"][5]),
    }


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print(f"Phase 23c: k={K} + Q-suffix bankrupt-ticker filter on Phase 23 preds")
    print("=" * 72)

    preds_wide = pd.read_parquet(PHASE_DIR / "predictions.parquet")
    returns_wide = pd.read_parquet(RETURNS_FILE)
    features = pd.read_parquet(FEATURES_FILE)
    sector_map = (
        features.reset_index().groupby("ticker")["sector"].first().to_dict()
    )
    print(f"  predictions: {preds_wide.shape}")
    print(f"  returns: {returns_wide.shape}")
    print(f"  sector_map: {len(sector_map)} tickers")
    n_q = sum(1 for t in preds_wide.index.get_level_values("ticker").unique()
              if is_bankruptcy_ticker(t))
    print(f"  Q-suffix bankrupt tickers in predictions universe: {n_q}")

    rows = []
    rets_by_model = {}
    for model in MODELS:
        if model not in preds_wide.columns:
            continue
        print(f"\n[{model}] reconstructing k={K} portfolio with Q-filter...")
        preds = preds_wide[model].dropna()
        rets = build_portfolio_returns(preds, returns_wide, sector_map, k=K)
        rets_by_model[model] = rets

        for win, lo, hi in [
            ("test (2019-2024)", TEST_START, TEST_END),
            ("long-OOS (2015-2024)", LONG_OOS_START, TEST_END),
            ("full-OOS (2012-2024)", pd.Timestamp("2012-04-01"), TEST_END),
        ]:
            sl = rets[(rets.index >= lo) & (rets.index <= hi)]
            if len(sl) < 12:
                continue
            stats = summary_stats(sl)
            ff5 = ff5_regress(rets, lo, hi)
            row = {
                "model": model, "window": win, "n_months": len(sl),
                "sharpe": stats["sharpe_ratio"],
                "ann_return": stats["annualised_return"],
                "max_dd": stats["max_drawdown"],
                "vol": stats["annualised_volatility"],
                "ff5_alpha_ann_pct": ff5["alpha_ann"],
                "ff5_alpha_t": ff5["alpha_t"],
                "ff5_alpha_p": ff5["alpha_p"],
                "mkt_beta": ff5["Mkt-RF"][0],
                "mkt_t": ff5["Mkt-RF"][1],
                "hml_beta": ff5["HML"][0],
                "smb_beta": ff5["SMB"][0],
                "ff5_r2": ff5["r2"],
            }
            rows.append(row)
            sig = "*** SIG" if ff5["alpha_p"] < 0.05 else ""
            print(f"  {win:24s}  Sharpe={stats['sharpe_ratio']:+.3f}  "
                  f"FF5 α={ff5['alpha_ann']:+5.2f}%/yr (t={ff5['alpha_t']:+.2f}, "
                  f"p={ff5['alpha_p']:.3f}) {sig}  Mkt-β={ff5['Mkt-RF'][0]:+.2f}")

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_parquet(RESULTS_DIR / "metrics.parquet")
    with open(RESULTS_DIR / "portfolio_returns.pkl", "wb") as f:
        pickle.dump(rets_by_model, f)

    print()
    print("=" * 72)
    print("HEADLINE (XGBoost canonical):")
    print("=" * 72)
    xg = metrics_df[metrics_df["model"] == "XGBoost"]
    for _, r in xg.iterrows():
        sig = "*** SIG" if r["ff5_alpha_p"] < 0.05 else ""
        print(f"  {r['window']:24s}  Sharpe={r['sharpe']:+.3f}  "
              f"Ann ret={r['ann_return']*100:+5.2f}%  DD={r['max_dd']*100:+6.2f}%  "
              f"FF5α={r['ff5_alpha_ann_pct']:+5.2f}%/yr (t={r['ff5_alpha_t']:+.2f}) {sig}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
