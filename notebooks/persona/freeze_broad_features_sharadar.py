"""freeze_broad_features_sharadar.py - broad-universe feature panel (Sharadar).

Companion to freeze_broad_panel_sharadar.py (returns). Builds the same 13-feature
stack as factors.build_feature_panel, but sourced entirely from the local
Sharadar archive on the broad survivorship-free universe -- so Person B can
retune Phase 23 from two small files (returns + features) without the 3.3GB raw.

Sources (definitions match factors.build_feature_panel exactly):
  mom/rev/mvol/ivol   <- broad monthly returns (factors.* pure functions)
  log_mktcap          <- DAILY marketcap (month-end, logged)
  dvol                <- SEP closeunadj x volume (monthly mean daily $vol, logged)
  bm / ep             <- SF1 ARQ/ART (factors.load_value_factors_monthly)
  roe/roa/de/asset_growth/accruals <- SF1 (factors.load_extended_fundamentals_monthly)
  sector              <- TICKERS

To let the SF1 PIT loaders run OFFLINE on the broad universe, the local raw
SF1 (all ~17k tickers) is copied into load_fundamentals' processed cache path.

    python -m notebooks.persona.freeze_broad_features_sharadar            # full
    python -m notebooks.persona.freeze_broad_features_sharadar --limit 50 # smoke
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.data_loader import (
    RAW_DIR, PROCESSED_DIR, SHARADAR_SF1_CACHE_TMPL,
    _load_universe_meta, _load_daily_marketcap_monthly,
)
from src.factors import (
    momentum, reversal, monthly_volatility, idiosyncratic_volatility,
    sector_relative_rank, load_value_factors_monthly,
    load_extended_fundamentals_monthly,
)

START, END = "2002-01-01", "2024-12-31"
RETURNS_PANEL = PROCESSED_DIR / "returns_broad_sharadar_2002_2024.parquet"
OUT = PROCESSED_DIR / "features_broad_sharadar_2002_2024.parquet"
INCLUDE = ("mom", "rev", "mvol", "ivol", "log_mktcap", "bm", "ep", "dvol",
           "roe", "roa", "de", "asset_growth", "accruals")


def populate_sf1_cache() -> None:
    """Copy local raw SF1 (full universe) into load_fundamentals' processed
    cache so the value/ext PIT loaders run offline on the broad universe.
    Always overwrites: the raw is a superset of any S&P-union cache."""
    for dim, raw in [("ARQ", "sf1_AR_arq.parquet"), ("ART", "sf1_AR_art.parquet")]:
        dst = PROCESSED_DIR / SHARADAR_SF1_CACHE_TMPL.format(dimension=dim)
        df = pd.read_parquet(RAW_DIR / raw)
        df.to_parquet(dst)
        print(f"  SF1 cache <- {raw}: {len(df):,} rows, {df['ticker'].nunique():,} tickers -> {dst.name}")


def _period_wide_to_returns_index(period_wide: pd.DataFrame, ret_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Reindex a period('M')-indexed wide feature onto the returns panel's
    trading-day month-end index."""
    ret_periods = ret_index.to_period("M")
    out = period_wide.reindex(ret_periods)
    out.index = ret_index
    return out


def compute_log_mktcap(ret: pd.DataFrame) -> pd.DataFrame:
    mc = _load_daily_marketcap_monthly()  # [period, ticker, marketcap]
    wide = mc.pivot(index="period", columns="ticker", values="marketcap")
    wide = _period_wide_to_returns_index(wide, ret.index).reindex(columns=ret.columns)
    return np.log(wide.where(wide > 0))


def compute_dvol(ret: pd.DataFrame) -> pd.DataFrame:
    cols = ["ticker", "date", "closeunadj", "volume"]
    filt = [("date", ">=", pd.Timestamp(START).date()), ("date", "<=", pd.Timestamp(END).date()),
            ("ticker", "in", list(ret.columns))]
    df = pq.read_table(RAW_DIR / "sep.parquet", columns=cols, filters=filt).to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    df["dv"] = df["closeunadj"] * df["volume"]
    df["period"] = df["date"].dt.to_period("M")
    monthly = df.groupby(["period", "ticker"])["dv"].mean().unstack("ticker")
    wide = _period_wide_to_returns_index(monthly, ret.index).reindex(columns=ret.columns)
    return np.log(wide.where(wide > 0))


def build(ret: pd.DataFrame, sector_rank: bool = True) -> pd.DataFrame:
    cols = tuple(ret.columns)
    feat: dict[str, pd.DataFrame] = {}
    print("  price features (mom/rev/mvol/ivol) ...")
    feat["mom"] = momentum(ret, lookback=11, skip=1)
    feat["rev"] = reversal(ret)
    feat["mvol"] = monthly_volatility(ret, window=6)
    market = ret.mean(axis=1, skipna=True)
    feat["ivol"] = idiosyncratic_volatility(ret, market, window=24)
    print("  log_mktcap (DAILY) ...")
    feat["log_mktcap"] = compute_log_mktcap(ret)
    print("  dvol (SEP closeunadj x volume) ...")
    feat["dvol"] = compute_dvol(ret)
    print("  value factors bm/ep (SF1 ARQ/ART) ...")
    vf = load_value_factors_monthly(start=START, end=END, tickers=cols, target_dates=ret.index)
    feat["bm"] = vf["bm"].unstack("ticker").reindex(index=ret.index, columns=ret.columns)
    feat["ep"] = vf["ep_ttm"].unstack("ticker").reindex(index=ret.index, columns=ret.columns)
    print("  extended fundamentals roe/roa/de/asset_growth/accruals (SF1) ...")
    ext = load_extended_fundamentals_monthly(start=START, end=END, tickers=cols, target_dates=ret.index)
    for k in ("roe", "roa", "de", "asset_growth", "accruals"):
        feat[k] = ext[k].unstack("ticker").reindex(index=ret.index, columns=ret.columns)

    # sector from TICKERS
    meta = _load_universe_meta()
    ticker_to_sector = dict(zip(meta["ticker"], meta["sector"]))

    print("  stacking to long + sector-relative ranks ...")
    long_frames = []
    for name, wide in feat.items():
        long = wide.stack(future_stack=True).rename(name).to_frame()
        long.index = long.index.set_names(["date", "ticker"])
        long_frames.append(long)
    panel = pd.concat(long_frames, axis=1).sort_index()
    panel["sector"] = (
        panel.index.get_level_values("ticker").map(ticker_to_sector).fillna("Unknown")
    )
    feature_cols = [c for c in panel.columns if c != "sector"]
    panel = panel.dropna(subset=feature_cols, how="all")
    if sector_rank:
        panel = sector_relative_rank(panel, feature_cols=feature_cols)
    return panel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap universe (smoke test)")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    print("Populating SF1 cache from local raw ...")
    populate_sf1_cache()

    ret = pd.read_parquet(RETURNS_PANEL)
    if args.limit:
        ret = ret.iloc[:, : args.limit]
        print(f"[smoke] limited to {ret.shape[1]} tickers")
    print(f"Returns panel: {ret.shape[0]} months x {ret.shape[1]} tickers")

    panel = build(ret)
    print(f"\nFeature panel: {len(panel):,} rows x {panel.shape[1]} cols")
    print(f"  columns: {list(panel.columns)}")
    print(f"  non-null per feature (% of rows):")
    for c in [c for c in panel.columns if c != "sector"]:
        print(f"    {c:14s} {panel[c].notna().mean()*100:5.1f}%")

    if not args.no_write and not args.limit:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(OUT)
        print(f"\nFroze -> {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")
    else:
        print("\n(smoke / --no-write: not frozen)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
