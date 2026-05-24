"""Compute two new GKX-top-5 features and extend the broad Sharadar
features panel. Adds:

  * chmom = mom6m - mom_prev6m  (change in 6-month momentum / acceleration)
  * maxret = max daily return in prior calendar month  (lottery effect)

chmom needs only monthly returns (have in returns_broad_sharadar parquet).
maxret needs daily closeadj from SHARADAR/SEP (raw parquet, ~1GB, only on
Bowen's machine). If sep.parquet is missing locally, maxret is skipped
and a message is printed — Bowen should run this on his end and push the
resulting parquet to data/processed/.

Output:
  data/processed/features_broad_sharadar_with_chmom_maxret.parquet

This is the existing features panel + the new feature columns, with
sector-relative ranking applied (matching the rest of the panel's
preprocessing).

Run with:
    .venv/bin/python -m notebooks.personb.compute_chmom_maxret_features
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RETURNS_FILE = DATA_DIR / "processed" / "returns_broad_sharadar_2002_2024.parquet"
FEATURES_IN = DATA_DIR / "processed" / "features_broad_sharadar_2002_2024.parquet"
SEP_RAW = DATA_DIR / "raw" / "sharadar" / "sep.parquet"
FEATURES_OUT = DATA_DIR / "processed" / "features_broad_sharadar_with_chmom_maxret.parquet"


def sector_relative_gaussian_rank(s: pd.Series, sectors: pd.Series) -> pd.Series:
    """Per-(date, sector) percentile rank, then Gaussian transform.

    Matches the preprocessing applied to the existing 13 features in
    features_broad_sharadar_2002_2024.parquet so the new features sit on the
    same N(0,1)-ish scale.
    """
    # Index is (date, ticker). Group by date AND sector.
    dates = s.index.get_level_values("date")
    grp = pd.DataFrame({"x": s.values, "sector": sectors.values,
                         "date": dates}).groupby(["date", "sector"])
    ranks = grp["x"].rank(pct=True, method="average")
    # Avoid 0/1 (inf under norm.ppf); clip.
    ranks = ranks.clip(1e-4, 1 - 1e-4)
    z = norm.ppf(ranks.values)
    return pd.Series(z, index=s.index, name=s.name)


def compute_chmom(returns_wide: pd.DataFrame) -> pd.Series:
    """chmom = cum_return(t-6 to t-1) - cum_return(t-12 to t-7).

    Captures acceleration: is the recent 6m stronger than the prior 6m?
    Returns long-format Series indexed by (date, ticker).
    """
    # Cumulative product transform: (1+r1)(1+r2)... - 1
    # rolling apply np.prod is slow; use exp(log(1+r).rolling.sum())
    log1p = np.log1p(returns_wide)
    # mom6m_recent: cum over t-6..t-1 (so shift(1) then 6-window)
    cum_recent6 = log1p.shift(1).rolling(6).sum()
    mom6m_recent = np.expm1(cum_recent6)
    # mom6m_prior: cum over t-12..t-7 (shift(7), 6-window)
    cum_prior6 = log1p.shift(7).rolling(6).sum()
    mom6m_prior = np.expm1(cum_prior6)
    chmom_wide = mom6m_recent - mom6m_prior
    chmom_long = chmom_wide.stack(future_stack=True).rename("chmom")
    chmom_long.index = chmom_long.index.set_names(["date", "ticker"])
    return chmom_long


def compute_maxret(sep_path: Path, panel_index: pd.MultiIndex) -> pd.Series:
    """maxret = max daily return within the prior calendar month.

    Uses SHARADAR/SEP `closeadj` (split + dividend adjusted). For each
    (ticker, calendar-month) compute the maximum of daily total returns,
    then assign that value to the panel's trading-day month-end of the
    NEXT month — because the maxret of January is known at the panel
    rebalance date that falls in February. Properly aligned to the
    features panel's dates (trading-day month-ends, NOT calendar ones).
    """
    print(f"  loading SEP daily (~1GB)...")
    sep = pd.read_parquet(sep_path, columns=["ticker", "date", "closeadj"])
    sep["date"] = pd.to_datetime(sep["date"])
    sep = sep.sort_values(["ticker", "date"])
    # Daily returns within each ticker
    sep["dret"] = sep.groupby("ticker")["closeadj"].pct_change()
    sep = sep.dropna(subset=["dret"])
    # Max per (ticker, calendar month-period)
    sep["ym"] = sep["date"].dt.to_period("M")
    print(f"  computing max per (ticker, calendar-month) on {len(sep):,} daily rows...")
    monthly_max = sep.groupby(["ticker", "ym"])["dret"].max().reset_index()

    # Shift period forward: max during ym is the FEATURE value at the
    # panel's rebalance date that falls in (ym + 1).
    monthly_max["ym_feat"] = monthly_max["ym"] + 1

    # Build a map from year-month period -> the panel's actual trading-day
    # month-end date in that period (e.g., period 2024-02 -> 2024-02-29).
    panel_dates = pd.Series(panel_index.get_level_values("date").unique())
    period_to_panel_date = pd.Series(
        panel_dates.values, index=panel_dates.dt.to_period("M"),
    )
    # Map each row's ym_feat to the panel's trading-day month-end
    monthly_max["date"] = monthly_max["ym_feat"].map(period_to_panel_date)
    # Drop any (ticker, ym) whose ym+1 doesn't have a panel rebalance date
    # (i.e., before panel start or after panel end)
    monthly_max = monthly_max.dropna(subset=["date"])
    out = monthly_max.set_index(["date", "ticker"])["dret"].rename("maxret")
    return out


def main() -> int:
    print("=" * 72)
    print("Compute chmom + maxret features for broad Sharadar panel")
    print("=" * 72)

    print(f"\n[1/4] Loading existing features panel: {FEATURES_IN.name}")
    features = pd.read_parquet(FEATURES_IN)
    print(f"  features shape: {features.shape}")
    print(f"  existing columns: {list(features.columns)}")

    print(f"\n[2/4] Loading monthly returns: {RETURNS_FILE.name}")
    returns_wide = pd.read_parquet(RETURNS_FILE)
    print(f"  returns shape: {returns_wide.shape}")

    print("\n[3/4] Computing chmom = mom6m_recent - mom6m_prior...")
    chmom_long = compute_chmom(returns_wide)
    # Apply sector-relative ranking matching the existing panel's preprocessing
    # First align chmom_long to the features index (drop ticker-dates not in panel)
    chmom_aligned = chmom_long.reindex(features.index)
    n_valid = chmom_aligned.notna().sum()
    print(f"  chmom non-null: {n_valid:,} of {len(features):,} rows")
    chmom_ranked = sector_relative_gaussian_rank(chmom_aligned, features["sector"])
    print(f"  chmom: mean={chmom_ranked.mean():+.4f}, std={chmom_ranked.std():.4f}")

    # Quick orthogonality check
    print(f"\n  chmom correlation with existing features:")
    for col in ["mom", "rev", "mvol", "ivol", "log_mktcap", "dvol"]:
        cor = chmom_ranked.corr(features[col])
        print(f"    chmom vs {col:<14s}: {cor:+.4f}")

    # Add to features
    features_new = features.copy()
    features_new["chmom"] = chmom_ranked

    print(f"\n[4/4] Computing maxret = max daily return in prior month...")
    if SEP_RAW.exists():
        maxret_long = compute_maxret(SEP_RAW, features.index)
        maxret_aligned = maxret_long.reindex(features.index)
        n_valid = maxret_aligned.notna().sum()
        print(f"  maxret non-null: {n_valid:,} of {len(features):,} rows")
        maxret_ranked = sector_relative_gaussian_rank(maxret_aligned, features["sector"])
        print(f"  maxret: mean={maxret_ranked.mean():+.4f}, std={maxret_ranked.std():.4f}")
        print(f"\n  maxret correlation with existing features:")
        for col in ["mom", "rev", "mvol", "ivol", "log_mktcap", "dvol"]:
            cor = maxret_ranked.corr(features[col])
            print(f"    maxret vs {col:<14s}: {cor:+.4f}")
        features_new["maxret"] = maxret_ranked
    else:
        print(f"  SEP raw parquet not found at {SEP_RAW}.")
        print(f"  --> SKIPPED maxret. Bowen needs to:")
        print(f"      (a) push data/raw/sharadar/sep.parquet (1GB, probably too big)")
        print(f"      OR (b) run this script on his machine and push the resulting")
        print(f"          data/processed/features_broad_sharadar_with_chmom_maxret.parquet")
        print(f"  For now, writing panel with chmom only.")

    print(f"\nWriting {FEATURES_OUT.name}  ({features_new.shape})")
    features_new.to_parquet(FEATURES_OUT)
    size_mb = FEATURES_OUT.stat().st_size / 1024**2
    print(f"  size: {size_mb:.1f} MB")
    print(f"  columns: {list(features_new.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
