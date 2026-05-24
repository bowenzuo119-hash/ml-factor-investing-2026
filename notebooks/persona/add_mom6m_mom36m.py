"""add_mom6m_mom36m.py - Priority 2: two more GKX momentum features.

Extends features_broad_sharadar_with_chmom_maxret.parquet in place with:
  * mom6m  = cum return t-6..t-1   (momentum(lookback=6,  skip=1))
  * mom36m = cum return t-36..t-13 (momentum(lookback=24, skip=13)) -- the
             GKX/Green-Hand-Zhang long-term momentum/reversal window, lagged a
             year so it doesn't just re-express the 12-1 `mom`.

Both come from the monthly returns panel (already in repo). Sector-relative
Gaussian rank to match chmom/maxret. Columns are ADDED (the panel is read by
column selection via INCLUDE_FEATURES, so this is non-breaking). Prints
orthogonality vs existing features -- if mom6m overlaps `mom`/`chmom` too much
(|corr| > 0.6) it's redundant; flagged for B to drop.

    python -m notebooks.persona.add_mom6m_mom36m
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.data_loader import PROCESSED_DIR
from src.factors import momentum

RETURNS = PROCESSED_DIR / "returns_broad_sharadar_2002_2024.parquet"
FEATURES = PROCESSED_DIR / "features_broad_sharadar_with_chmom_maxret.parquet"


def sector_gaussian_rank(s: pd.Series, sectors: pd.Series) -> pd.Series:
    """Per-(date, sector) percentile rank -> Gaussian (matches chmom/maxret)."""
    dates = s.index.get_level_values("date")
    grp = pd.DataFrame({"x": s.values, "sector": sectors.values, "date": dates}
                       ).groupby(["date", "sector"])
    ranks = grp["x"].rank(pct=True, method="average").clip(1e-4, 1 - 1e-4)
    return pd.Series(norm.ppf(ranks.values), index=s.index, name=s.name)


def main() -> int:
    ret = pd.read_parquet(RETURNS)
    feats = pd.read_parquet(FEATURES)
    print(f"panel in: {feats.shape}, cols: {list(feats.columns)}")

    new_wide = {
        "mom6m": momentum(ret, lookback=6, skip=1),
        "mom36m": momentum(ret, lookback=24, skip=13),
    }
    existing = ["mom", "rev", "mvol", "ivol", "log_mktcap", "dvol", "chmom", "maxret"]
    for name, wide in new_wide.items():
        long = wide.stack(future_stack=True).rename(name)
        long.index = long.index.set_names(["date", "ticker"])
        aligned = long.reindex(feats.index)
        n = aligned.notna().sum()
        ranked = sector_gaussian_rank(aligned, feats["sector"])
        print(f"\n{name}: non-null {n:,}/{len(feats):,}")
        max_abs = 0.0
        for c in existing:
            if c in feats.columns:
                cor = ranked.corr(feats[c])
                max_abs = max(max_abs, abs(cor))
                flag = "  <-- |corr|>0.6" if abs(cor) > 0.6 else ""
                print(f"    {name} vs {c:<12s}: {cor:+.4f}{flag}")
        # B's redundancy rule: drop a feature collinear (|corr|>0.6) with an existing one.
        if max_abs > 0.6:
            print(f"  -> DROPPING {name}: max |corr| {max_abs:.2f} > 0.6 (redundant).")
        else:
            feats[name] = ranked
            print(f"  -> keeping {name} (max |corr| {max_abs:.2f}).")

    feats.to_parquet(FEATURES)
    print(f"\nwrote {feats.shape} -> {FEATURES.name} ({FEATURES.stat().st_size/1e6:.1f} MB)")
    print(f"  cols: {list(feats.columns)}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
