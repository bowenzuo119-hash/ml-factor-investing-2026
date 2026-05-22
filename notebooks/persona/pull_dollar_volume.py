"""pull_dollar_volume.py - Warm the monthly dollar-volume cache (Feature 4).

Run with:
    python -m notebooks.persona.pull_dollar_volume

WARNING: pulls yfinance DAILY data for the whole S&P 500 union over the
project window, then derives the trailing-21-day average dollar volume and
samples it at month-ends. The daily download is large (~960 tickers x ~20
years) and chunked 100/call with a polite delay; first run takes several
minutes, later runs hit the parquet cache.

Why this exists
---------------
The provided CRSP MSF extract has no volume column and the Sharadar SEP
table is sample-only (ends 2018), so yfinance daily close x volume is the
only full-window source for Feature 4 (Dollar Volume). See DECISIONS
2026-05-22 'Dollar volume (Feature 4) from yfinance daily'.

Cache written
-------------
    data/processed/yfinance_dollar_volume_monthly.parquet
"""

from __future__ import annotations

from src.data_loader import _sp500_union_in_window, load_dollar_volume_monthly


UNIVERSE_START = "2005-01-01"
UNIVERSE_END = "2025-12-31"


def main() -> int:
    print("=" * 70)
    print("Warm dollar-volume cache: S&P 500 union, monthly (Feature 4)")
    print("=" * 70)

    universe = _sp500_union_in_window(UNIVERSE_START, UNIVERSE_END)
    print(f"\nUniverse: {len(universe)} tickers, {UNIVERSE_START} -> {UNIVERSE_END}\n")

    dv = load_dollar_volume_monthly(
        start=UNIVERSE_START, end=UNIVERSE_END, universe=universe
    )

    n_tickers = dv.index.get_level_values("ticker").nunique()
    d = dv.index.get_level_values("date")
    print()
    print(f"  -> {len(dv):,} rows, {n_tickers} tickers, "
          f"{d.min().date()} -> {d.max().date()}")
    missing = sorted(set(universe) - set(dv.index.get_level_values("ticker")))
    print(f"  -> {len(missing)} of {len(universe)} tickers had no yfinance "
          f"volume (delisted/renamed). First 10: {missing[:10]}")
    print("\nDone. load_dollar_volume_monthly() now serves from cache.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
