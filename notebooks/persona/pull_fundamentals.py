"""pull_fundamentals.py - One-time bulk pull of Sharadar SF1 fundamentals.

Run with:
    python -m notebooks.persona.pull_fundamentals

WARNING: this makes real Nasdaq Data Link API calls against the paid
Sharadar SF1 subscription (needs NASDAQ_DATA_LINK_API_KEY in .env). It
is the bulk-fetch that warms the per-dimension parquet caches; once those
exist, load_fundamentals() serves from disk with no key and no network.

What this does
--------------
1. Resolves the universe = every ticker that was in the S&P 500 at any
   point in the project window (2005-2025), via _sp500_union_in_window.
2. Pulls TWO dimensions, because the two value factors need different ones:
     * ARQ (As-Reported Quarterly)  -> book equity for B/M
     * ART (As-Reported TTM)        -> trailing-12m net income for E/P
   (A single-quarter ARQ net income makes E/P ~4x too small / noisy --
   confirmed against the vendor pe column during validation.)
3. Starts the pull at 2003-01-01, two years before the 2005 backtest
   start, so the earliest 2005 rebalances have a recent filing to look up
   (filings post 30-90 days after period end -> need pre-2005 periods).

Caches written
--------------
    data/processed/sharadar_sf1_ARQ.parquet
    data/processed/sharadar_sf1_ART.parquet
"""

from __future__ import annotations

from src.data_loader import _sp500_union_in_window, load_fundamentals


# Project window per DECISIONS / Framework: 2005-2025. Pull from 2003 so the
# first 2005 rebalances have a filed-and-public fundamental to join against.
UNIVERSE_START = "2005-01-01"
UNIVERSE_END = "2025-12-31"
PULL_START = "2003-01-01"


def main() -> int:
    print("=" * 70)
    print("Bulk pull: Sharadar SF1 fundamentals for the S&P 500 union")
    print("=" * 70)

    universe = _sp500_union_in_window(UNIVERSE_START, UNIVERSE_END)
    print(
        f"\nUniverse: {len(universe)} tickers in S&P 500 at some point "
        f"{UNIVERSE_START} -> {UNIVERSE_END}"
    )
    print(f"Pulling fundamentals from {PULL_START} (2yr buffer before 2005).\n")

    # ARQ -> book equity for B/M; ART -> TTM net income for E/P.
    for dimension, why in [("ARQ", "book equity for B/M"),
                           ("ART", "TTM net income for E/P")]:
        print("-" * 70)
        print(f"Dimension {dimension}  ({why})")
        print("-" * 70)
        fund = load_fundamentals(
            tickers=universe,
            start=PULL_START,
            end=None,
            dimension=dimension,
        )
        n_tickers = fund.index.get_level_values("ticker").nunique()
        dk = fund.index.get_level_values("datekey")
        print(
            f"  -> {len(fund):,} rows, {n_tickers} tickers, "
            f"datekey {dk.min().date()} -> {dk.max().date()}"
        )
        missing = sorted(set(universe) - set(fund.index.get_level_values("ticker")))
        print(
            f"  -> {len(missing)} of {len(universe)} universe tickers had no "
            f"SF1 data (delisted/never-covered). First 10: {missing[:10]}"
        )
        print()

    print("Done. load_fundamentals() now serves both dimensions from cache.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
