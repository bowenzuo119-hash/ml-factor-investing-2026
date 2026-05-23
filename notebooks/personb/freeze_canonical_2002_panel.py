"""Freeze the OFFICIAL CANONICAL returns panel: 2002-04 → 2024-12.

Supersedes the 2003-2024 canonical panel from `freeze_canonical_panel.py`.
Rationale:

  * Sharadar SF1 fundamentals coverage of the S&P 500 stabilises at ~73-75%
    by April 2002 (Jan-Mar 2002 dip to 69-72% as Q4-2001 filings come in).
    Starting at 2002-04 gives data parity with the 2003 panel.
  * 2002-04 start gives walk-forward a first prediction at 2012-04-30,
    extending long-OOS from 12 years (2013-2024) to ~12.75 years.
  * 2024-12-31 end unchanged.

Run with:
    .venv/bin/python -m notebooks.personb.freeze_canonical_2002_panel
"""
from __future__ import annotations

from src.data_loader import PROCESSED_DIR, load_prices_spliced


START = "2002-04-01"
END = "2024-12-31"
OUTPUT_FILE = PROCESSED_DIR / "returns_spliced_2002_2024.parquet"


def main() -> int:
    print(f"Freezing CANONICAL returns panel: {START} -> {END}")
    panel = load_prices_spliced(start=START, end=END)
    print(
        f"  loaded {len(panel):,} rows, "
        f"{panel.index.get_level_values('ticker').nunique()} tickers, "
        f"{panel.index.get_level_values('date').nunique()} months"
    )
    print(f"  sources: {panel.groupby('source').size().to_dict()}")

    returns_wide = panel["ret"].unstack(level="ticker").sort_index()
    nan_frac = returns_wide.isna().sum().sum() / returns_wide.size
    print(
        f"  pivoted to {returns_wide.shape[0]} months x "
        f"{returns_wide.shape[1]} tickers (NaN fraction {nan_frac:.1%})"
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    returns_wide.to_parquet(OUTPUT_FILE, compression="snappy")
    size_mb = OUTPUT_FILE.stat().st_size / 1024**2
    print(f"  wrote {OUTPUT_FILE.name} ({size_mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
