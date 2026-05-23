"""Freeze the OFFICIAL CANONICAL returns panel: 2003-2024.

The project's canonical configuration as of 2026-05-23:
  * 2003-01-01 start (gives walk-forward a first prediction at 2013-01-31,
    extending long-OOS from 10 years to 12)
  * 2024-12-31 end (matches the framework's prescribed test window;
    excludes 2025 which we have data for but where the strategy degraded)

Earlier panels (2005-2024 from `freeze_long_panel.py`, 2003-2025 from
`freeze_extended_panel.py`) are kept on disk for reproducibility but
canonical = this one.

Run with:
    .venv/bin/python -m notebooks.personb.freeze_canonical_panel
"""
from __future__ import annotations

from src.data_loader import PROCESSED_DIR, load_prices_spliced


START = "2003-01-01"
END = "2024-12-31"
OUTPUT_FILE = PROCESSED_DIR / "returns_spliced_2003_2024.parquet"


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
