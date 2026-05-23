"""Freeze the spliced returns panel for the extended 2003-2025 sample.

Same machinery as `freeze_long_panel.py` (which covers 2005-2024), just
a wider window. The longer panel gives the walk-forward backtest a
first OOS prediction at 2013-01-31 (vs 2015-01-31 with the 2005 start),
extending the long-OOS window from 10 to 13 years.

Run with:
    .venv/bin/python -m notebooks.personb.freeze_extended_panel
"""
from __future__ import annotations

from src.data_loader import PROCESSED_DIR, load_prices_spliced


START = "2003-01-01"
END = "2025-12-31"
OUTPUT_FILE = PROCESSED_DIR / "returns_spliced_2003_2025.parquet"


def main() -> int:
    print(f"Freezing spliced returns panel: {START} -> {END}")

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
