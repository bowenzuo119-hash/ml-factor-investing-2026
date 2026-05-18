"""freeze_returns_panel.py - Pre-cook the spliced returns panel for Person B.

Run with:
    python -m notebooks.persona.freeze_returns_panel

What this does
--------------
1. Calls `load_prices_spliced(start, end)` to get the full CRSP+yfinance
   panel (uses the parquet cache built by Person A; ~5 sec on cache hit).
2. Pivots the long (date, ticker) frame into a wide returns matrix
   (rows = month-ends, cols = ticker, values = monthly total return).
3. Saves the result to `data/processed/returns_spliced_2019_2024.parquet`.

Why this exists
---------------
Person B's model code shouldn't care about the existence of the splice
or which functions to call. With this frozen panel they can just::

    import pandas as pd
    returns = pd.read_parquet(
        "data/processed/returns_spliced_2019_2024.parquet"
    )

and start training. If we change the splice mechanics later, we re-run
this script and B's code keeps working.

The frozen panel is NOT in git (parquet caches are gitignored).
Re-run this script after any change to the loaders to refresh it.
"""

from __future__ import annotations

from pathlib import Path

from src.data_loader import load_prices_spliced, PROCESSED_DIR


START = "2019-01-01"
END = "2024-12-31"

# Filename encodes the window so it's clear which slice the file holds.
OUTPUT_FILE = PROCESSED_DIR / "returns_spliced_2019_2024.parquet"


def main() -> int:
    print("=" * 70)
    print(f"Freezing spliced returns panel: {START} -> {END}")
    print("=" * 70)

    # 1. Load spliced data (uses cache if available)
    print(f"\n[1/3] Loading spliced panel via load_prices_spliced...")
    panel = load_prices_spliced(start=START, end=END)
    src_counts = panel.groupby("source").size().to_dict()
    print(
        f"      {len(panel):,} rows, "
        f"{panel.index.get_level_values('ticker').nunique()} tickers, "
        f"{panel.index.get_level_values('date').nunique()} months"
    )
    print(f"      sources: {src_counts}")

    # 2. Pivot to wide format: rows = date, cols = ticker, values = ret
    print("\n[2/3] Pivoting to wide returns matrix (date x ticker)...")
    returns_wide = panel["ret"].unstack(level="ticker").sort_index()
    nan_frac = returns_wide.isna().sum().sum() / returns_wide.size
    print(
        f"      shape: {returns_wide.shape[0]} months x "
        f"{returns_wide.shape[1]} tickers  "
        f"(NaN fraction: {nan_frac:.1%})"
    )

    # 3. Save
    print(f"\n[3/3] Writing {OUTPUT_FILE}...")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    returns_wide.to_parquet(OUTPUT_FILE, compression="snappy")
    size_kb = OUTPUT_FILE.stat().st_size / 1024
    print(f"      wrote {size_kb:.1f} KB")

    print(
        f"\nDone. Person B can now read with:"
        f"\n  import pandas as pd"
        f"\n  returns = pd.read_parquet(\"{OUTPUT_FILE.relative_to(Path.cwd()) if OUTPUT_FILE.is_absolute() else OUTPUT_FILE}\")"
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
