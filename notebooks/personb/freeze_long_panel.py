"""Freeze the spliced returns panel for the full 2005-2024 sample.

Person B's long-window analogue of `notebooks/persona/freeze_returns_panel.py`,
which only goes back to 2019. The Project Framework's training window is
2005-2015 (§7.2), so the model needs at least that much history.

Run with:
    .venv/bin/python -m notebooks.personb.freeze_long_panel

What this does
--------------
Calls `load_prices_spliced(start="2005-01-01", end="2024-12-31")`, which
uses the CRSP MSF cache for 2005-2022 and yfinance for 2023-2024 (the
splice is validated in DECISIONS.md 2026-05-18). Pivots into a wide
returns matrix and writes it to data/processed/returns_spliced_2005_2024.parquet
for fast reuse by the evaluation notebooks.
"""
from __future__ import annotations

from src.data_loader import PROCESSED_DIR, load_prices_spliced


START = "2005-01-01"
END = "2024-12-31"
OUTPUT_FILE = PROCESSED_DIR / "returns_spliced_2005_2024.parquet"


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
