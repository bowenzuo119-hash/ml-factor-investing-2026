"""verify_sharadar_pulls.py - Block A2: sanity-check the bulk Sharadar archive.

For each pulled parquet under data/raw/sharadar/: assert it clears a minimum
row count, print row count / ticker count / date span / size, and sample 3
rows. Run AFTER `pull_all_sharadar.py`.

    python -m notebooks.persona.verify_sharadar_pulls

Row counts come from parquet metadata and stats from only the ticker/date
columns, so this stays fast and memory-light even on the ~46M-row SEP table.
Exits non-zero if any expected table is missing or below its minimum, so it
doubles as the hand-off gate ("all tables exist + look right").
"""

from __future__ import annotations

import sys

import pandas as pd
import pyarrow.parquet as pq

from src.data_loader import RAW_DIR, _sharadar_parquet_stats

pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 10)

# table file -> minimum expected rows
EXPECTED: dict[str, int] = {
    "tickers.parquet":    10_000,
    "sp500.parquet":       5_000,
    "actions.parquet":    50_000,
    "sf1_all.parquet":  1_000_000,
    "sf1_AR_arq.parquet": 300_000,
    "sf1_AR_art.parquet": 300_000,
    "sf1_MR_arq.parquet": 300_000,
    "daily.parquet":    5_000_000,
    "sep.parquet":      5_000_000,
}


def _sample(path, n: int = 3) -> pd.DataFrame:
    """First n rows without reading the whole file."""
    return next(pq.ParquetFile(path).iter_batches(batch_size=n)).to_pandas()


def main() -> int:
    failures: list[str] = []
    print(f"Verifying Sharadar archive under {RAW_DIR}\n" + "=" * 78)

    for fname, min_rows in EXPECTED.items():
        path = RAW_DIR / fname
        if not path.exists():
            print(f"[MISSING] {fname}")
            failures.append(f"{fname} missing")
            continue

        s = _sharadar_parquet_stats(path)
        status = "OK" if s["rows"] >= min_rows else "LOW"
        span = (
            f"{s.get('date_min', '?')}..{s.get('date_max', '?')}"
            if "date_min" in s else "n/a"
        )
        print(
            f"[{status:>4}] {fname:18s} {s['rows']:>12,} rows | "
            f"{str(s.get('n_tickers', '?')):>6} tickers | {span} | {s['mb']:8.1f} MB"
        )
        if s["rows"] < min_rows:
            failures.append(f"{fname}: {s['rows']:,} rows < expected {min_rows:,}")

        cols = [c for c in _sample(path, 1).columns][:8]
        print(_sample(path)[cols].to_string())
        print("-" * 78)

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nAll expected tables present and above minimum row counts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
