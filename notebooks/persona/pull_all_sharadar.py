"""pull_all_sharadar.py - Block A: bulk-archive the full Sharadar tables.

One-time pull of the whole Sharadar tables to local parquet under
``data/raw/sharadar/`` before the subscription lapses. Raw data is gitignored
(``data/**``); only this script + ``data/raw/sharadar/MANIFEST.txt`` are
committed.

Usage:
    # validate the pipeline on tiny tables first
    python -m notebooks.persona.pull_all_sharadar --tables sp500 tickers

    # then the full archive (SEP/DAILY are multi-GB -- run in background)
    python -m notebooks.persona.pull_all_sharadar
    python -m notebooks.persona.pull_all_sharadar --tables sep daily sf1
    python -m notebooks.persona.pull_all_sharadar --force

Needs NASDAQ_DATA_LINK_API_KEY in .env. Idempotent: existing parquet files are
skipped unless --force, so an interrupted pull resumes where it left off.
"""

from __future__ import annotations

import argparse

from src.data_loader import bulk_download_all_sharadar, SHARADAR_BULK_TABLES


def main() -> int:
    ap = argparse.ArgumentParser(description="Bulk-pull full Sharadar tables.")
    ap.add_argument(
        "--tables", nargs="*", default=None,
        help=f"subset of {sorted(SHARADAR_BULK_TABLES)}; default = all",
    )
    ap.add_argument("--force", action="store_true", help="re-download existing")
    ap.add_argument(
        "--no-sf1-split", action="store_true", help="skip per-dimension SF1 splits"
    )
    args = ap.parse_args()

    tables = tuple(args.tables) if args.tables else None
    manifest = bulk_download_all_sharadar(
        tables=tables, force=args.force, sf1_split=not args.no_sf1_split,
    )
    print("\n=== MANIFEST ===")
    for key, s in manifest.items():
        print(f"  {key}: {s.get('rows', 0):,} rows, {s.get('mb', 0):.1f} MB")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
