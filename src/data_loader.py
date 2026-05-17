"""data_loader.py - Data loading and preprocessing utilities.

This module owns all I/O against external data sources (Yahoo Finance for
equity prices, FRED for macro series) and the on-disk parquet cache under
`data/raw/` and `data/processed/`. No other module in `src/` should hit the
network directly - they import from here.

Design principles:
    * Hit the network at most once per (ticker, date-range) tuple; persist
      everything to parquet immediately.
    * Adjusted close ONLY (handles splits + dividends). Raw close is a foot-gun.
    * Return a long-format DataFrame with a (date, ticker) MultiIndex so the
      downstream factor / backtest code can pivot however it likes.
    * Every function is deterministic given its arguments + the cached files;
      no hidden randomness, no "latest" magic.
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

# Project-relative data directory. Files here are gitignored; the *code* that
# produces them lives in this module.
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# S&P 500 historical membership source: github.com/fja05680/sp500
# Maintained dataset of point-in-time index constituents, used to avoid
# survivorship bias when building the investable universe.
SP500_REPO_RAW = "https://raw.githubusercontent.com/fja05680/sp500/master"
SP500_MEMBERSHIP_FILE = "sp500_ticker_start_end.csv"
SP500_CURRENT_FILE = "sp500.csv"


def download_sp500_universe(*, force: bool = False) -> dict[str, Path]:
    """Download S&P 500 historical membership data to ``data/raw/``.

    Pulls two CSVs from the fja05680/sp500 GitHub repository:

    * ``sp500_ticker_start_end.csv`` - one row per (ticker, membership spell)
      with columns ``ticker, start_date, end_date``. A blank ``end_date``
      means the ticker is still in the index. This is the canonical source
      for the point-in-time universe.
    * ``sp500.csv`` - the current 500 constituents with GICS sector,
      sub-industry, headquarters location, and date first added. Used as a
      sector-classification reference table.

    Parameters
    ----------
    force : bool, keyword-only, default False
        If False (default), skip the download when the file already exists
        locally. If True, re-download and overwrite.

    Returns
    -------
    dict[str, Path]
        Mapping from a short key (``"membership"``, ``"current"``) to the
        local path of each downloaded file.

    Notes
    -----
    Together these two files are ~5 MB. Re-running this with ``force=False``
    is effectively free (a stat call per file), so other loader functions
    can call it unconditionally to ensure the cache is populated.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    targets = {
        "membership": (SP500_MEMBERSHIP_FILE, RAW_DIR / SP500_MEMBERSHIP_FILE),
        "current": (SP500_CURRENT_FILE, RAW_DIR / SP500_CURRENT_FILE),
    }

    for key, (filename, local_path) in targets.items():
        if local_path.exists() and not force:
            print(f"[download_sp500_universe] cached: {local_path.name}")
            continue
        url = f"{SP500_REPO_RAW}/{filename}"
        print(f"[download_sp500_universe] fetching {url}")
        urlretrieve(url, local_path)
        print(f"[download_sp500_universe] saved:  {local_path}")

    return {key: path for key, (_, path) in targets.items()}


def _load_membership_table() -> pd.DataFrame:
    """Read the cached membership CSV into a typed DataFrame.

    Auto-downloads if the file is missing. Parses dates and treats a blank
    ``end_date`` as "still active" (mapped to ``pd.NaT``, which compares
    correctly against any asof date when wrapped in fillna).

    Returns
    -------
    pd.DataFrame
        Columns: ``ticker`` (str), ``start_date`` (datetime64[ns]),
        ``end_date`` (datetime64[ns], NaT means still in the index).
    """
    download_sp500_universe()  # no-op if cached
    path = RAW_DIR / SP500_MEMBERSHIP_FILE
    df = pd.read_csv(path, parse_dates=["start_date", "end_date"])
    df["ticker"] = df["ticker"].astype(str).str.strip()
    return df


def load_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load daily adjusted-close prices for a list of tickers.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols as recognised by Yahoo Finance (e.g. `["AAPL", "MSFT"]`).
        Order is not preserved in the output; pivot on the `ticker` index level
        if you need a specific column order.
    start : str
        Inclusive start date in ISO format, e.g. `"2010-01-01"`.
    end : str, optional
        Exclusive end date in ISO format. `None` (default) means "up to the
        latest available trading day".
    use_cache : bool, keyword-only, default True
        If True, read from `data/raw/prices.parquet` when the requested
        (tickers, start, end) window is a subset of what is already cached,
        and only hit the network for the missing slice. Set to False to force
        a fresh pull (e.g. after a corporate action).

    Returns
    -------
    pd.DataFrame
        Long-format frame indexed by a `(date, ticker)` MultiIndex with a
        single column `adj_close` of dtype `float64`. Dates are
        timezone-naive `pd.Timestamp` at midnight UTC. Missing observations
        (e.g. ticker not yet listed) are *dropped*, not filled - downstream
        code must handle ragged panels explicitly.

    Raises
    ------
    ValueError
        If `tickers` is empty, or if `start >= end`.
    RuntimeError
        If the yfinance request returns zero rows for every ticker (usually a
        network or rate-limit issue).

    Notes
    -----
    Survivorship bias warning: yfinance only knows about *currently listed*
    tickers. For a backtest that does not lie, `tickers` must come from a
    point-in-time S&P 500 membership table, not today's index. See
    `load_sp500_membership` (TODO) for the historical roster.
    """
    raise NotImplementedError(
        "load_prices: implement in the next commit on persona-data-pipeline"
    )


def load_macro(
    series_ids: list[str],
    start: str,
    end: str | None = None,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load macro/regime features from FRED.

    Parameters
    ----------
    series_ids : list[str]
        FRED series IDs (e.g. `["VIXCLS", "T10Y2Y", "BAA10Y"]`).
    start, end : str
        Same semantics as :func:`load_prices`.
    use_cache : bool, keyword-only, default True
        Reuse `data/raw/macro.parquet` when possible.

    Returns
    -------
    pd.DataFrame
        Wide-format frame indexed by `date` with one column per series ID.
        Forward-filled to business-day frequency (macro series are released
        less often than equities trade); the original release date is
        preserved in a sibling column `<series_id>__asof` to avoid
        look-ahead bias when joining onto the equity panel.
    """
    raise NotImplementedError(
        "load_macro: implement after load_prices is green"
    )


def load_sp500_membership(asof: str) -> list[str]:
    """Return the S&P 500 constituent tickers as of a given date.

    Critical for avoiding survivorship bias: a 2015 backtest must use the
    2015 roster, not today's. Source: github.com/fja05680/sp500 (see
    :func:`download_sp500_universe`), which provides per-ticker membership
    spells back to the 1990s.

    Parameters
    ----------
    asof : str
        ISO date (``"YYYY-MM-DD"``). Returns the membership that was in
        force at end-of-day on this date.

    Returns
    -------
    list[str]
        Tickers, sorted alphabetically for reproducibility. Empty list if
        the date is before any recorded membership starts.

    Raises
    ------
    ValueError
        If ``asof`` cannot be parsed as a date.
    """
    asof_ts = pd.Timestamp(asof)
    table = _load_membership_table()
    # A ticker was in the index on `asof` iff its spell had started by then
    # and had not yet ended. Open spells have NaT end_date; treat as +inf.
    started = table["start_date"] <= asof_ts
    not_ended = table["end_date"].isna() | (table["end_date"] >= asof_ts)
    active = table.loc[started & not_ended, "ticker"]
    return sorted(active.unique().tolist())


# --------------------------------------------------------------------------
# Script entry point - run the data pipeline end-to-end.
# --------------------------------------------------------------------------

if __name__ == "__main__":
    paths = download_sp500_universe()
    print()
    print("Files on disk:")
    for key, path in paths.items():
        size_kb = path.stat().st_size / 1024
        print(f"  {key:12s} {path}  ({size_kb:.1f} KB)")

    # Smoke test: how many tickers were in the index at a few historical dates?
    print()
    print("Universe-size smoke test:")
    for asof in ["2008-09-15", "2015-06-30", "2020-03-23", "2024-12-31"]:
        tickers = load_sp500_membership(asof)
        print(f"  {asof}: {len(tickers)} tickers (first 5: {tickers[:5]})")
