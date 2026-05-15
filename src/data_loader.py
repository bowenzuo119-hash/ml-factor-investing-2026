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

import pandas as pd

# Project-relative data directory. Files here are gitignored; the *code* that
# produces them lives in this module.
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


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
    2015 roster, not today's. Source TBD (Wikipedia historical revisions,
    CRSP, or a manually curated CSV checked into `data/reference/`).

    Parameters
    ----------
    asof : str
        ISO date. Returns the membership that was in force at end-of-day on
        this date.

    Returns
    -------
    list[str]
        Tickers, sorted alphabetically for reproducibility.
    """
    raise NotImplementedError("load_sp500_membership: needs a point-in-time source")
