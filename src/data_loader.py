"""data_loader.py - Data loading and preprocessing utilities.

This module owns all I/O against external data sources and the on-disk
parquet cache under `data/raw/` and `data/processed/`. No other module in
`src/` should hit the network or read raw CSVs directly - they import from
here.

Sources (see DECISIONS.md for the why):
    * Monthly equity prices and returns: CRSP MSF (vendor-provided CSV in
      `data/raw/CRSPData_*.csv`), loaded via `load_prices`.
    * S&P 500 historical membership: fja05680/sp500 GitHub CSVs, loaded via
      `load_sp500_membership`.
    * Macro / regime features: FRED via `pandas-datareader` (TODO,
      `load_macro`).

Design principles:
    * Read each raw source at most once; persist a cleaned version to
      `data/processed/*.parquet` and serve from there on subsequent calls.
    * Return long-format DataFrames with a `(date, permno)` MultiIndex so
      downstream factor / backtest code can pivot however it likes.
    * Every function is deterministic given its arguments + the cached
      files; no hidden randomness, no "latest" magic.
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

# CRSP Monthly Stock File (vendor-provided; not in git, not downloaded by
# this module). Shared by a course TA in spring 2026; covers 1925-12 to
# 2022-12, all US listed common stocks (~37k unique PERMNOs).
CRSP_MONTHLY_FILE = "CRSPData_1925_2022.csv"
CRSP_MONTHLY_CACHE = "crsp_monthly.parquet"


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


def _load_crsp_monthly_raw(*, force_rebuild: bool = False) -> pd.DataFrame:
    """Read the local CRSP MSF CSV, clean it, and cache to parquet.

    Handles the CRSP-specific data conventions that would otherwise bite
    downstream code:

    * **Negative PRC** = bid-ask midpoint (no trade that month); take
      ``abs()`` to get a usable price. Zero or NaN PRC -> NaN.
    * **RET / RETX special-character codes** (``"B"``, ``"C"``, etc.)
      indicate missing or first-period returns; coerced to NaN.
    * **PERMNO** (not TICKER) is the stable identifier: tickers change
      when companies restructure, but PERMNO is permanent. Output is
      indexed by ``(date, permno)``; ``ticker`` is kept as a column for
      human readability only.
    * **SHROUT** is in thousands of shares; ``market_cap`` is computed
      as ``price * shrout * 1_000`` (USD).

    Parameters
    ----------
    force_rebuild : bool, keyword-only, default False
        If False, read from the cached parquet when it exists (~1-2 sec).
        If True, re-parse the 471 MB CSV from scratch (~30-60 sec).

    Returns
    -------
    pd.DataFrame
        Long-format frame indexed by ``(date, permno)`` MultiIndex with
        columns: ``ticker, comnam, cusip, sic_code, price, ret, retx,
        bid, ask, shrout, market_cap, bid_ask_spread``.
    """
    cache_path = PROCESSED_DIR / CRSP_MONTHLY_CACHE
    if cache_path.exists() and not force_rebuild:
        print(f"[load_prices] reading cache: {cache_path.name}")
        return pd.read_parquet(cache_path)

    raw_path = RAW_DIR / CRSP_MONTHLY_FILE
    if not raw_path.exists():
        raise FileNotFoundError(
            f"CRSP monthly file not found at {raw_path}.\n"
            f"Place {CRSP_MONTHLY_FILE} in data/raw/ and retry. The file "
            f"is vendor-provided (shared by the course TA in spring 2026); "
            f"see DECISIONS.md 2026-05-13 'Switch primary price source'."
        )

    print(f"[load_prices] reading {raw_path.name} (~30-60 sec, 471 MB)...")
    df = pd.read_csv(raw_path, parse_dates=["date"], low_memory=False)

    print(f"[load_prices] cleaning {len(df):,} rows...")

    # Strip whitespace from string columns
    for col in ("TICKER", "COMNAM", "CUSIP", "SICCD"):
        df[col] = df[col].astype("string").str.strip()

    # CRSP convention: negative PRC = bid-ask midpoint (no trade).
    # abs() recovers a usable price; zeros become NaN (no data).
    price = df["PRC"].abs()
    df["price"] = price.where(price > 0, other=pd.NA)

    # RET / RETX may contain alpha codes ("B" = no valid return,
    # "C" = first listing month) - coerce to numeric, codes -> NaN.
    df["ret"] = pd.to_numeric(df["RET"], errors="coerce")
    df["retx"] = pd.to_numeric(df["RETX"], errors="coerce")

    # Derived columns
    df["market_cap"] = df["price"] * df["SHROUT"] * 1_000  # SHROUT is thousands
    midpoint = (df["BID"] + df["ASK"]) / 2
    df["bid_ask_spread"] = (df["ASK"] - df["BID"]) / midpoint

    # Rename to consistent lowercase, drop raw columns we've replaced
    df = df.rename(
        columns={
            "PERMNO": "permno",
            "TICKER": "ticker",
            "COMNAM": "comnam",
            "CUSIP": "cusip",
            "SICCD": "sic_code",
            "BID": "bid",
            "ASK": "ask",
            "SHROUT": "shrout",
        }
    ).drop(columns=["PRC", "RET", "RETX"])

    # Sorted MultiIndex on (date, permno) for fast date-range slicing
    df = df.set_index(["date", "permno"]).sort_index()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, compression="snappy")
    size_mb = cache_path.stat().st_size / 1024**2
    print(f"[load_prices] cached to {cache_path.name} ({size_mb:.1f} MB)")
    return df


def load_prices(
    *,
    start: str | None = None,
    end: str | None = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Load monthly CRSP stock-level prices, returns, and market data.

    Source is the local CRSP MSF CSV in ``data/raw/`` (see
    :data:`CRSP_MONTHLY_FILE`). The first call parses and caches; subsequent
    calls read the parquet cache (~1-2 sec).

    Parameters
    ----------
    start : str, optional
        Inclusive start date (ISO ``"YYYY-MM-DD"``). ``None`` means earliest
        available (1925-12-31).
    end : str, optional
        Inclusive end date. ``None`` means latest available (2022-12-30).
    force_rebuild : bool, keyword-only, default False
        Re-parse the raw CSV instead of using the parquet cache. Use after
        the vendor file has been updated.

    Returns
    -------
    pd.DataFrame
        Long-format frame indexed by ``(date, permno)`` MultiIndex with
        columns:

        ============== ===========================================
        Column         Description
        ============== ===========================================
        ticker         Current trading symbol (changes over time)
        comnam         Company name
        cusip          CUSIP identifier (for joins with other data)
        sic_code       4-digit SIC industry code (string)
        price          Month-end price in USD (abs(PRC), so the
                       bid-ask-midpoint sign is dropped)
        ret            Monthly total return (with dividends)
        retx           Monthly return excluding dividends
        bid, ask       Month-end bid/ask quotes (USD)
        shrout         Shares outstanding (thousands)
        market_cap     Market capitalization (USD)
        bid_ask_spread Relative bid-ask spread, (ask-bid)/midpoint
        ============== ===========================================

    Notes
    -----
    * Universe is *all* US-listed common stocks (~37k unique PERMNOs over
      the full sample). To restrict to S&P 500 only, filter on the result
      using :func:`load_sp500_membership` (PERMNO/ticker mapping pending,
      see DECISIONS.md 2026-05-13).
    * Data ends 2022-12-30. The Project Framework's 2019-2024 test window
      will be served by a future yfinance splice for 2023-2024 (DECISIONS
      2026-05-13 'Yfinance splice for 2023-2024').
    * Fundamentals (B/M, E/P, D/P) are NOT in this dataset; that's
      Compustat, not CRSP. See DECISIONS 2026-05-13 'Defer fundamentals'.

    Raises
    ------
    FileNotFoundError
        If the CRSP CSV is not in ``data/raw/``.
    """
    df = _load_crsp_monthly_raw(force_rebuild=force_rebuild)
    if start is None and end is None:
        return df
    start_ts = pd.Timestamp(start) if start else df.index.get_level_values("date").min()
    end_ts = pd.Timestamp(end) if end else df.index.get_level_values("date").max()
    return df.loc[(slice(start_ts, end_ts), slice(None)), :]


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
    # ---- S&P 500 membership (from fja05680) ----
    paths = download_sp500_universe()
    print()
    print("S&P 500 membership files on disk:")
    for key, path in paths.items():
        size_kb = path.stat().st_size / 1024
        print(f"  {key:12s} {path}  ({size_kb:.1f} KB)")

    print()
    print("Universe-size smoke test:")
    for asof in ["2008-09-15", "2015-06-30", "2020-03-23", "2024-12-31"]:
        tickers = load_sp500_membership(asof)
        print(f"  {asof}: {len(tickers)} tickers (first 5: {tickers[:5]})")

    # ---- CRSP monthly prices ----
    print()
    print("=" * 70)
    print("CRSP monthly prices smoke test")
    print("=" * 70)
    prices = load_prices(start="2008-01-01", end="2008-12-31")
    print()
    print(f"Shape: {prices.shape[0]:,} rows x {prices.shape[1]} cols")
    print(f"Date range: {prices.index.get_level_values('date').min().date()} "
          f"-> {prices.index.get_level_values('date').max().date()}")
    print(f"Unique stocks (PERMNOs): {prices.index.get_level_values('permno').nunique():,}")
    print()
    print("Columns + dtypes:")
    print(prices.dtypes.to_string())
    print()
    print("First 5 rows:")
    print(prices.head().to_string())

    # Famous-event spot check: LEH (Lehman) should have a final row in Sept 2008
    print()
    print("Famous-event spot check (Lehman / Sept 2008):")
    leh = prices.xs("LEH", level=0, drop_level=False, axis=0) if False else \
          prices[prices["ticker"] == "LEH"]
    if len(leh):
        print(f"  LEH has {len(leh)} rows in 2008")
        last_row = leh.tail(1)
        last_date = last_row.index.get_level_values("date")[0].date()
        last_ret = last_row["ret"].iloc[0]
        print(f"  Last available month: {last_date}, monthly return: {last_ret:.2%}")
    else:
        print("  LEH not found - data integrity bug?")
