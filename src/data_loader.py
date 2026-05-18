"""data_loader.py - Data loading and preprocessing utilities.

This module owns all I/O against external data sources and the on-disk
parquet cache under `data/raw/` and `data/processed/`. No other module in
`src/` should hit the network or read raw CSVs directly - they import from
here.

Sources (see DECISIONS.md for the why):
    * Monthly equity prices and returns: CRSP MSF (vendor-provided CSV in
      `data/raw/CRSPData_*.csv`), loaded via `load_prices`. Covers 1925-12
      to 2022-12.
    * Out-of-sample tail (2023-2024): yfinance, loaded via
      `load_prices_yfinance`. Used to extend CRSP through the most recent
      year for live-data evaluation; school has no ongoing CRSP licence.
    * S&P 500 historical membership: fja05680/sp500 GitHub CSVs, loaded via
      `load_sp500_membership`.
    * Macro / regime features: FRED via `pandas-datareader`, loaded via
      `load_macro`.

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

# yfinance cache: union of all (ticker, month) ever fetched. Lets the loader
# serve overlapping requests (e.g. two backtests with different date windows
# on the same universe) without re-hitting the network. See
# `_load_yfinance_monthly_raw` for the cache hit/miss rules.
YFINANCE_MONTHLY_CACHE = "yfinance_monthly.parquet"

# FRED macro series for Person C's regime model. Picked per Project
# Framework §4.4 ("Macro for Person C: VIX, DGS10, DGS2, DBAA, DAAA, DFF").
# All have history back to at least 1990, plenty for the 2005-2024 window.
DEFAULT_MACRO_SERIES: tuple[str, ...] = (
    "VIXCLS",  # CBOE Volatility Index (VIX), daily close
    "DGS10",   # 10-Year Treasury constant-maturity yield
    "DGS2",    # 2-Year Treasury constant-maturity yield (term spread = DGS10-DGS2)
    "DBAA",    # Moody's Baa corporate bond yield
    "DAAA",    # Moody's Aaa corporate bond yield (credit spread = DBAA-DAAA)
    "DFF",     # Federal Funds effective rate
)
MACRO_CACHE = "macro_daily.parquet"


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


def _load_yfinance_monthly_raw(
    tickers: tuple[str, ...] | list[str],
    *,
    start: str,
    end: str,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Download daily OHLCV from yfinance, resample to month-end, cache to parquet.

    yfinance is the project's fallback price source for 2023+ since the
    school has no ongoing CRSP licence (DECISIONS 2026-05-18 'School has no
    CRSP — yfinance for 2023+'). This loader is schema-compatible with the
    CRSP loader so the two can be spliced:

        * Same MultiIndex layout (date, identifier) -- here ``ticker`` not
          ``permno`` because yfinance has no PERMNO concept.
        * Same monthly ``ret`` column, computed from auto-adjusted close
          (splits AND dividends adjusted, so equivalent to CRSP's total
          return).
        * Same month-end date convention (last available trading day of
          the month).

    Cache strategy: store union of all (ticker, date) ever fetched in one
    parquet. A cache HIT happens when (a) every requested ticker is in the
    cache AND (b) the cached date range fully covers the request. Otherwise
    the full request is re-fetched and merged in. Costs bandwidth on the
    first call but is essentially free thereafter.

    Parameters
    ----------
    tickers : tuple[str, ...] or list[str]
        Yahoo ticker symbols (e.g. ``["AAPL", "MSFT"]``). Order does not
        matter; symbols are uppercased and de-duplicated. Output is sorted
        by (date, ticker).
    start : str
        Inclusive start date (ISO).
    end : str
        Inclusive end date (ISO). yfinance's own ``end`` is exclusive; this
        loader bumps it by one day so the user-facing semantics match the
        rest of this module.
    force_rebuild : bool, keyword-only, default False
        Bypass the cache and re-download.

    Returns
    -------
    pd.DataFrame
        Long-format frame indexed by ``(date, ticker)`` MultiIndex with
        columns:

        =========== =================================================
        Column      Description
        =========== =================================================
        open        Month-end-day's open price (USD, split-adjusted)
        high        Month-end-day's high
        low         Month-end-day's low
        adj_close   Month-end-day's adjusted close (splits + divs)
        volume      Month-end-day's volume (shares)
        ret         Monthly total return derived from adj_close
        =========== =================================================

    Notes
    -----
    * The first month of any ticker's data has ``ret == NaN`` (no prior
      month to diff against). Downstream code should ``.dropna()`` or be
      tolerant; subsequent calls that extend the cache backward will fill
      in the missing return.
    * Tickers that return no data (delisted before ``start``, never existed,
      Yahoo outage) are logged and silently skipped. Caller must check
      ``df.index.get_level_values("ticker").unique()`` against what they
      asked for.
    """
    cache_path = PROCESSED_DIR / YFINANCE_MONTHLY_CACHE
    requested = tuple(sorted({str(t).strip().upper() for t in tickers}))
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    # --- Cache hit path -------------------------------------------------
    # Compare in month-end space: a request for [2022-12-01, 2023-12-31]
    # actually wants month-ends 2022-12-31 ... 2023-12-31, so cache that
    # spans those month-ends is a hit even if the literal `start_ts` falls
    # before any cached date.
    expected_months = pd.date_range(start_ts, end_ts, freq="ME")
    if not force_rebuild and cache_path.exists() and len(expected_months) > 0:
        cached = pd.read_parquet(cache_path)
        cached_tickers = set(cached.index.get_level_values("ticker"))
        cached_dates = cached.index.get_level_values("date")
        needed_min, needed_max = expected_months[0], expected_months[-1]
        if (
            set(requested).issubset(cached_tickers)
            and cached_dates.min() <= needed_min
            and cached_dates.max() >= needed_max
        ):
            print(
                f"[load_yfinance] cache hit: {len(requested)} tickers, "
                f"{start_ts.date()} -> {end_ts.date()}"
            )
            d = cached.index.get_level_values("date")
            t = cached.index.get_level_values("ticker")
            mask = (d >= start_ts) & (d <= end_ts) & t.isin(requested)
            return cached.loc[mask].sort_index()

    # --- Cache miss: fetch from Yahoo -----------------------------------
    import yfinance as yf

    # yfinance's `end` is exclusive; bump by 1 day for inclusive semantics.
    yf_end = (end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    print(
        f"[load_yfinance] downloading {len(requested)} tickers "
        f"({start_ts.date()} -> {end_ts.date()})..."
    )
    raw = yf.download(
        tickers=list(requested),
        start=start,
        end=yf_end,
        interval="1d",
        auto_adjust=True,    # Close column = split+dividend adjusted
        progress=False,
        threads=True,
        group_by="ticker",
    )

    # Column layout depends on ticker count:
    #   * len > 1  -> MultiIndex (ticker, field)
    #   * len == 1 -> flat columns; wrap to look the same.
    if isinstance(raw.columns, pd.MultiIndex):
        per_ticker = {
            t: raw[t]
            for t in requested
            if t in raw.columns.get_level_values(0)
        }
    else:
        per_ticker = {requested[0]: raw}

    frames: list[pd.DataFrame] = []
    for tkr, sub in per_ticker.items():
        if sub.empty or sub["Close"].dropna().empty:
            print(f"[load_yfinance] no data: {tkr}")
            continue
        # Resample daily -> month-end: take the last trading day's row.
        monthly = sub.resample("ME").last()
        monthly["ret"] = monthly["Close"].pct_change()
        monthly = monthly.rename(
            columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "adj_close", "Volume": "volume",
            }
        )
        monthly["ticker"] = tkr
        monthly.index.name = "date"
        frames.append(
            monthly.reset_index()
            .set_index(["date", "ticker"])
            [["open", "high", "low", "adj_close", "volume", "ret"]]
        )

    if not frames:
        raise RuntimeError(
            f"[load_yfinance] no data returned for any of {len(requested)} "
            f"tickers. Check tickers and network."
        )

    fresh = pd.concat(frames).sort_index()

    # Merge with existing cache to grow the union, then persist.
    if not force_rebuild and cache_path.exists():
        cached = pd.read_parquet(cache_path)
        merged = pd.concat([cached, fresh])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    else:
        merged = fresh

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(cache_path, compression="snappy")
    size_kb = cache_path.stat().st_size / 1024
    print(
        f"[load_yfinance] cached {len(merged):,} rows total ({size_kb:.1f} KB)"
    )

    d = merged.index.get_level_values("date")
    t = merged.index.get_level_values("ticker")
    mask = (d >= start_ts) & (d <= end_ts) & t.isin(requested)
    return merged.loc[mask].sort_index()


def _load_macro_raw(
    series_ids: tuple[str, ...],
    *,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Fetch FRED series, ffill to business-day frequency, cache to parquet.

    The cache stores the *union* of all series ever fetched, so subsequent
    calls with a subset of series are served entirely from the cache.
    Adding a never-fetched series triggers a refresh of the full bundle
    (FRED is fast enough that fine-grained caching is overkill).

    Parameters
    ----------
    series_ids : tuple[str, ...]
        FRED series IDs (e.g. ``("VIXCLS", "DGS10")``). Tuple (not list)
        because tuples are hashable and the cache key is order-independent.
    force_rebuild : bool, default False
        Re-fetch from FRED even if the cache covers the requested series.

    Returns
    -------
    pd.DataFrame
        Wide-format frame indexed by ``date`` (business-day frequency)
        with one column per requested series ID. Missing observations
        (weekends, holidays, gaps before a series existed) are
        forward-filled; the first available value is the oldest non-null
        observation FRED has on file.
    """
    cache_path = PROCESSED_DIR / MACRO_CACHE
    requested = set(series_ids)

    if cache_path.exists() and not force_rebuild:
        cached = pd.read_parquet(cache_path)
        if requested.issubset(cached.columns):
            print(f"[load_macro] cache hit: {sorted(requested)}")
            return cached[list(series_ids)]
        missing = requested - set(cached.columns)
        print(f"[load_macro] cache missing {missing}; refetching the bundle")

    # Heavy import: deferred so callers not touching FRED don't pay for it.
    import pandas_datareader.data as pdr

    print(f"[load_macro] fetching {len(series_ids)} FRED series since 1990...")
    df = pdr.DataReader(list(series_ids), "fred", start="1990-01-01")
    df.index.name = "date"

    # FRED daily series occasionally have NaN on holidays / when a series
    # didn't yet exist. Forward-fill to a clean business-day index so
    # downstream join-on-date code doesn't have to handle gaps.
    bdays = pd.bdate_range(df.index.min(), df.index.max())
    df = df.reindex(bdays).ffill()
    df.index.name = "date"

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, compression="snappy")
    size_kb = cache_path.stat().st_size / 1024
    print(f"[load_macro] cached to {cache_path.name} ({size_kb:.1f} KB)")
    return df


def load_macro(
    *,
    start: str | None = None,
    end: str | None = None,
    series_ids: tuple[str, ...] | list[str] | None = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Load macro / regime features from FRED.

    First call fetches over the network (~5-10 sec for the default bundle);
    subsequent calls read the parquet cache.

    Parameters
    ----------
    start : str, optional
        Inclusive start date (ISO ``"YYYY-MM-DD"``). ``None`` means earliest
        FRED has on file (1990 for the loader's default fetch window).
    end : str, optional
        Inclusive end date. ``None`` means latest available.
    series_ids : tuple[str, ...] or list[str], optional
        FRED series IDs to return. ``None`` (default) returns the
        Framework-specified bundle: see :data:`DEFAULT_MACRO_SERIES`.
    force_rebuild : bool, keyword-only, default False
        Re-fetch from FRED instead of using the parquet cache.

    Returns
    -------
    pd.DataFrame
        Wide-format frame indexed by ``date`` (business-day frequency)
        with one column per series ID. Forward-filled.

    Notes
    -----
    * Daily S&P 500 index level is NOT in the default bundle. FRED's
      ``SP500`` series only goes back to ~2014, too short for the
      2005-2024 project window. Person C currently sources their
      S&P 500 index data independently; a future PR may add
      `load_market_index` that splices CRSP + yfinance to cover the
      full window.
    * Macro features are released with publication lags (FRED typically
      delivers same-day for daily series, monthly with ~1-month lag for
      monthly series). This loader does not enforce as-of dating; if
      strict look-ahead avoidance matters for your use case, apply
      ``.shift(N)`` downstream.

    Examples
    --------
    >>> # Default bundle, 2005-2024
    >>> df = load_macro(start="2005-01-01", end="2024-12-31")
    >>> df["DGS10"].head()

    >>> # Just one series
    >>> vix = load_macro(series_ids=["VIXCLS"], start="2020-01-01")
    """
    if series_ids is None:
        series_ids = DEFAULT_MACRO_SERIES
    series_ids = tuple(series_ids)  # ensure hashable

    df = _load_macro_raw(series_ids, force_rebuild=force_rebuild)

    if start is not None:
        df = df.loc[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df.loc[df.index <= pd.Timestamp(end)]
    return df


def _sp500_union_in_window(start: str, end: str) -> tuple[str, ...]:
    """Return every ticker that was S&P 500 at any point in ``[start, end]``.

    A membership spell ``[s, e]`` (with ``e = NaT`` meaning "still active")
    overlaps the window iff ``s <= end`` AND (``e`` is open OR ``e >= start``).
    Used by :func:`load_prices_yfinance` to decide which tickers to fetch.

    Returns
    -------
    tuple[str, ...]
        Sorted, de-duplicated. Empty if the window is entirely before the
        first recorded membership.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    table = _load_membership_table()
    overlaps = (
        (table["start_date"] <= end_ts)
        & (table["end_date"].isna() | (table["end_date"] >= start_ts))
    )
    return tuple(sorted(table.loc[overlaps, "ticker"].unique()))


def load_prices_yfinance(
    *,
    start: str,
    end: str,
    universe: tuple[str, ...] | list[str] | None = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Load monthly stock prices from yfinance, restricted to an S&P 500 universe.

    Public-API wrapper around :func:`_load_yfinance_monthly_raw` that picks
    a sensible default universe: every ticker that was an S&P 500 member at
    any point in ``[start, end]`` (call it the "S&P 500 union"). This
    over-fetches relative to a strict point-in-time filter, but is cached
    once and lets downstream code apply :func:`load_sp500_membership` at
    each rebalance date for true PIT filtering.

    Parameters
    ----------
    start : str
        Inclusive start date (ISO).
    end : str
        Inclusive end date (ISO).
    universe : tuple[str, ...] or list[str], optional
        Explicit ticker list. ``None`` (default) computes the S&P 500 union
        over the window.
    force_rebuild : bool, keyword-only, default False
        Bypass the cache and re-download.

    Returns
    -------
    pd.DataFrame
        Long-format frame indexed by ``(date, ticker)`` MultiIndex. Columns
        as in :func:`_load_yfinance_monthly_raw` (open, high, low,
        adj_close, volume, ret).

        Note the identifier is ``ticker`` (str), NOT ``permno`` (int) --
        yfinance has no PERMNO concept. The splice function in a future PR
        will bridge the two ID spaces.

    Notes
    -----
    * yfinance silently drops tickers with no data (delisted before
      ``start``, never existed, name changes Yahoo no longer maps).
      The returned frame's ticker set may therefore be smaller than the
      requested universe; compare ``df.index.get_level_values('ticker').unique()``
      against the requested list to spot drops.
    * For the 2023+ splice use case the default universe is correct: it
      includes every ticker that touched the index, regardless of when it
      joined or left, so PIT filtering downstream is always feasible.
    """
    if universe is None:
        tickers = _sp500_union_in_window(start, end)
        print(
            f"[load_prices_yfinance] S&P 500 union over "
            f"{start} -> {end}: {len(tickers)} tickers"
        )
    else:
        tickers = tuple(universe)

    if not tickers:
        raise ValueError(
            f"Empty universe for window {start} -> {end}. "
            f"Check that download_sp500_universe() has run."
        )

    return _load_yfinance_monthly_raw(
        tickers, start=start, end=end, force_rebuild=force_rebuild
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

    # ---- yfinance monthly prices ----
    print()
    print("=" * 70)
    print("yfinance monthly smoke test (5 large-caps, 2023)")
    print("=" * 70)
    yf_tickers = ("AAPL", "MSFT", "GOOG", "JPM", "XOM")
    # Pull 2022-12 too so first 2023 return is non-NaN.
    yf_df = _load_yfinance_monthly_raw(
        yf_tickers, start="2022-12-01", end="2023-12-31"
    )
    print()
    print(f"Shape: {yf_df.shape[0]:,} rows x {yf_df.shape[1]} cols")
    print(
        f"Date range: {yf_df.index.get_level_values('date').min().date()} -> "
        f"{yf_df.index.get_level_values('date').max().date()}"
    )
    print(
        f"Tickers returned: "
        f"{sorted(yf_df.index.get_level_values('ticker').unique())}"
    )
    print()
    # Sanity: 2023 was a strong year for big tech. AAPL ~+49%, MSFT ~+58%,
    # GOOG ~+58%; JPM ~+27%; XOM only ~-2% (oil pulled back). If these are
    # wildly off, something's wrong with the adj_close / return chain.
    print("2023 cumulative total return (Jan -> Dec, 12 months):")
    for tkr in yf_tickers:
        sub = yf_df.xs(tkr, level="ticker")
        rets_2023 = sub.loc["2023-01-01":"2023-12-31", "ret"].dropna()
        if len(rets_2023) < 12:
            print(f"  {tkr}: only {len(rets_2023)} return observations -- check?")
            continue
        cum = (1 + rets_2023).prod() - 1
        print(f"  {tkr}: {cum:+.1%} (over {len(rets_2023)} months)")

    # Second call must be a cache hit (no network).
    print()
    print("Cache-hit check (second call should print 'cache hit'):")
    _ = _load_yfinance_monthly_raw(
        yf_tickers, start="2022-12-01", end="2023-12-31"
    )

    # ---- yfinance public loader + S&P 500 universe wiring ----
    print()
    print("=" * 70)
    print("load_prices_yfinance smoke test (universe wiring)")
    print("=" * 70)

    # Step 1: universe lookup only (no download). Should be ~500-600 names
    # over a 2-year window once index churn is counted in.
    for win in [("2024-01-01", "2024-12-31"), ("2023-01-01", "2024-12-31")]:
        uni = _sp500_union_in_window(*win)
        print(
            f"  S&P 500 union {win[0]} -> {win[1]}: "
            f"{len(uni)} unique tickers (first 5: {sorted(uni)[:5]})"
        )

    # Step 2: end-to-end via the public function with an EXPLICIT small
    # universe (so we don't burn the network on 500 tickers in a smoke test).
    print()
    print("Public API end-to-end (explicit 3-ticker universe, 2024 Q4):")
    df_q4 = load_prices_yfinance(
        start="2024-09-01", end="2024-12-31",
        universe=("NVDA", "TSLA", "META"),
    )
    print(
        f"  Returned: {df_q4.shape[0]} rows, "
        f"{df_q4.index.get_level_values('ticker').nunique()} tickers, "
        f"{df_q4.index.get_level_values('date').nunique()} months"
    )
    print(f"  Index names: {df_q4.index.names}")
    print(f"  Columns: {list(df_q4.columns)}")

    # ---- FRED macro features ----
    print()
    print("=" * 70)
    print("FRED macro smoke test")
    print("=" * 70)
    macro = load_macro(start="2008-01-01", end="2008-12-31")
    print()
    print(f"Shape: {macro.shape[0]:,} rows x {macro.shape[1]} cols")
    print(f"Date range: {macro.index.min().date()} -> {macro.index.max().date()}")
    print(f"Series: {list(macro.columns)}")
    print()
    print("Famous-event spot check (Lehman week, Sept 15 2008):")
    week = macro.loc["2008-09-12":"2008-09-19"]
    print(week.to_string())
    print()
    # Sanity: VIX should spike that week; term spread (DGS10 - DGS2) widens too.
    spread = macro["DGS10"] - macro["DGS2"]
    credit = macro["DBAA"] - macro["DAAA"]
    print(f"Derived features quick-check on 2008-09-15:")
    print(f"  VIX:                       {macro.loc['2008-09-15', 'VIXCLS']:.2f}  (spiked from ~25 to 30+)")
    print(f"  Term spread (DGS10-DGS2):  {spread.loc['2008-09-15']:.2f}  (recession-watch indicator)")
    print(f"  Credit spread (DBAA-DAAA): {credit.loc['2008-09-15']:.2f}  (default-risk gauge)")
