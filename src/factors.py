"""factors.py - Cross-sectional feature construction for the alpha model.

Person B owns this file. It turns the raw price / market-cap panel produced
by `data_loader` into a long-format feature matrix that
`run_walk_forward_backtest` consumes:

    features.index = MultiIndex(date, ticker)
    features.columns = ["mom", "rev", "log_mktcap", "mvol", "ivol", "sector", ...]

The backtest engine reads the date level off the index to align with
returns; the asset level identifies the cross-section. See
`src.backtest.run_walk_forward_backtest`.

Feature stack (Project Framework §3.2)
--------------------------------------
Layer 1: sector-relative ranks (`sector_relative_rank`). This module's
    primary contribution to sector neutrality. Replaces every raw feature
    with a within-(date, sector) percentile rank in [0, 1].
Layer 2: sector-relative target. Done outside this file (model-side wrapper
    or as a pre-fit transform). Deferred until the basic pipeline runs.
Layer 3: sector-neutral portfolio construction. Owned by Person A in
    `backtest.py` (k_per_sector lever, currently a warn-only stub).

Data-availability reality (2026-05-21)
--------------------------------------
Person A's pipeline gives MONTHLY observations (CRSP MSF + yfinance
resampled to month-end), so the daily formulae in
`documents/Person_A_Feature_Spec.docx` for 21-day volatility, dollar
volume, and 21-day idiosyncratic volatility cannot be implemented as
written. Feasible substitutes are documented per-feature below. B/M and
E/P are blocked entirely pending Compustat (DECISIONS.md 2026-05-13).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import (
    DATA_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    SP500_CURRENT_FILE,
    download_sp500_universe,
    load_prices,
    load_prices_spliced,
)


# --------------------------------------------------------------------------
# Sector mapping (Layer 1 prerequisite)
# --------------------------------------------------------------------------

# 2-digit SIC -> coarse GICS-flavoured sector. Used as a fallback for
# tickers not present in the current `sp500.csv` (delisted / pre-merger
# names that no longer appear in the live index). Not as granular as real
# GICS but adequate for an 11-bucket sector-neutral construction at the
# course-project level. Refine if a downstream check shows a bucket with
# very few stocks per rebalance.
_SIC2_TO_SECTOR: dict[int, str] = {
    # 10-14: Mining & Energy
    10: "Energy", 12: "Energy", 13: "Energy", 14: "Materials",
    # 15-17: Construction -> Industrials
    15: "Industrials", 16: "Industrials", 17: "Industrials",
    # 20-39: Manufacturing -> mostly Industrials / Consumer / Tech / Health
    20: "Consumer Staples", 21: "Consumer Staples",
    22: "Consumer Discretionary", 23: "Consumer Discretionary",
    24: "Industrials", 25: "Consumer Discretionary",
    26: "Materials", 27: "Communication Services",
    28: "Materials",   # chemicals -- ambiguous; many pharma sit here too
    29: "Energy",
    30: "Consumer Discretionary", 31: "Consumer Discretionary",
    32: "Materials", 33: "Materials", 34: "Industrials",
    35: "Industrials", 36: "Information Technology",
    37: "Consumer Discretionary", 38: "Health Care",
    39: "Consumer Discretionary",
    # 40-49: Transport / Utilities / Communications
    40: "Industrials", 41: "Industrials", 42: "Industrials",
    43: "Industrials", 44: "Industrials", 45: "Industrials",
    46: "Energy", 47: "Industrials",
    48: "Communication Services", 49: "Utilities",
    # 50-59: Wholesale / Retail
    50: "Consumer Discretionary", 51: "Consumer Staples",
    52: "Consumer Discretionary", 53: "Consumer Discretionary",
    54: "Consumer Staples", 55: "Consumer Discretionary",
    56: "Consumer Discretionary", 57: "Consumer Discretionary",
    58: "Consumer Discretionary", 59: "Consumer Discretionary",
    # 60-67: Finance
    60: "Financials", 61: "Financials", 62: "Financials",
    63: "Financials", 64: "Financials", 65: "Real Estate",
    67: "Financials",
    # 70-89: Services -> mixed
    70: "Consumer Discretionary", 72: "Consumer Discretionary",
    73: "Information Technology", 75: "Consumer Discretionary",
    78: "Communication Services", 79: "Communication Services",
    80: "Health Care", 82: "Consumer Discretionary",
    87: "Information Technology",
}


def load_sector_map() -> dict[str, str]:
    """Build a ticker -> sector lookup.

    First source is `data/raw/sp500.csv` from fja05680/sp500, which carries
    GICS sector for CURRENT index members. That covers ~500 names; for
    delisted-but-historically-S&P-500 tickers we fall back at lookup time
    to the CRSP `sic_code` -> coarse-sector map.

    Returns
    -------
    dict[str, str]
        Mapping from uppercase ticker to GICS sector string. Use
        :func:`get_sector` (below) to query with the SIC fallback rolled in.

    Notes
    -----
    Survivorship caveat: the GICS labels in `sp500.csv` are the *current*
    classification, not the as-of-date classification. For most stocks the
    GICS sector is stable over years; for those that genuinely changed
    sector (re-organisations, M&A), this map is approximate. The framework
    accepts this as a course-project-grade trade-off.
    """
    download_sp500_universe()  # no-op if cached
    path = RAW_DIR / SP500_CURRENT_FILE
    df = pd.read_csv(path)

    # fja05680/sp500 column names: confirm by reading the file once
    # interactively before relying on this. The known headers are
    # "Symbol" and "GICS Sector" but the repo has shifted them before.
    ticker_col = next(
        (c for c in df.columns if c.lower() in ("symbol", "ticker")),
        None,
    )
    sector_col = next(
        (c for c in df.columns if "sector" in c.lower()),
        None,
    )
    if ticker_col is None or sector_col is None:
        raise RuntimeError(
            f"load_sector_map: could not find ticker/sector columns in "
            f"{path.name}. Saw: {list(df.columns)}"
        )

    return {
        str(t).strip().upper(): str(s).strip()
        for t, s in zip(df[ticker_col], df[sector_col])
        if pd.notna(t) and pd.notna(s)
    }


def get_sector(ticker: str, sic_code: str | int | None,
               current_map: dict[str, str]) -> str:
    """Resolve a single (ticker, sic_code) to a sector label.

    Lookup order: (1) current_map[ticker] if present; (2) SIC 2-digit
    fallback; (3) "Unknown" so the row doesn't silently disappear during
    sector-relative ranking (a sector with a single stock collapses the
    rank to 0.5 but the row still survives the filter).
    """
    if ticker:
        s = current_map.get(str(ticker).strip().upper())
        if s:
            return s
    if sic_code is not None and not (isinstance(sic_code, float) and np.isnan(sic_code)):
        try:
            sic_int = int(float(sic_code))
            return _SIC2_TO_SECTOR.get(sic_int // 100, "Unknown")
        except (ValueError, TypeError):
            pass
    return "Unknown"


# --------------------------------------------------------------------------
# Individual feature computations (consume wide-format monthly returns)
# --------------------------------------------------------------------------

def momentum(monthly_returns_wide: pd.DataFrame,
             lookback: int = 11, skip: int = 1) -> pd.DataFrame:
    """Compute 12-1 cross-sectional momentum from monthly returns.

    Standard Jegadeesh-Titman / GKX construction: cumulative return over
    the past ``lookback`` months ending ``skip`` months before t (i.e.,
    [t-12, t-2] for the default 11/1). The skip-month removes the
    short-term reversal effect, which we capture separately via
    :func:`reversal`.

    Parameters
    ----------
    monthly_returns_wide : pd.DataFrame
        Rows = month-end dates, columns = ticker, values = monthly total
        returns. Same shape as `data/processed/returns_spliced_*.parquet`.
    lookback : int, default 11
        Number of months over which to accumulate, ending at ``t - skip``.
    skip : int, default 1
        Number of most-recent months to exclude.

    Returns
    -------
    pd.DataFrame
        Same shape as the input. NaN for the first ``lookback + skip``
        months and wherever the underlying return was NaN.
    """
    # Shift by `skip` so a row dated t holds returns ending at t-skip.
    shifted = monthly_returns_wide.shift(skip)
    # rolling product of (1 + r) - 1. min_periods enforces sufficient history.
    return (
        (1.0 + shifted)
        .rolling(lookback, min_periods=lookback)
        .apply(np.prod, raw=True)
        .subtract(1.0)
    )


def reversal(monthly_returns_wide: pd.DataFrame) -> pd.DataFrame:
    """Prior 1-month return (Jegadeesh 1990 short-term reversal)."""
    return monthly_returns_wide.copy()


def monthly_volatility(monthly_returns_wide: pd.DataFrame,
                       window: int = 6) -> pd.DataFrame:
    """Trailing rolling standard deviation of monthly returns.

    Substitutes for Person A's 21-day daily volatility (feature #5),
    which the current monthly pipeline cannot produce. ``window`` of 6
    months trades a longer signal half-life for a more stable estimate;
    revisit if cross-sectional rank stability looks poor at backtest time.
    """
    return monthly_returns_wide.rolling(window, min_periods=window).std()


def idiosyncratic_volatility(
    monthly_returns_wide: pd.DataFrame,
    market_returns: pd.Series,
    window: int = 24,
) -> pd.DataFrame:
    """Rolling residual volatility from regressing stock on market.

    Per-stock: at each month t, regress r_i,[t-w+1..t] on r_m,[t-w+1..t]
    and one constant. Output is the std of the residuals. Window of 24
    months is a compromise between estimation noise and recency; daily
    21-day equivalents (Ang et al. 2006) are not achievable on monthly data.

    Parameters
    ----------
    monthly_returns_wide : pd.DataFrame
        Rows = month-end dates, columns = ticker.
    market_returns : pd.Series
        Single time series of market returns aligned to the same dates
        (e.g., equal-weighted average of the universe each month, or
        S&P 500 total return if available).
    window : int, default 24

    Returns
    -------
    pd.DataFrame
        Same shape as input. NaN for the first ``window`` months.
    """
    mkt = market_returns.reindex(monthly_returns_wide.index)
    # Demean market and stocks within each rolling window manually for
    # speed; closed-form residual std avoids a Python loop over stocks.
    out = pd.DataFrame(index=monthly_returns_wide.index,
                       columns=monthly_returns_wide.columns,
                       dtype=float)
    for ticker in monthly_returns_wide.columns:
        y = monthly_returns_wide[ticker]
        joint = pd.concat([y.rename("y"), mkt.rename("m")], axis=1)

        def _resid_std(block: pd.DataFrame) -> float:
            block = block.dropna()
            if len(block) < window // 2:
                return np.nan
            y_b = block["y"].to_numpy()
            m_b = block["m"].to_numpy()
            # OLS with intercept: residual std of y on [1, m]
            m_demean = m_b - m_b.mean()
            y_demean = y_b - y_b.mean()
            denom = (m_demean ** 2).sum()
            if denom <= 0:
                return float(np.std(y_demean, ddof=1)) if len(y_demean) > 1 else np.nan
            beta = (m_demean * y_demean).sum() / denom
            resid = y_demean - beta * m_demean
            return float(np.std(resid, ddof=1)) if len(resid) > 1 else np.nan

        out[ticker] = (
            joint.rolling(window, min_periods=window // 2)
            .apply(lambda _: np.nan, raw=False)  # placeholder triggers index
        )
        # Re-do via a fast per-window loop (rolling.apply doesn't support
        # multi-column custom funcs without engine='cython').
        vals = np.full(len(joint), np.nan)
        arr_y = joint["y"].to_numpy()
        arr_m = joint["m"].to_numpy()
        for i in range(window - 1, len(joint)):
            yw = arr_y[i - window + 1: i + 1]
            mw = arr_m[i - window + 1: i + 1]
            mask = ~(np.isnan(yw) | np.isnan(mw))
            if mask.sum() < window // 2:
                continue
            yw, mw = yw[mask], mw[mask]
            m_d = mw - mw.mean()
            y_d = yw - yw.mean()
            denom = (m_d ** 2).sum()
            if denom <= 0:
                vals[i] = float(np.std(y_d, ddof=1)) if len(y_d) > 1 else np.nan
                continue
            beta = (m_d * y_d).sum() / denom
            resid = y_d - beta * m_d
            vals[i] = float(np.std(resid, ddof=1)) if len(resid) > 1 else np.nan
        out[ticker] = vals
    return out


def log_market_cap_from_crsp(start: str, end: str) -> pd.DataFrame:
    """Pull month-end log market cap from CRSP for [start, end].

    Returns a wide DataFrame (rows = date, cols = ticker, values = log mcap).
    Tickers absent from CRSP (e.g. yfinance-era 2023+) are absent here too;
    caller must reindex / merge.
    """
    crsp = load_prices(start=start, end=end)
    # CRSP loader gives (date, permno) MultiIndex with `ticker` as a column.
    # Map per-PERMNO to the latest in-window ticker so it splices cleanly
    # with the yfinance side (same convention as `load_prices_spliced`).
    permno_to_ticker = (
        crsp.reset_index()
        .sort_values("date")
        .groupby("permno")["ticker"]
        .last()
    )
    mcap_long = (
        crsp[["market_cap"]]
        .reset_index()
        .assign(ticker=lambda d: d["permno"].map(permno_to_ticker))
        .dropna(subset=["ticker", "market_cap"])
    )
    # If a (date, ticker) cell has multiple PERMNOs (rare ticker reuse),
    # take the largest market cap (most likely the surviving primary listing).
    mcap_long = (
        mcap_long.groupby(["date", "ticker"])["market_cap"].max().reset_index()
    )
    wide = (
        mcap_long.pivot(index="date", columns="ticker", values="market_cap")
        .sort_index()
    )
    return np.log(wide.where(wide > 0))


# --------------------------------------------------------------------------
# Sector-relative ranks (Layer 1 of the three-layer stack)
# --------------------------------------------------------------------------

def sector_relative_rank(
    long_panel: pd.DataFrame,
    feature_cols: list[str],
    sector_col: str = "sector",
) -> pd.DataFrame:
    """Replace raw features with within-(date, sector) percentile ranks.

    Mutates a copy of ``long_panel``. For each column ``c`` in
    ``feature_cols`` and each (date, sector) group, replace the raw value
    with its percentile rank in [0, 1] computed across stocks in the same
    sector at the same date.

    Parameters
    ----------
    long_panel : pd.DataFrame
        Long-format frame indexed by (date, ticker). Must contain
        ``sector_col`` and every name in ``feature_cols``.
    feature_cols : list[str]
        Columns to rank in place.
    sector_col : str, default "sector"

    Returns
    -------
    pd.DataFrame
        Copy of ``long_panel`` with the listed feature columns replaced
        by their sector-relative ranks. NaN inputs stay NaN.
    """
    out = long_panel.copy()
    date_level = out.index.names[0]
    for col in feature_cols:
        out[col] = (
            out.groupby([date_level, sector_col], observed=True)[col]
            .rank(pct=True)
        )
    return out


# --------------------------------------------------------------------------
# Top-level orchestrator -- this is what models / the backtest consume
# --------------------------------------------------------------------------

FROZEN_RETURNS_FILE = PROCESSED_DIR / "returns_spliced_2019_2024.parquet"


def build_feature_panel(
    *,
    start: str = "2005-01-01",
    end: str = "2024-12-31",
    include: tuple[str, ...] = ("mom", "rev", "log_mktcap", "mvol", "ivol"),
    sector_rank: bool = True,
) -> pd.DataFrame:
    """Build the long-format feature panel the backtest engine expects.

    Pulls returns via `load_prices_spliced(start, end)` (CRSP <= 2022-12
    + yfinance >= 2023-01), enriches with CRSP-era market-cap and SIC for
    sector mapping, computes the requested features, and (optionally)
    applies the Layer-1 sector-relative-rank transform.

    Parameters
    ----------
    start, end : str
        ISO date window. The default spans the Framework's full sample
        (training 2005-2015 + validation 2016-2018 + test 2019-2024).
    include : tuple[str, ...]
        Which features to compute. Valid keys: ``"mom"``, ``"rev"``,
        ``"log_mktcap"``, ``"mvol"``, ``"ivol"``.
    sector_rank : bool, default True
        If True, replace each feature column with its sector-relative
        percentile rank. If False, return raw values (useful for
        diagnostics).

    Returns
    -------
    pd.DataFrame
        Long-format with MultiIndex(date, ticker). Columns:
        the included feature columns plus ``sector``.

    Notes
    -----
    Market cap is only available for the CRSP era (<= 2022-12). For the
    yfinance tail, ``log_mktcap`` will be NaN; rows where every requested
    feature is NaN are dropped to keep the panel tight, but rows with at
    least one usable feature survive (the model can handle NaNs columnwise
    via imputation or column-subset training).
    """
    # 1. Returns (wide format) -----------------------------------------
    spliced = load_prices_spliced(start=start, end=end)
    returns_wide = (
        spliced["ret"].unstack(level="ticker").sort_index()
    )

    # 2. Sector map ----------------------------------------------------
    current_sector_map = load_sector_map()

    # SIC code per (date, ticker) from CRSP; map to GICS-flavoured sector.
    crsp = load_prices(start=start, end=min(end, "2022-12-30"))
    permno_to_ticker = (
        crsp.reset_index().sort_values("date")
        .groupby("permno")["ticker"].last()
    )
    sic_long = (
        crsp[["sic_code"]].reset_index()
        .assign(ticker=lambda d: d["permno"].map(permno_to_ticker))
        .dropna(subset=["ticker"])
        .groupby(["date", "ticker"])["sic_code"].first()
    )

    # Build a per-ticker fallback sector from each ticker's most recent
    # CRSP SIC. Then resolve sector at each (date, ticker) using
    # current_sector_map first, SIC fallback second.
    latest_sic = sic_long.reset_index().sort_values("date").groupby("ticker")["sic_code"].last()
    ticker_to_sector: dict[str, str] = {}
    for ticker in returns_wide.columns:
        sic = latest_sic.get(ticker, None)
        ticker_to_sector[ticker] = get_sector(ticker, sic, current_sector_map)

    # 3. Per-feature computation ---------------------------------------
    feature_wide: dict[str, pd.DataFrame] = {}
    if "mom" in include:
        feature_wide["mom"] = momentum(returns_wide, lookback=11, skip=1)
    if "rev" in include:
        feature_wide["rev"] = reversal(returns_wide)
    if "mvol" in include:
        feature_wide["mvol"] = monthly_volatility(returns_wide, window=6)
    if "ivol" in include:
        # Crude market proxy: equal-weighted cross-sectional mean.
        # Swap for the actual S&P 500 monthly TR once `load_market_index`
        # ships (see DECISIONS.md 2026-05-13 "Macro data" entry).
        market = returns_wide.mean(axis=1, skipna=True)
        feature_wide["ivol"] = idiosyncratic_volatility(returns_wide, market, window=24)
    if "log_mktcap" in include:
        mcap = log_market_cap_from_crsp(start=start, end=min(end, "2022-12-30"))
        feature_wide["log_mktcap"] = mcap.reindex(
            index=returns_wide.index, columns=returns_wide.columns
        )

    # 4. Stack to long format -----------------------------------------
    long_frames = []
    for name, wide in feature_wide.items():
        long = (
            wide.stack(future_stack=True)
            .rename(name)
            .to_frame()
        )
        long.index = long.index.set_names(["date", "ticker"])
        long_frames.append(long)
    if not long_frames:
        raise ValueError("build_feature_panel: nothing in `include` produced data")
    panel = pd.concat(long_frames, axis=1).sort_index()

    # Attach sector
    panel["sector"] = (
        panel.index.get_level_values("ticker").map(ticker_to_sector).fillna("Unknown")
    )

    # Drop rows where ALL feature columns are NaN
    feature_cols = [c for c in panel.columns if c != "sector"]
    panel = panel.dropna(subset=feature_cols, how="all")

    # 5. Sector-relative ranks (Layer 1) ------------------------------
    if sector_rank:
        panel = sector_relative_rank(panel, feature_cols=feature_cols)

    return panel


# --------------------------------------------------------------------------
# Blocked features -- documented stubs so future-Person-B knows the gap
# --------------------------------------------------------------------------

def daily_dollar_volume(*args, **kwargs):  # noqa: ARG001
    """BLOCKED. Person A's pipeline produces monthly data only.

    To implement feature #4 from the spec we would need either:
      (a) Person A to add a daily-frequency loader (CRSP DSF or yfinance
          daily), or
      (b) A monthly proxy like price * shrout (= mktcap) which is just
          feature #3 again, so not informative as a separate signal.
    """
    raise NotImplementedError(
        "Daily dollar volume requires daily data, which the current pipeline "
        "does not expose. See factors.py module docstring."
    )


def daily_volatility_21d(*args, **kwargs):  # noqa: ARG001
    """BLOCKED. Use `monthly_volatility` as the feasible substitute."""
    raise NotImplementedError(
        "21-day daily volatility requires daily returns. "
        "Use monthly_volatility() instead until a daily loader exists."
    )


def book_to_market(*args, **kwargs):  # noqa: ARG001
    """BLOCKED pending Compustat access.

    See DECISIONS.md 2026-05-13 "Defer fundamentals". CRSP MSF has no
    book value, no net income, no shares outstanding history that would
    let us build B/M or E/P. If the TA replies with Compustat access,
    implement here.
    """
    raise NotImplementedError("B/M needs Compustat fundamentals (not available)")


def earnings_to_price(*args, **kwargs):  # noqa: ARG001
    """BLOCKED pending Compustat access. See `book_to_market`."""
    raise NotImplementedError("E/P needs Compustat fundamentals (not available)")


# --------------------------------------------------------------------------
# Smoke test (python -m src.factors)
# --------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("factors.py smoke test")
    print("=" * 70)

    panel = build_feature_panel(
        start="2019-01-01", end="2024-12-31",
        include=("mom", "rev", "mvol", "log_mktcap"),
        sector_rank=True,
    )

    print(f"\nPanel shape: {panel.shape}")
    print(f"Date range: {panel.index.get_level_values('date').min().date()} "
          f"-> {panel.index.get_level_values('date').max().date()}")
    print(f"Unique tickers: {panel.index.get_level_values('ticker').nunique()}")
    print(f"\nColumns + dtypes:\n{panel.dtypes}")
    print(f"\nSector counts (first rebalance):")
    first_date = panel.index.get_level_values("date").min()
    print(panel.xs(first_date, level="date")["sector"].value_counts())
    print(f"\nHead:\n{panel.head(10)}")
    print(f"\nNaN fraction per column:\n{panel.isna().mean()}")
