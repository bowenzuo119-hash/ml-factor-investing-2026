"""metrics.py - Performance metrics for backtest output evaluation.

All functions are pure: they take a pandas Series of periodic returns
(typically monthly, indexed by date) and return a scalar or another
Series. None of them mutate inputs or touch the filesystem.

The conventions used throughout this module:

    * "Returns" means simple (arithmetic) returns expressed as decimals,
      e.g., 0.012 for +1.2%. NOT log returns.
    * Annualisation factor defaults to 12 (monthly returns). Pass
      `periods_per_year=252` for daily.
    * Risk-free rate defaults to 0 (excess returns assumed). For a
      strict Sharpe, pass the per-period risk-free rate (e.g. monthly
      T-bill).
    * Information ratio's benchmark must be a Series aligned by date
      with the strategy returns; mis-aligned dates are dropped.

Used by:
    * `src.backtest` smoke tests for Sharpe and friends
    * Report-time tear sheet code (TODO, downstream)
    * Person B's model selection / Diebold-Mariano comparisons
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def annualised_return(returns: pd.Series, *, periods_per_year: int = 12) -> float:
    """Geometric-mean annualised return.

    ``(1 + r_t).prod() ** (PPY / n) - 1`` — the rate that, compounded
    PPY times per year, reproduces the observed cumulative growth.

    Returns 0.0 on an empty input.
    """
    r = returns.dropna()
    if len(r) == 0:
        return 0.0
    cum = (1.0 + r).prod()
    return float(cum ** (periods_per_year / len(r)) - 1.0)


def annualised_volatility(returns: pd.Series, *, periods_per_year: int = 12) -> float:
    """Standard deviation of returns scaled by ``sqrt(periods_per_year)``.

    Uses sample std (ddof=1). Returns 0.0 if fewer than 2 observations.
    """
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    *,
    periods_per_year: int = 12,
    risk_free: float = 0.0,
) -> float:
    """Annualised Sharpe ratio.

    Defined as ``(annualised_excess_return) / (annualised_volatility)``.
    With the default ``risk_free=0`` the returns are treated as already
    in excess form (which is the typical convention for L/S strategies
    that are self-funding).

    Returns 0.0 if volatility is zero or input is empty.
    """
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    excess = r - risk_free
    vol = annualised_volatility(excess, periods_per_year=periods_per_year)
    if vol == 0.0:
        return 0.0
    return float(annualised_return(excess, periods_per_year=periods_per_year) / vol)


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough loss in cumulative-wealth space.

    Returns a NEGATIVE number (or 0.0 if the series never goes below
    a prior peak). The most common reporting convention is positive
    magnitude — callers should ``abs()`` if they prefer that.

    Computed as ``min(wealth_t / running_peak_t - 1)``.
    """
    r = returns.dropna()
    if len(r) == 0:
        return 0.0
    wealth = (1.0 + r).cumprod()
    running_peak = wealth.cummax()
    drawdown = wealth / running_peak - 1.0
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, *, periods_per_year: int = 12) -> float:
    """Annualised return divided by the magnitude of max drawdown.

    Higher is better. Returns 0.0 if there is no drawdown (or all
    returns are positive, which makes max drawdown == 0).
    """
    dd = max_drawdown(returns)
    if dd == 0.0:
        return 0.0
    return float(annualised_return(returns, periods_per_year=periods_per_year) / abs(dd))


def hit_rate(returns: pd.Series) -> float:
    """Fraction of strictly positive returns in (0, 1].

    NaN observations are dropped before counting.
    """
    r = returns.dropna()
    if len(r) == 0:
        return 0.0
    return float((r > 0).mean())


def information_ratio(
    returns: pd.Series,
    benchmark: pd.Series,
    *,
    periods_per_year: int = 12,
) -> float:
    """Annualised information ratio against a benchmark return series.

    IR = (annualised mean of active returns) / (annualised tracking error),
    where ``active_t = strategy_t - benchmark_t``. Returns 0.0 if
    tracking error is zero or the joint sample is empty.

    The two series are aligned by index; dates appearing in only one
    are dropped.
    """
    joined = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    if len(joined) < 2:
        return 0.0
    active = joined.iloc[:, 0] - joined.iloc[:, 1]
    te = annualised_volatility(active, periods_per_year=periods_per_year)
    if te == 0.0:
        return 0.0
    return float(annualised_return(active, periods_per_year=periods_per_year) / te)


def summary_stats(
    returns: pd.Series,
    *,
    benchmark: pd.Series | None = None,
    periods_per_year: int = 12,
    risk_free: float = 0.0,
) -> dict[str, float]:
    """Convenience: compute all single-series metrics in one call.

    Returns a dict keyed by metric name, suitable for printing or for
    appending to BacktestResult.metadata. If ``benchmark`` is provided,
    the dict also includes ``information_ratio``.
    """
    out: dict[str, float] = {
        "annualised_return": annualised_return(returns, periods_per_year=periods_per_year),
        "annualised_volatility": annualised_volatility(returns, periods_per_year=periods_per_year),
        "sharpe_ratio": sharpe_ratio(
            returns, periods_per_year=periods_per_year, risk_free=risk_free
        ),
        "max_drawdown": max_drawdown(returns),
        "calmar_ratio": calmar_ratio(returns, periods_per_year=periods_per_year),
        "hit_rate": hit_rate(returns),
        "n_periods": int(returns.dropna().shape[0]),
    }
    if benchmark is not None:
        out["information_ratio"] = information_ratio(
            returns, benchmark, periods_per_year=periods_per_year
        )
    return out
