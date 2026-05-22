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


# --------------------------------------------------------------------------
# Prediction-quality metrics (Person B's Level-1 evaluation)
# --------------------------------------------------------------------------
#
# These operate on flat (y_true, y_pred) Series indexed by (date, ticker),
# NOT on a single returns series. They complement the portfolio-level
# metrics above by measuring how well the *model* forecasts individual
# stock returns, before any portfolio is built.
#
# Conventions:
#   * y_true is the realised next-period return at (date, ticker).
#   * y_pred is the model's forecast for the same (date, ticker).
#   * Both must share a MultiIndex with names ("date", <asset_level>).
#   * Joint sample = intersection of the two indices, with NaNs dropped.

def oos_r2(
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    benchmark: str = "zero",
) -> float:
    """Out-of-sample R-squared in the Gu-Kelly-Xiu (2020) sense.

    The numerator is always the sum of squared forecast errors. The
    denominator depends on ``benchmark``:

    * ``"zero"`` (default, GKX equation 4)::

          R2 = 1 - SUM (y - y_pred)^2 / SUM y^2

      Asks "did we beat a constant forecast of zero?". This is the
      stringent benchmark GKX favours -- most of the cross-sectional
      variation in monthly returns IS noise, so demeaning the
      denominator (the classic R^2 definition) overstates a model's
      apparent skill.

    * ``"mean"``::

          R2 = 1 - SUM (y - y_pred)^2 / SUM (y - mean_t(y))^2

      Asks "did we beat predicting the cross-sectional mean at each
      date?". This is the "S&P 500 mean" benchmark from Framework
      section 8.2.

    Parameters
    ----------
    y_true, y_pred : pd.Series
        Indexed by (date, ticker) MultiIndex. Mis-aligned rows are
        dropped via inner join; NaN observations are dropped after
        alignment.
    benchmark : {"zero", "mean"}, default "zero"

    Returns
    -------
    float
        R^2 in the same units as y. Can be negative (worse than the
        benchmark). NaN if the joint sample is empty or the
        denominator is zero.
    """
    if benchmark not in ("zero", "mean"):
        raise ValueError(f"benchmark must be 'zero' or 'mean', got {benchmark!r}")

    joint = pd.concat([y_true.rename("y"), y_pred.rename("p")],
                      axis=1, join="inner").dropna()
    if joint.empty:
        return float("nan")

    err = (joint["y"] - joint["p"]).to_numpy()
    ss_err = float(np.dot(err, err))

    if benchmark == "zero":
        y = joint["y"].to_numpy()
        ss_tot = float(np.dot(y, y))
    else:
        # Per-date cross-sectional mean of y, broadcast back to (date, ticker)
        date_level = joint.index.names[0]
        date_mean = joint.groupby(level=date_level)["y"].transform("mean")
        centred = (joint["y"] - date_mean).to_numpy()
        ss_tot = float(np.dot(centred, centred))

    if ss_tot <= 0.0:
        return float("nan")
    return float(1.0 - ss_err / ss_tot)


def information_coefficient(
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    method: str = "spearman",
) -> dict[str, float]:
    """Cross-sectional rank correlation, averaged across dates.

    For each rebalance date, computes the Spearman (default) or Pearson
    correlation between predicted and realised returns across the
    cross-section. Returns the time-series mean, std, and IC information
    ratio ``mean / std``. The classical "IC" reported in alpha-research
    notes is the mean.

    A positive IC means the model ranks stocks correctly on average. A
    Spearman IC of 0.05 across decades is considered strong; 0.02-0.03
    is typical for production cross-sectional equity models.

    Parameters
    ----------
    y_true, y_pred : pd.Series
        Indexed by (date, ticker). Same alignment rules as ``oos_r2``.
    method : {"spearman", "pearson"}, default "spearman"

    Returns
    -------
    dict[str, float]
        Keys: ``ic_mean``, ``ic_std``, ``ic_ir`` (mean/std), ``n_dates``
        (number of rebalance dates that contributed a non-NaN
        correlation). A date is skipped if its cross-section has < 2
        non-NaN pairs.
    """
    if method not in ("spearman", "pearson"):
        raise ValueError(f"method must be 'spearman' or 'pearson', got {method!r}")

    joint = pd.concat([y_true.rename("y"), y_pred.rename("p")],
                      axis=1, join="inner").dropna()
    if joint.empty:
        return {"ic_mean": float("nan"), "ic_std": float("nan"),
                "ic_ir": float("nan"), "n_dates": 0}

    date_level = joint.index.names[0]
    per_date_ic: list[float] = []
    for _, grp in joint.groupby(level=date_level, sort=True):
        if len(grp) < 2:
            continue
        c = grp["y"].corr(grp["p"], method=method)
        if pd.notna(c):
            per_date_ic.append(float(c))

    if not per_date_ic:
        return {"ic_mean": float("nan"), "ic_std": float("nan"),
                "ic_ir": float("nan"), "n_dates": 0}

    arr = np.asarray(per_date_ic, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    ir = float(mean / std) if std > 0 else float("nan")
    return {"ic_mean": mean, "ic_std": std, "ic_ir": ir, "n_dates": len(arr)}


def diebold_mariano(
    pred_a: pd.Series,
    pred_b: pd.Series,
    y_true: pd.Series,
    *,
    newey_west_lags: int = 12,
) -> dict[str, float]:
    """Adapted Diebold-Mariano test comparing two model forecasts.

    Implements the cross-sectional variant from Gu-Kelly-Xiu (2020),
    section 2.6: at each rebalance date t, average the per-stock squared
    forecast errors across the cross-section to get a single scalar per
    period; the loss differential ``d_t = mse_a,t - mse_b,t`` is then a
    plain univariate time series whose mean is tested for being zero.
    This avoids the serial-and-cross-section dependence issue you get if
    you naively run DM on the flat (stock, date) panel.

    The variance of ``d_bar`` is estimated with a Newey-West HAC
    correction (Bartlett kernel) to allow for monthly serial correlation
    in the loss differential. Two-sided p-value from a standard normal.

    Parameters
    ----------
    pred_a, pred_b : pd.Series
        Forecasts from the two models, both indexed by (date, ticker).
    y_true : pd.Series
        Realised next-period returns, same index.
    newey_west_lags : int, default 12
        Maximum lag for the HAC variance. 12 is the conventional choice
        for monthly data with possible annual seasonality.

    Returns
    -------
    dict
        Keys:
        - ``dm_stat`` : standardised loss differential. Negative means
          model A has a SMALLER mean squared error (model A is better).
        - ``p_value`` : two-sided p-value from the standard normal.
        - ``mean_diff`` : raw ``d_bar``. Negative means A wins on MSE.
        - ``n_dates`` : number of rebalance dates that contributed.
    """
    joint = pd.concat(
        [pred_a.rename("a"), pred_b.rename("b"), y_true.rename("y")],
        axis=1, join="inner",
    ).dropna()
    if joint.empty:
        return {"dm_stat": float("nan"), "p_value": float("nan"),
                "mean_diff": float("nan"), "n_dates": 0}

    date_level = joint.index.names[0]
    err_a_sq = (joint["y"] - joint["a"]) ** 2
    err_b_sq = (joint["y"] - joint["b"]) ** 2
    d_t = (err_a_sq - err_b_sq).groupby(level=date_level).mean().sort_index()

    T = len(d_t)
    if T < 2:
        return {"dm_stat": float("nan"), "p_value": float("nan"),
                "mean_diff": float(d_t.mean()) if T else float("nan"),
                "n_dates": int(T)}

    d_bar = float(d_t.mean())
    d_centered = (d_t - d_bar).to_numpy()

    # Newey-West HAC: gamma_0 + 2 * sum_{k=1..L} (1 - k/(L+1)) * gamma_k
    gamma_0 = float((d_centered * d_centered).mean())
    hac_var = gamma_0
    L = min(newey_west_lags, T - 1)
    for k in range(1, L + 1):
        gamma_k = float(
            (d_centered[k:] * d_centered[:-k]).mean()
        )
        weight = 1.0 - k / (L + 1)
        hac_var += 2.0 * weight * gamma_k
    hac_var = max(hac_var, 1e-300)  # guard against negative/zero HAC

    se = float(np.sqrt(hac_var / T))
    dm_stat = d_bar / se if se > 0 else 0.0

    # Two-sided p-value from standard normal. scipy.stats kept out of
    # the top-level import so metrics.py stays cheap.
    from scipy.stats import norm
    p_value = float(2.0 * (1.0 - norm.cdf(abs(dm_stat))))

    return {
        "dm_stat": float(dm_stat),
        "p_value": p_value,
        "mean_diff": d_bar,
        "n_dates": int(T),
    }


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
