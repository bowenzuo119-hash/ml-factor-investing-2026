"""backtest.py - Walk-forward backtesting engine for the L/S equity strategy.

This module is the **integration point** between three workstreams:

    Person A (this file): provides `run_walk_forward_backtest`.
    Person B           : provides a `fit` / `predict` model object that
                          produces cross-sectional return forecasts.
    Person C           : provides a `leverage_fn` that maps a date to a
                          gross-leverage multiplier in [0, 1] based on the
                          regime detected on that date.

If the interface in this file changes, B and C need to know - bump the
`INTERFACE_VERSION` constant at the bottom and shout in standup.

Methodology mirrors Gu, Kelly & Xiu (2020): monthly cross-sectional ranking,
top-decile long / bottom-decile short, equal-weighted within decile, rebalanced
on the last trading day of each month. Transaction costs are charged on
turnover at `transaction_cost_bps` basis points per dollar traded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import pandas as pd


# --------------------------------------------------------------------------
# Public types
# --------------------------------------------------------------------------

class CrossSectionalModel(Protocol):
    """Minimal interface B's model must satisfy.

    A walk-forward step calls `fit` on the training window, then `predict`
    on the test window. The backtest engine does NOT see the model's
    internals - it only consumes the predicted scores.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CrossSectionalModel":
        """Fit on a (features, next-period return) panel. Returns self."""
        ...

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Return a Series of predicted next-period returns, indexed like `X`."""
        ...


# Type alias for C's regime overlay.
LeverageFn = Callable[[pd.Timestamp], float]


@dataclass(frozen=True)
class BacktestResult:
    """Container for everything a downstream tear sheet needs.

    Attributes
    ----------
    portfolio_returns : pd.Series
        Net daily returns of the long-short portfolio AFTER transaction costs
        and AFTER the regime leverage overlay has been applied. Indexed by
        trading day.
    gross_returns : pd.Series
        Same shape as `portfolio_returns` but BEFORE costs and BEFORE the
        leverage overlay. Useful for attributing performance to alpha vs.
        regime timing vs. cost drag.
    weights : pd.DataFrame
        Wide-format target weights (one row per rebalance date, one column
        per ticker, values sum to ~0 because long+short).
    turnover : pd.Series
        L1 turnover at each rebalance date, expressed as a fraction of NAV.
    leverage : pd.Series
        The gross leverage actually applied on each trading day (output of
        `leverage_fn` carried forward between rebalances).
    metadata : dict
        Free-form bookkeeping: train/test window sizes, cost assumption,
        random seed, software versions. Goes straight into the report.
    """

    portfolio_returns: pd.Series
    gross_returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    leverage: pd.Series
    metadata: dict


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def run_walk_forward_backtest(
    returns: pd.DataFrame,
    features: pd.DataFrame,
    model: CrossSectionalModel,
    *,
    train_window: int,
    test_window: int,
    transaction_cost_bps: float = 10.0,
    leverage_fn: LeverageFn | None = None,
    long_quantile: float = 0.9,
    short_quantile: float = 0.1,
    rebalance: str = "M",
    random_state: int = 42,
) -> BacktestResult:
    """Run a walk-forward backtest of the long-short factor strategy.

    Walk-forward scheme
    -------------------
    Time is partitioned into contiguous, non-overlapping **test** blocks of
    length `test_window` (in rebalance periods, NOT in days). Each test
    block is preceded by a **train** window of length `train_window`
    immediately before it. Schematically::

        |---- train ----||-- test --||---- train ----||-- test --| ...
                          ^ refit                      ^ refit
                          here                         here

    Within a test block the model is FROZEN; only at the start of the next
    test block is it refit on the most recent `train_window` of data.
    This avoids the standard k-fold leakage that destroys time-series
    backtests (see Lopez de Prado, AFML ch. 11).

    Parameters
    ----------
    returns : pd.DataFrame
        Wide-format daily total returns. Index = trading days
        (timezone-naive). Columns = tickers. NaN means "not in the
        investable universe on that day" (delisted, not yet listed, halted)
        and is treated as out-of-universe, not as a zero return.
    features : pd.DataFrame
        Long-format feature panel indexed by `(date, ticker)`. Each row is
        ONE asset on ONE rebalance date. The columns are the factor values
        (momentum, value, size, etc.) produced by Person B's pipeline. Must
        be lagged appropriately - the value at date `t` is what was
        observable at the close of `t`, and is used to predict the return
        from `t` to the next rebalance date.
    model : CrossSectionalModel
        Object satisfying the `CrossSectionalModel` protocol above. The
        engine clones the trained model between blocks; it does not mutate
        the instance the caller passes in.
    train_window : int
        Number of rebalance periods to use for fitting at each refit.
        Example: with `rebalance="M"` and `train_window=60`, the model
        sees 5 years of monthly cross-sections.
    test_window : int
        Number of rebalance periods to use the frozen model for before the
        next refit. Smaller = more refits = more compute. Typical: 12.
    transaction_cost_bps : float, default 10.0
        Linear cost in basis points charged on the L1 turnover at each
        rebalance. 10 bps round-trip is a sane S&P 500 baseline.
    leverage_fn : Callable[[pd.Timestamp], float], optional
        Person C's regime overlay. Called once per trading day with that
        day's timestamp; must return a scalar in [0, 1]. If `None`, the
        backtest runs at constant 1.0x gross leverage (no regime overlay).
    long_quantile, short_quantile : float
        Cross-sectional quantile cutoffs for the long and short legs.
        Defaults reproduce Gu-Kelly-Xiu's top/bottom decile.
    rebalance : str, default "M"
        Pandas offset alias for the rebalance frequency. "M" = last
        business day of each month.
    random_state : int, default 42
        Forwarded to the model and to any sampling step inside the engine.
        Set explicitly so re-runs are bit-identical.

    Returns
    -------
    BacktestResult
        See the `BacktestResult` dataclass docstring.

    Raises
    ------
    ValueError
        If `returns` and `features` cover disjoint date ranges, or if
        `train_window + test_window` exceeds the available history.

    Notes
    -----
    * No look-ahead: features at date `t` are used to predict returns
      realised AFTER `t`. The engine asserts this with an explicit shift.
    * Survivorship: only assets present in `returns.columns` on a given
      rebalance date are eligible. Pass the point-in-time universe from
      `data_loader.load_sp500_membership` to do this correctly.
    * Transaction costs are charged at the moment of rebalance, on the
      L1 weight change, BEFORE the regime leverage is applied (so a
      regime-induced de-leverage also pays costs).
    """
    raise NotImplementedError(
        "run_walk_forward_backtest: see issue #TODO for the implementation plan"
    )


# --------------------------------------------------------------------------
# Interface versioning - bump when the signature above changes.
# --------------------------------------------------------------------------

INTERFACE_VERSION = "0.1.0"
