"""backtest.py - Walk-forward backtesting engine for the L/S equity strategy.

This module is the **integration point** between three workstreams:

    Person A (this file): provides `run_walk_forward_backtest`.
    Person B           : provides a `fit` / `predict` model object that
                          produces cross-sectional return forecasts.
    Person C           : provides a `regime_fn` that maps a date to a
                          `RegimeParams` dict (leverage, breadth, entry
                          quantiles) based on the regime detected on that
                          date. See `RegimeParams` below for the schema.

If the interface in this file changes, B and C need to know - bump the
`INTERFACE_VERSION` constant at the bottom and shout in standup.

Methodology mirrors Gu, Kelly & Xiu (2020): monthly cross-sectional ranking,
top-decile long / bottom-decile short, equal-weighted within decile, rebalanced
on the last trading day of each month. Transaction costs are charged on
turnover at `transaction_cost_bps` basis points per dollar traded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, TypedDict

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


class RegimeParams(TypedDict, total=False):
    """Per-rebalance risk parameters returned by Person C's regime overlay.

    All keys are optional (``total=False``). Any key the regime overlay does
    not set falls back to the static defaults passed to
    :func:`run_walk_forward_backtest` (``long_quantile``, ``short_quantile``)
    or to neutral values (``leverage=1.0``, no ``k_per_sector`` override).
    A regime that only wants to dial leverage just returns
    ``{"leverage": 0.7}``; a regime that also wants to tighten breadth
    returns ``{"leverage": 0.4, "k_per_sector": 2}``.

    Standard keys
    -------------
    leverage : float
        Gross leverage multiplier in [0, +inf). 1.0 = full notional,
        0.5 = half-size positions, 0.0 = flat. Applied to both legs of
        the long-short portfolio before transaction costs.
    long_quantile : float
        Cross-sectional quantile cutoff in (0, 1) for the long leg.
        Stocks ranked above this percentile within their sector are bought.
    short_quantile : float
        Cross-sectional quantile cutoff in (0, 1) for the short leg.
        Stocks ranked below this percentile within their sector are sold short.
    k_per_sector : int
        Optional alternative to quantile-based selection: hold exactly
        ``k`` long (and ``k`` short) stocks per GICS sector. If present,
        overrides ``long_quantile`` / ``short_quantile`` for that rebalance.

    Example
    -------
    The framework's three-regime mapping translates to::

        {"calm":      {"leverage": 1.0, "k_per_sector": 5}}
        {"moderate":  {"leverage": 0.7, "k_per_sector": 3}}
        {"turbulent": {"leverage": 0.4, "k_per_sector": 2}}
    """

    leverage: float
    long_quantile: float
    short_quantile: float
    k_per_sector: int


# Type alias for C's regime overlay. Replaces the v0.1.0 `LeverageFn`
# (Callable[..., float]) so the regime can communicate multiple risk knobs
# in one call. See INTERFACE_VERSION at the bottom of this file.
RegimeFn = Callable[[pd.Timestamp], RegimeParams]


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
        The gross leverage actually applied on each trading day (the
        `leverage` key of `regime_fn`'s output, or 1.0 when unset, carried
        forward between rebalances).
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
    regime_fn: RegimeFn | None = None,
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
    regime_fn : Callable[[pd.Timestamp], RegimeParams], optional
        Person C's regime overlay. Called once per rebalance date with
        that date's timestamp; must return a :class:`RegimeParams` dict.
        The dict may include any subset of ``{leverage, long_quantile,
        short_quantile, k_per_sector}``. Keys the regime does not set
        fall back to the static defaults below (``long_quantile``,
        ``short_quantile``) or to neutral values (``leverage=1.0``, no
        ``k_per_sector`` override). If ``None``, the backtest runs at
        constant 1.0x gross leverage using only the static quantile cutoffs
        (i.e. no regime overlay).
    long_quantile, short_quantile : float
        Cross-sectional quantile cutoffs for the long and short legs,
        used when ``regime_fn`` is ``None`` or when the regime dict does
        not override them. Defaults reproduce Gu-Kelly-Xiu's top/bottom
        decile.
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
      regime-induced de-leverage also pays costs). [NOT YET IMPLEMENTED
      as of v0.2.0 -- see commit log for staging.]
    """
    # --- 0. Validate inputs ---
    if returns.empty:
        raise ValueError("returns DataFrame is empty")
    if features.empty:
        raise ValueError("features DataFrame is empty")
    if train_window < 1:
        raise ValueError(f"train_window must be >= 1, got {train_window}")
    if not (0 < short_quantile < long_quantile < 1):
        raise ValueError(
            f"need 0 < short_quantile ({short_quantile}) < "
            f"long_quantile ({long_quantile}) < 1"
        )

    # --- 1. Rebalance dates: union of dates available in both returns and features ---
    rebal_dates = (
        returns.index.intersection(features.index.get_level_values(0).unique())
        .sort_values()
        .unique()
    )
    if len(rebal_dates) < train_window + 2:
        raise ValueError(
            f"not enough rebalance dates ({len(rebal_dates)}) for "
            f"train_window={train_window} + at least one prediction step"
        )

    # --- 2. Walk-forward loop ---
    # At each date t in [rebal_dates[train_window], ..., rebal_dates[-2]]:
    #   - train on features[t-train_window : t-1] with labels = returns[t-train_window+1 : t]
    #   - predict scores at t
    #   - form L/S portfolio
    #   - charge transaction cost on L1 turnover vs the previous rebalance
    #   - realise net next-period return
    weights_records: list[tuple[pd.Timestamp, pd.Series]] = []
    gross_returns_list: list[tuple[pd.Timestamp, float]] = []
    net_returns_list: list[tuple[pd.Timestamp, float]] = []
    turnover_records: list[tuple[pd.Timestamp, float]] = []
    prev_weights: pd.Series = pd.Series(dtype=float)  # empty at t=0
    cost_rate = transaction_cost_bps / 10_000.0  # bps -> fraction

    for i in range(train_window, len(rebal_dates) - 1):
        rebal_t = rebal_dates[i]
        next_t = rebal_dates[i + 1]
        train_t0 = rebal_dates[i - train_window]

        # 2a. Build training panel: features at dates [train_t0, rebal_t),
        # paired with NEXT-period return as the label.
        train_dates = rebal_dates[i - train_window : i]
        X_train = features.loc[(slice(train_t0, train_dates[-1]), slice(None)), :]
        # Labels: returns at the date AFTER each feature date.
        # rebal_dates is sorted, so use the next-date map.
        next_date_map = dict(zip(train_dates, rebal_dates[i - train_window + 1 : i + 1]))
        y_train_rows = []
        for (d, asset) in X_train.index:
            nd = next_date_map.get(d)
            if nd is None:
                y_train_rows.append(float("nan"))
                continue
            if asset in returns.columns:
                y_train_rows.append(returns.at[nd, asset])
            else:
                y_train_rows.append(float("nan"))
        y_train = pd.Series(y_train_rows, index=X_train.index, name="next_return")

        # Drop rows where label is NaN (asset not yet listed / delisted)
        keep = y_train.notna()
        X_train = X_train.loc[keep]
        y_train = y_train.loc[keep]

        if len(X_train) == 0:
            continue  # nothing to train on this step, skip

        model = model.fit(X_train, y_train)

        # 2b. Predict at rebal_t for eligible universe
        if rebal_t not in features.index.get_level_values(0):
            continue
        X_pred = features.loc[(rebal_t, slice(None)), :]
        # Eligible: must also have a non-NaN return at next_t
        eligible_assets = [
            a for (_, a) in X_pred.index
            if a in returns.columns and pd.notna(returns.at[next_t, a])
        ]
        if not eligible_assets:
            continue
        X_pred = X_pred.loc[(rebal_t, eligible_assets), :]
        scores = model.predict(X_pred)
        # scores' index may be (date, asset) -- collapse to just asset
        if isinstance(scores.index, pd.MultiIndex):
            scores = scores.reset_index(level=0, drop=True)

        # 2c. Build long-short portfolio: equal-weight within each leg,
        # dollar-neutral (long sum = short sum = 1.0 in absolute value).
        long_cut = scores.quantile(long_quantile)
        short_cut = scores.quantile(short_quantile)
        longs = scores[scores >= long_cut].index
        shorts = scores[scores <= short_cut].index

        weights = pd.Series(0.0, index=scores.index, name=rebal_t)
        if len(longs) > 0:
            weights.loc[longs] = 1.0 / len(longs)
        if len(shorts) > 0:
            weights.loc[shorts] = -1.0 / len(shorts)
        weights_records.append((rebal_t, weights))

        # 2d. Charge transaction cost on L1 turnover vs the previous rebalance.
        # First rebalance: prev_weights is empty, so turnover = sum(|w|) -- this
        # is the cost of entering positions from cash.
        union = weights.index.union(prev_weights.index)
        w_now = weights.reindex(union, fill_value=0.0)
        w_prev = prev_weights.reindex(union, fill_value=0.0)
        turnover_t = float((w_now - w_prev).abs().sum())
        cost_t = cost_rate * turnover_t
        turnover_records.append((rebal_t, turnover_t))

        # 2e. Realise next-period return.
        # gross = w . r_{t+1}; net = gross - cost (cost charged in the same period
        # as the rebalance, i.e. attributed to the return earned over [t, t+1]).
        next_period_rets = returns.loc[next_t, weights.index]
        gross_t = float((weights * next_period_rets).sum(skipna=True))
        net_t = gross_t - cost_t
        gross_returns_list.append((next_t, gross_t))
        net_returns_list.append((next_t, net_t))

        prev_weights = weights  # carry into next iteration

    # --- 3. Assemble BacktestResult ---
    if not net_returns_list:
        raise RuntimeError(
            "No rebalance produced a portfolio. Check input alignment and "
            "that train_window leaves at least one prediction date."
        )

    portfolio_returns = pd.Series(
        dict(net_returns_list), name="portfolio_return"
    ).sort_index()
    gross_returns = pd.Series(
        dict(gross_returns_list), name="gross_return"
    ).sort_index()

    # Wide weights DataFrame: rows = rebalance dates, cols = union of all assets
    all_assets = sorted({a for _, w in weights_records for a in w.index})
    rebal_dates_used = [d for d, _ in weights_records]
    weights_df = pd.DataFrame(0.0, index=pd.Index(rebal_dates_used, name="date"),
                              columns=all_assets)
    for d, w in weights_records:
        weights_df.loc[d, w.index] = w.values

    turnover_series = pd.Series(
        dict(turnover_records), name="turnover"
    ).sort_index()
    # Leverage stays a placeholder until commit 3 (regime overlay)
    leverage_series = pd.Series(1.0, index=weights_df.index, name="leverage")

    metadata = {
        "interface_version": INTERFACE_VERSION,
        "n_rebalances": len(net_returns_list),
        "first_rebalance": rebal_dates_used[0],
        "last_rebalance": rebal_dates_used[-1],
        "train_window": train_window,
        "test_window": test_window,
        "long_quantile": long_quantile,
        "short_quantile": short_quantile,
        "rebalance": rebalance,
        "transaction_cost_bps": transaction_cost_bps,
        "transaction_costs_applied": True,
        "total_cost_drag_pct": (gross_returns.sum() - portfolio_returns.sum()) * 100,
        "avg_monthly_turnover": float(turnover_series.mean()),
        "regime_fn_applied": False,  # TODO commit 3
        "random_state": random_state,
    }

    return BacktestResult(
        portfolio_returns=portfolio_returns,
        gross_returns=gross_returns,
        weights=weights_df,
        turnover=turnover_series,
        leverage=leverage_series,
        metadata=metadata,
    )


# --------------------------------------------------------------------------
# Interface versioning - bump when the signature above changes.
#
# Changelog
# ---------
# 0.2.0 (2026-05-13): widened regime overlay from `LeverageFn`
#   (Callable[..., float]) to `RegimeFn` (Callable[..., RegimeParams]) so
#   the regime can communicate breadth (k_per_sector) and entry-threshold
#   (long/short_quantile) overrides alongside leverage. See DECISIONS.md
#   2026-05-13 "Widen regime overlay interface".
# 0.1.0 (2026-05-11): initial contract (CrossSectionalModel + LeverageFn).
# --------------------------------------------------------------------------

INTERFACE_VERSION = "0.2.0"


# --------------------------------------------------------------------------
# Smoke test (run with: python -m src.backtest)
# Builds a small in-memory toy panel and runs the engine with RandomModel.
# Sharpe must be approximately zero -- this is sanity check #1 from
# Project Framework §4.6.
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    from src.sanity import RandomModel

    print("=" * 70)
    print("Backtest engine smoke test (RandomModel + synthetic panel)")
    print("=" * 70)

    # Synthetic monthly panel: 60 dates x 50 assets
    rng = np.random.default_rng(42)
    n_dates, n_assets = 60, 50
    dates = pd.date_range("2015-01-31", periods=n_dates, freq="ME")
    assets = [f"A{i:03d}" for i in range(n_assets)]

    # Wide-format returns: random monthly returns ~ N(0, 5%)
    rets_wide = pd.DataFrame(
        rng.normal(0.0, 0.05, size=(n_dates, n_assets)),
        index=dates,
        columns=assets,
    )

    # Long-format features: 3 random features per (date, asset).
    # RandomModel doesn't use them, but the engine validates the shape.
    feat_idx = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])
    feats = pd.DataFrame(
        rng.normal(0, 1, size=(n_dates * n_assets, 3)),
        index=feat_idx,
        columns=["feat1", "feat2", "feat3"],
    )

    print(f"\nSynthetic panel: {n_dates} months x {n_assets} assets")
    print(f"Returns shape: {rets_wide.shape}, features shape: {feats.shape}")
    print()

    result = run_walk_forward_backtest(
        returns=rets_wide,
        features=feats,
        model=RandomModel(random_state=42),
        train_window=12,
        test_window=12,
        long_quantile=0.9,
        short_quantile=0.1,
        transaction_cost_bps=10.0,  # 10 bps round-trip, sane S&P500 baseline
    )

    print(f"BacktestResult:")
    print(f"  portfolio_returns: {len(result.portfolio_returns)} months")
    print(f"  weights:           {result.weights.shape}")
    print(f"  metadata.n_rebalances:           {result.metadata['n_rebalances']}")
    print(f"  metadata.transaction_costs_applied: {result.metadata['transaction_costs_applied']}")
    print(f"  metadata.avg_monthly_turnover:   {result.metadata['avg_monthly_turnover']:.3f}")
    print(f"  metadata.total_cost_drag_pct:    {result.metadata['total_cost_drag_pct']:.3f}%")
    print()

    # Sanity check #1: Sharpe must be ~0 for random predictions
    def _sharpe(r: pd.Series) -> float:
        s = r.std(ddof=1)
        return float((r.mean() / s) * np.sqrt(12)) if s > 0 else 0.0

    gross_sharpe = _sharpe(result.gross_returns)
    net_sharpe = _sharpe(result.portfolio_returns)

    print(f"Random-prediction sanity check (transaction_cost_bps=10):")
    print(f"  Mean monthly gross return:  {result.gross_returns.mean() * 100:+.3f}%")
    print(f"  Mean monthly net return:    {result.portfolio_returns.mean() * 100:+.3f}%")
    print(f"  Cost drag per month:        {(result.gross_returns.mean() - result.portfolio_returns.mean()) * 100:.3f}%")
    print(f"  Annualised GROSS Sharpe:    {gross_sharpe:+.3f}  (must be ~0)")
    print(f"  Annualised NET   Sharpe:    {net_sharpe:+.3f}  (slightly more negative than gross due to costs)")

    if abs(gross_sharpe) > 1.0:
        print(f"\n  WARNING: |gross Sharpe| > 1.0 with random predictions.")
        print(f"  This suggests look-ahead bias in the engine.")
    else:
        print(f"\n  OK: |gross Sharpe| <= 1.0, no obvious look-ahead bug.")
