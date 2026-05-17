"""sanity.py - Pseudo-models for backtest sanity checks.

Three deliberately-broken / deliberately-perfect "models" used to verify
that the backtest engine itself is correctly wired before any real model
(Lasso, XGBoost, NN) is trusted. Required gate per Person A Pipeline
Checklist Q15 and Project Framework §4.6:

    1. RandomModel  -> Sharpe must be approximately zero.
                        If materially > 0, there is look-ahead in the
                        pipeline (training labels leaking into test).
    2. OracleModel  -> Sharpe must be very large (5+).
                        If not, the backtest is mis-signing trades.
    3. UniformModel -> Strategy return must be approximately zero.
                        If not, the portfolio is secretly dependent on
                        prediction magnitudes (e.g., dollar-neutrality bug).

All three satisfy the `CrossSectionalModel` protocol in `src.backtest`:
they expose `fit(X, y)` (no-op for two of them) and `predict(X)`. They
take no hyperparameters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class RandomModel:
    """Predicts uniform random scores. Sharpe must be ~0 in a clean backtest.

    If a backtest with `RandomModel` produces materially positive Sharpe,
    there is look-ahead bias somewhere -- the model is "predicting" but
    that prediction is uninformative, so the only way to make money is
    if information about the future has leaked into the trading rule
    (e.g., the universe filter is using forward-looking data).
    """

    def __init__(self, *, random_state: int = 42) -> None:
        self._rng = np.random.default_rng(random_state)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomModel":  # noqa: ARG002
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(
            self._rng.standard_normal(len(X)),
            index=X.index,
            name="random_score",
        )


class OracleModel:
    """Predicts the realized next-period return -- perfect foresight.

    Sharpe must be very large (typically > 5) in a correctly-wired
    backtest. If it is not, the engine is mis-signing trades, mis-aligning
    dates, or otherwise failing to actually trade on the predictions.

    Construct with the same wide-format returns panel the backtest will
    use; predict() looks up the next-period return for each (date, asset)
    in the input cross-section.
    """

    def __init__(self, returns: pd.DataFrame) -> None:
        # Keep a sorted-date copy for fast "what's the next date" lookups.
        self._returns = returns.sort_index()
        self._date_index = self._returns.index

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "OracleModel":  # noqa: ARG002
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        # X.index is a MultiIndex of (date, asset) with a single date level
        # (the cross-section at one rebalance date).
        dates = X.index.get_level_values(0).unique()
        if len(dates) != 1:
            raise ValueError(
                f"OracleModel.predict expects a single rebalance date in X.index; "
                f"got {len(dates)} distinct dates"
            )
        rebal_t = dates[0]
        try:
            pos = self._date_index.get_loc(rebal_t)
        except KeyError as exc:
            raise KeyError(
                f"OracleModel: rebalance date {rebal_t} not in returns index"
            ) from exc
        if pos + 1 >= len(self._date_index):
            # No next period available; return zeros to avoid trading.
            return pd.Series(0.0, index=X.index, name="oracle_score")
        next_t = self._date_index[pos + 1]
        next_row = self._returns.loc[next_t]
        scores = [
            float(next_row[a]) if (a in next_row.index and pd.notna(next_row[a])) else 0.0
            for (_, a) in X.index
        ]
        return pd.Series(scores, index=X.index, name="oracle_score")


class UniformModel:
    """Predicts a constant for every (date, asset). Strategy must be flat.

    A correctly-implemented L/S engine that uses quantile cutoffs on
    predictions must produce a ZERO portfolio when all predictions are
    identical (no stock is in the top decile if every stock is at the
    same level). The realized return must therefore be approximately
    zero.

    If the engine produces a non-trivial return under this model, it is
    secretly conditioning on prediction magnitudes (e.g., weighting by
    prediction value), or it has a tie-breaking bug that puts the entire
    universe into one leg.
    """

    def __init__(self, *, constant: float = 0.5) -> None:
        self._c = float(constant)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "UniformModel":  # noqa: ARG002
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self._c, index=X.index, name="uniform_score")


# Threshold for "approximately zero" mean monthly return (10 bps/month).
_UNIFORM_RETURN_THRESHOLD = 0.001
# Threshold for "approximately zero" Sharpe on random predictions.
_RANDOM_SHARPE_THRESHOLD = 1.0
# Threshold for "very large" Sharpe on oracle predictions.
_ORACLE_SHARPE_THRESHOLD = 5.0


def run_sanity_checks(
    returns: pd.DataFrame,
    features: pd.DataFrame,
    *,
    train_window: int = 12,
    long_quantile: float = 0.9,
    short_quantile: float = 0.1,
    random_state: int = 42,
) -> dict[str, dict]:
    """Run the three mandatory backtest sanity checks.

    Calls ``run_walk_forward_backtest`` once for each pseudo-model
    (RandomModel, OracleModel, UniformModel) on the supplied
    (returns, features) panel, with transaction costs set to zero so
    the test isolates the model logic from the cost machinery.

    Parameters mirror ``run_walk_forward_backtest`` for the subset that
    affects model evaluation.

    Returns
    -------
    dict[str, dict]
        Keyed by check name (``"random"``, ``"oracle"``, ``"uniform"``).
        Each value is a dict with::

            {
                "sharpe":      float,   # annualised gross Sharpe
                "mean_return": float,   # mean monthly gross return
                "pass":        bool,    # threshold met?
                "message":     str,     # human-readable verdict
            }
    """
    # Import locally to avoid a circular import at module load time
    # (backtest.py imports from sanity.py for its smoke test).
    from src.backtest import run_walk_forward_backtest
    from src.metrics import sharpe_ratio

    common_kwargs = dict(
        returns=returns,
        features=features,
        train_window=train_window,
        test_window=12,
        long_quantile=long_quantile,
        short_quantile=short_quantile,
        transaction_cost_bps=0.0,  # isolate model behaviour from cost noise
    )

    results: dict[str, dict] = {}

    # --- 1. Random model: Sharpe must be ~0 ---
    rnd_res = run_walk_forward_backtest(
        model=RandomModel(random_state=random_state), **common_kwargs
    )
    rnd_sharpe = sharpe_ratio(rnd_res.gross_returns)
    rnd_pass = abs(rnd_sharpe) < _RANDOM_SHARPE_THRESHOLD
    results["random"] = {
        "sharpe": rnd_sharpe,
        "mean_return": float(rnd_res.gross_returns.mean()),
        "pass": rnd_pass,
        "message": (
            f"|Sharpe| = {abs(rnd_sharpe):.3f} < {_RANDOM_SHARPE_THRESHOLD}: "
            f"no look-ahead detected"
            if rnd_pass else
            f"|Sharpe| = {abs(rnd_sharpe):.3f} >= {_RANDOM_SHARPE_THRESHOLD}: "
            f"look-ahead bias likely (random predictions should not make money)"
        ),
    }

    # --- 2. Oracle model: Sharpe must be very large ---
    orc_res = run_walk_forward_backtest(
        model=OracleModel(returns=returns), **common_kwargs
    )
    orc_sharpe = sharpe_ratio(orc_res.gross_returns)
    orc_pass = orc_sharpe > _ORACLE_SHARPE_THRESHOLD
    results["oracle"] = {
        "sharpe": orc_sharpe,
        "mean_return": float(orc_res.gross_returns.mean()),
        "pass": orc_pass,
        "message": (
            f"Sharpe = {orc_sharpe:.2f} > {_ORACLE_SHARPE_THRESHOLD}: "
            f"engine trades correctly on the prediction sign"
            if orc_pass else
            f"Sharpe = {orc_sharpe:.2f} <= {_ORACLE_SHARPE_THRESHOLD}: "
            f"with perfect foresight, the engine is mis-signing trades, "
            f"mis-aligning dates, or otherwise not actually using predictions"
        ),
    }

    # --- 3. Uniform model: return must be ~0 ---
    uni_res = run_walk_forward_backtest(
        model=UniformModel(constant=0.5), **common_kwargs
    )
    uni_mean = float(uni_res.gross_returns.mean())
    uni_pass = abs(uni_mean) < _UNIFORM_RETURN_THRESHOLD
    results["uniform"] = {
        "sharpe": sharpe_ratio(uni_res.gross_returns),
        "mean_return": uni_mean,
        "pass": uni_pass,
        "message": (
            f"|mean monthly return| = {abs(uni_mean)*1e4:.2f} bps < "
            f"{_UNIFORM_RETURN_THRESHOLD*1e4:.0f} bps: "
            f"portfolio is flat when predictions don't differentiate stocks"
            if uni_pass else
            f"|mean monthly return| = {abs(uni_mean)*1e4:.2f} bps >= "
            f"{_UNIFORM_RETURN_THRESHOLD*1e4:.0f} bps: "
            f"engine secretly depends on prediction magnitudes (tie-breaking bug?)"
        ),
    }

    return results


# --------------------------------------------------------------------------
# Run with: python -m src.sanity
# Self-contained synthetic-panel sanity check.
# --------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Backtest engine sanity checks (Project Framework §4.6)")
    print("=" * 70)

    # Synthetic monthly panel
    rng = np.random.default_rng(42)
    n_dates, n_assets = 60, 50
    dates = pd.date_range("2015-01-31", periods=n_dates, freq="ME")
    assets = [f"A{i:03d}" for i in range(n_assets)]

    returns = pd.DataFrame(
        rng.normal(0.0, 0.05, size=(n_dates, n_assets)),
        index=dates, columns=assets,
    )
    feat_idx = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])
    features = pd.DataFrame(
        rng.normal(0, 1, size=(n_dates * n_assets, 3)),
        index=feat_idx, columns=["feat1", "feat2", "feat3"],
    )

    print(f"Synthetic panel: {n_dates} months x {n_assets} assets")
    print(f"  (transaction costs zeroed out to isolate model behaviour)\n")

    results = run_sanity_checks(returns=returns, features=features)

    print(f"{'Check':>10s}  {'Sharpe':>10s}  {'Mean ret %':>12s}  {'Pass':>6s}  Message")
    print("-" * 100)
    for name, r in results.items():
        flag = "PASS" if r["pass"] else "FAIL"
        print(
            f"{name:>10s}  "
            f"{r['sharpe']:>+10.3f}  "
            f"{r['mean_return']*100:>+12.4f}  "
            f"{flag:>6s}  {r['message']}"
        )

    n_pass = sum(1 for r in results.values() if r["pass"])
    print()
    print(f"Summary: {n_pass} / 3 sanity checks passed.")
    if n_pass < 3:
        import sys
        sys.exit(1)
