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
