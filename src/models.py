"""models.py - Cross-sectional return-prediction models.

Person B owns this file. Each model in here satisfies the
`CrossSectionalModel` protocol from `src.backtest`:

    fit(X: pd.DataFrame, y: pd.Series) -> self
    predict(X: pd.DataFrame) -> pd.Series  # indexed like X

That is the only contract the backtest engine relies on. Models therefore
do not need to know which date or which stock a row is for -- the engine
slices and aligns. Training data is the cross-section concatenated across
the train window; prediction is the cross-section at a single rebalance.

Model lineup (Project Framework §3.4)
-------------------------------------
* ``LassoModel``    -- L1-regularised linear baseline (sklearn LassoCV).
* ``XGBoostModel``  -- gradient-boosted trees (primary, per GKX 2020).
* ``NNModel``       -- small feedforward net (secondary baseline).

All three set ``random_state=42`` per the project's reproducibility rule.
Hyperparameters here are sane defaults; tuning happens later on the
validation window (2016-2018) -- see Framework §7.2.
"""

from __future__ import annotations

import os

# Tolerate Anaconda's libomp loaded alongside the torch wheel's libomp -- a
# known deadlock cause on macOS x86_64 Python 3.12 setups (verified 2026-05-21).
# setdefault so user-supplied values win; safe to leave unconditionally.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from typing import Any

import numpy as np
import pandas as pd

# Sector is carried in the feature panel for downstream Layer-1 / Layer-2
# logic but is NOT itself a predictive feature -- models must drop it
# before fitting. This list extends if we add other non-predictive
# bookkeeping columns later.
_NON_PREDICTIVE_COLS: tuple[str, ...] = ("sector",)


def _split_X(X: pd.DataFrame) -> pd.DataFrame:
    """Return X with non-predictive columns (e.g., 'sector') stripped."""
    drop = [c for c in _NON_PREDICTIVE_COLS if c in X.columns]
    return X.drop(columns=drop) if drop else X


def _impute_train(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Median-impute NaNs in training features. Returns (X_filled, medians)."""
    medians = X.median(numeric_only=True)
    return X.fillna(medians), medians


def _impute_apply(X: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    """Apply training-time medians to predict-time features."""
    return X.fillna(medians)


# --------------------------------------------------------------------------
# Layer-2 helper: sector-relative target
# --------------------------------------------------------------------------

_VALID_TARGET_KINDS: tuple[str, ...] = ("raw", "sector_relative")


def _check_target_kind(target_kind: str) -> str:
    if target_kind not in _VALID_TARGET_KINDS:
        raise ValueError(
            f"target_kind must be one of {_VALID_TARGET_KINDS}, "
            f"got {target_kind!r}"
        )
    return target_kind


def _demean_y_by_sector_date(y: pd.Series, X: pd.DataFrame) -> pd.Series:
    """Subtract per-(date, sector) mean from y for the Layer-2 target.

    Implements Project Framework section 3.2 Layer 2: predict excess
    return over sector mean instead of raw return. The learning
    objective becomes sector-neutral by construction.

    Requirements:
      * X must carry a ``"sector"`` column (the feature panel produced by
        :func:`factors.build_feature_panel` does so by default; the
        notebook must NOT drop it before passing features to the engine).
      * X must have a (date, asset) MultiIndex (the date level is what
        the engine uses for the cross-section alignment).

    Parameters
    ----------
    y : pd.Series
        Raw next-period returns, indexed like X.
    X : pd.DataFrame
        Feature panel including 'sector' column.

    Returns
    -------
    pd.Series
        Same index as y, values = y - mean_over_(date, sector)(y).
    """
    if "sector" not in X.columns:
        raise ValueError(
            "target_kind='sector_relative' requires a 'sector' column in "
            "features. Call build_feature_panel(...) with the default "
            "sector_rank=True and do NOT drop 'sector' before passing X "
            "to the model."
        )
    if not isinstance(X.index, pd.MultiIndex) or len(X.index.names) < 2:
        raise ValueError(
            "target_kind='sector_relative' requires X to have a "
            "(date, asset) MultiIndex."
        )
    date = X.index.get_level_values(0)
    sector = X["sector"].to_numpy()
    sector_date_mean = y.groupby([date, sector]).transform("mean")
    demeaned = y - sector_date_mean
    # Fix 4 (audit 2026-05-23): verify the demean leaves zero residual within
    # every (date, sector) group. Should always pass; defensive against a
    # future regression (pandas groupby ordering quirk, accidental index
    # misalignment, etc.) that would silently kill Layer-2's sector-neutral
    # learning objective.
    residual_max = (
        demeaned.groupby([date, sector]).transform("mean").abs().max()
    )
    if residual_max > 1e-9:
        raise AssertionError(
            f"Layer-2 sector-demean residual = {residual_max:.2e} exceeds "
            f"1e-9 tolerance. The (date, sector) group means of the "
            f"demeaned target are not zero; demean is broken."
        )
    return demeaned


# --------------------------------------------------------------------------
# 1. Lasso baseline
# --------------------------------------------------------------------------

class LassoModel:
    """L1-regularised linear baseline with CV over alpha.

    Wraps :class:`sklearn.linear_model.LassoCV`. Standardises features
    on the training window before fitting so the L1 penalty is comparable
    across columns. Replaces NaNs with the column median (training-window
    median, applied unchanged at predict time).

    Parameters
    ----------
    alphas : array-like, optional
        Candidate regularisation strengths. ``None`` -> LassoCV's default
        grid (logspace from 1e-3 to 1).
    cv : int, default 5
        K-fold splits for alpha selection. The folds are random within
        the training window, which is fine because each row is a
        (stock, date) snapshot and the cross-section is exchangeable
        WITHIN a date. (Walk-forward is enforced at the OUTER loop by
        the backtest engine, not inside the model.)
    random_state : int, default 42
    target_kind : {"raw", "sector_relative"}, default "raw"
        ``"raw"``: train on next-period return directly. ``"sector_relative"``:
        train on next-period return minus the per-(date, sector) mean
        (Framework section 3.2 Layer 2). Requires a 'sector' column on X.
    """

    def __init__(
        self,
        alphas: "list[float] | None" = None,
        cv: int = 5,
        random_state: int = 42,
        target_kind: str = "raw",
    ) -> None:
        from sklearn.linear_model import LassoCV
        from sklearn.preprocessing import StandardScaler

        self.target_kind = _check_target_kind(target_kind)
        self._scaler = StandardScaler()
        self._lasso = LassoCV(
            alphas=alphas if alphas is not None else 100,
            cv=cv,
            random_state=random_state,
            n_jobs=1,  # n_jobs=-1 deadlocks on macOS+py3.12 joblib (verified 2026-05-21)
            max_iter=20_000,
        )
        self._medians: pd.Series | None = None
        self._feature_cols: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LassoModel":
        if self.target_kind == "sector_relative":
            y = _demean_y_by_sector_date(y, X)
        X_use = _split_X(X)
        X_filled, medians = _impute_train(X_use)
        self._medians = medians
        self._feature_cols = list(X_filled.columns)
        X_scaled = self._scaler.fit_transform(X_filled.to_numpy())
        self._lasso.fit(X_scaled, y.to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._medians is None:
            raise RuntimeError("LassoModel.predict called before fit()")
        X_use = _split_X(X).reindex(columns=self._feature_cols)
        X_filled = _impute_apply(X_use, self._medians)
        scores = self._lasso.predict(self._scaler.transform(X_filled.to_numpy()))
        return pd.Series(scores, index=X.index, name="lasso_score")


# --------------------------------------------------------------------------
# 2. XGBoost (primary)
# --------------------------------------------------------------------------

class XGBoostModel:
    """Gradient-boosted trees -- the project's primary model per GKX 2020.

    Defaults are conservative and roughly match what GKX use for the
    "shallow learning wins" finding (max_depth ~ 3-5, modest learning
    rate, a few hundred trees). XGBoost handles NaNs natively so no
    imputation step is needed.

    Parameters
    ----------
    n_estimators : int, default 150
        Phase-3 Optuna optimum. Half the textbook default (300); the
        tuner converged on a smaller-but-more-regularised forest.
    max_depth : int, default 4
    learning_rate : float, default 0.015
        Phase-3 optimum. Slow learning rate; lets the regularisation work.
    subsample : float, default 0.815
    colsample_bytree : float, default 0.734
    min_child_weight : int, default 15
        Phase-3 optimum. Higher than the textbook default of 1; cuts
        overfitting on small leaf subsamples.
    reg_alpha : float, default 0.395
        Phase-3 optimum. L1 regularisation on the leaf weights.
    reg_lambda : float, default 2.852
        Phase-3 optimum. L2 regularisation; tighter than the
        textbook default of 1.0.
    random_state : int, default 42
    extra_kwargs : dict, optional
        Forwarded to ``xgboost.XGBRegressor``. Use for one-off tuning
        without growing the explicit parameter list.
    target_kind : {"raw", "sector_relative"}, default "raw"
        See :class:`LassoModel` for semantics. XGBoost is not scale-
        sensitive but the loss landscape changes when the target is
        demeaned; expect a noticeable IC shift either way.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 3,
        learning_rate: float = 0.0115,
        subsample: float = 0.717,
        colsample_bytree: float = 0.890,
        min_child_weight: int = 11,
        reg_alpha: float = 0.794,
        reg_lambda: float = 2.305,
        random_state: int = 42,
        extra_kwargs: "dict[str, Any] | None" = None,
        target_kind: str = "raw",
    ) -> None:
        # Defaults are the Phase-8 Optuna optimum on the 13-feature panel
        # (validation window 2016-2018, objective: OOS R^2 vs zero).
        # See results/03_xgboost_tuning/best_params.json. Pattern shifted vs
        # the 8-feature tune: shallower trees (max_depth 4 -> 3), more
        # aggressive L1 (reg_alpha 0.44 -> 0.79) so the model uses L1 for
        # feature selection across the larger feature set; slightly less L2.
        # Previous 8-feature defaults were: n_estimators=200, max_depth=4,
        # learning_rate=0.0104, subsample=0.701, colsample_bytree=0.711,
        # min_child_weight=14, reg_alpha=0.444, reg_lambda=3.144.
        import xgboost as xgb

        self.target_kind = _check_target_kind(target_kind)
        self._xgb = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            tree_method="hist",
            random_state=random_state,
            n_jobs=1,  # safer on macOS; bump if training time becomes a bottleneck
            **(extra_kwargs or {}),
        )
        self._feature_cols: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostModel":
        if self.target_kind == "sector_relative":
            y = _demean_y_by_sector_date(y, X)
        X_use = _split_X(X)
        self._feature_cols = list(X_use.columns)
        self._xgb.fit(X_use.to_numpy(), y.to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._feature_cols is None:
            raise RuntimeError("XGBoostModel.predict called before fit()")
        X_use = _split_X(X).reindex(columns=self._feature_cols)
        scores = self._xgb.predict(X_use.to_numpy())
        return pd.Series(scores, index=X.index, name="xgb_score")

    @property
    def feature_importances_(self) -> pd.Series:
        """Gain-based feature importance from the fitted XGBoost model."""
        if self._feature_cols is None:
            raise RuntimeError("feature_importances_ accessed before fit()")
        return pd.Series(
            self._xgb.feature_importances_,
            index=self._feature_cols,
            name="gain",
        ).sort_values(ascending=False)


# --------------------------------------------------------------------------
# 3. Feedforward NN (secondary baseline)
# --------------------------------------------------------------------------

class NNModel:
    """Small feedforward net for the GKX "NN3" baseline.

    Architecture: input -> (Linear+ReLU+Dropout) x ``n_layers`` -> Linear(1).
    Trained with Adam, MSE loss, early stopping on a held-out 20%
    validation slice of the training window. CPU is fine for the scale
    of this project (~10k rows per training window, ~6 features).

    Parameters
    ----------
    hidden_dim : int, default 32
    n_layers : int, default 3
    dropout : float, default 0.2
    learning_rate : float, default 1e-3
    weight_decay : float, default 1e-4
    batch_size : int, default 512
    max_epochs : int, default 100
    patience : int, default 10
        Early-stopping patience on the val slice.
    random_state : int, default 42
    target_kind : {"raw", "sector_relative"}, default "raw"
        See :class:`LassoModel` for semantics.
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        n_layers: int = 3,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 512,
        max_epochs: int = 100,
        patience: int = 10,
        random_state: int = 42,
        target_kind: str = "raw",
    ) -> None:
        self.target_kind = _check_target_kind(target_kind)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state

        self._net: Any = None
        self._medians: pd.Series | None = None
        self._scaler: Any = None
        self._feature_cols: list[str] | None = None

    def _build_net(self, n_features: int) -> Any:
        import torch
        import torch.nn as nn

        torch.manual_seed(self.random_state)
        layers: list[nn.Module] = []
        in_dim = n_features
        for _ in range(self.n_layers):
            layers += [
                nn.Linear(in_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            ]
            in_dim = self.hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NNModel":
        import torch
        from sklearn.preprocessing import StandardScaler

        # Single-threaded torch in this fit. Two reasons:
        # (a) the dataset is small (~50k rows) so threading does not help;
        # (b) running NNModel.fit ~120 times in a walk-forward loop with
        #     multi-threaded torch deadlocks on this macOS/conda setup
        #     after roughly 100 iterations (observed 2026-05-22). Pinning
        #     to one thread eliminates the deadlock with no measurable
        #     wall-clock cost on a panel this size.
        torch.set_num_threads(1)

        if self.target_kind == "sector_relative":
            y = _demean_y_by_sector_date(y, X)
        X_use = _split_X(X)
        X_filled, medians = _impute_train(X_use)
        self._medians = medians
        self._feature_cols = list(X_filled.columns)
        self._scaler = StandardScaler().fit(X_filled.to_numpy())
        X_scaled = self._scaler.transform(X_filled.to_numpy())
        y_arr = y.to_numpy(dtype=np.float32).reshape(-1, 1)

        rng = np.random.default_rng(self.random_state)
        n = len(X_scaled)
        idx = rng.permutation(n)
        n_val = max(1, int(0.2 * n))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]

        device = torch.device("cpu")
        net = self._build_net(X_scaled.shape[1]).to(device)
        opt = torch.optim.Adam(
            net.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = torch.nn.MSELoss()

        X_tr = torch.from_numpy(X_scaled[tr_idx].astype(np.float32)).to(device)
        y_tr = torch.from_numpy(y_arr[tr_idx]).to(device)
        X_val = torch.from_numpy(X_scaled[val_idx].astype(np.float32)).to(device)
        y_val = torch.from_numpy(y_arr[val_idx]).to(device)

        best_val = float("inf")
        best_state: dict | None = None
        stale = 0
        for _epoch in range(self.max_epochs):
            net.train()
            perm = torch.randperm(len(X_tr), generator=torch.Generator().manual_seed(self.random_state))
            for s in range(0, len(X_tr), self.batch_size):
                batch = perm[s: s + self.batch_size]
                opt.zero_grad()
                pred = net(X_tr[batch])
                loss = loss_fn(pred, y_tr[batch])
                loss.backward()
                opt.step()

            net.eval()
            with torch.no_grad():
                val_pred = net(X_val)
                val_loss = float(loss_fn(val_pred, y_val).item())
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        self._net = net
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        import torch

        if self._net is None or self._medians is None:
            raise RuntimeError("NNModel.predict called before fit()")
        X_use = _split_X(X).reindex(columns=self._feature_cols)
        X_filled = _impute_apply(X_use, self._medians)
        X_scaled = self._scaler.transform(X_filled.to_numpy().astype(np.float32))
        self._net.eval()
        with torch.no_grad():
            scores = self._net(torch.from_numpy(X_scaled)).cpu().numpy().ravel()
        return pd.Series(scores, index=X.index, name="nn_score")


# --------------------------------------------------------------------------
# Smoke test: each model runs against the synthetic panel from sanity.py
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    print("=" * 70)
    print("models.py smoke test (synthetic 60mo x 50 assets)")
    print("=" * 70)

    rng = np.random.default_rng(42)
    n_dates, n_assets = 60, 50
    dates = pd.date_range("2015-01-31", periods=n_dates, freq="ME")
    assets = [f"A{i:03d}" for i in range(n_assets)]
    feat_idx = pd.MultiIndex.from_product([dates, assets], names=["date", "ticker"])
    X = pd.DataFrame(
        rng.normal(0, 1, size=(n_dates * n_assets, 4)),
        index=feat_idx,
        columns=["mom", "rev", "mvol", "log_mktcap"],
    )
    X["sector"] = rng.choice(["Tech", "Financials", "Health Care"], size=len(X))
    # Linear-ish true target so all three models have something to learn.
    y = (0.3 * X["mom"] - 0.2 * X["mvol"] + rng.normal(0, 0.5, size=len(X)))

    for name, model in [
        ("Lasso  ", LassoModel()),
        ("XGBoost", XGBoostModel(n_estimators=100, max_depth=3)),
        ("NN     ", NNModel(max_epochs=10, patience=5)),
    ]:
        model.fit(X, y)
        preds = model.predict(X.iloc[:n_assets])  # first cross-section
        in_corr = float(np.corrcoef(model.predict(X), y)[0, 1])
        print(f"  {name}: in-sample corr(pred, y) = {in_corr:+.3f}, "
              f"pred head: {preds.head(3).round(3).tolist()}")

    print("\nAll three models fit + predict without errors.")
