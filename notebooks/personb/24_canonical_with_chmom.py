"""Phase 23: FINAL CANONICAL on broad Sharadar universe.

Combines:
  * Broad universe (~2000 per date, top by mcap, PIT survivorship-free
    via TICKERS+DAILY) from Bowen's Sharadar pull
  * Pre-computed features panel (1.24M rows, 13 features + sector,
    sector-relative ranked) from Bowen's Block C
  * Pre-computed monthly returns (276 x 5897, SEP closeadj) from
    Bowen's Block B
  * Engine v0.5.0 with strict PIT (train+trade both filtered) via the
    cumulative ever-eligible universe derived from the features index
  * XGBoost retuned via Phase 23a Optuna on the broad panel

Differences from Phase 22:
  * Universe: ~2000 broad names (vs ~1000 relaxed-PIT S&P members)
  * Source: Sharadar SEP closeadj (vs CRSP+yfinance splice)
  * Features pre-computed (no factors.build_feature_panel call)
  * Returns pre-computed (no factors loaders required)
  * NN/Lasso/XGBoost params from Phase 23a's broad-panel Optuna

Output to ``results/23_canonical_broad_sharadar/``.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import run_walk_forward_backtest, RegimeParams
from src.metrics import (
    information_coefficient,
    oos_r2,
    summary_stats,
)
from src.models import LassoModel, NNModel, XGBoostModel


# Phase 24: 14 features (chmom added). XGBoost params from Phase 24a's
# retune on the 14-feature Q-filtered panel (val R^2 = +0.0055, +18% over
# Phase 23d's 13-feature tune). Lasso/NN params still from Phase 23a since
# they weren't retuned (less likely to benefit from chmom anyway).
import json as _json
_XGB_PARAMS_FILE = (
    Path(__file__).resolve().parents[2]
    / "results" / "24a_retune_xgb_with_chmom" / "best_params.json"
)
_OTHER_PARAMS_FILE = (
    Path(__file__).resolve().parents[2]
    / "results" / "23a_retune_broad_sharadar" / "best_params.json"
)
if _XGB_PARAMS_FILE.exists():
    with open(_XGB_PARAMS_FILE) as _f:
        RETUNED_XGB_PARAMS = _json.load(_f)["best_params"]
else:
    # Fallback: Phase 23a's tune (13-feature panel)
    with open(_OTHER_PARAMS_FILE) as _f:
        RETUNED_XGB_PARAMS = _json.load(_f)["by_model"]["xgboost"]["best_params"]
with open(_OTHER_PARAMS_FILE) as _f:
    _bp = _json.load(_f)["by_model"]
RETUNED_LASSO_PARAMS = _bp["lasso"]["best_params"]
RETUNED_NN_PARAMS = _bp["nn"]["best_params"]


def is_q_suffix_bankruptcy(t: str) -> bool:
    """Sharadar's bankruptcy convention: ticker ends in 'Q' (length >=4).
    See Phase 23c docstring for full justification."""
    t = str(t).upper().strip()
    return len(t) >= 4 and t.endswith("Q")


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

START = "2002-04-01"
END = "2024-12-31"
TEST_START = pd.Timestamp("2019-01-01")
TEST_END = pd.Timestamp("2024-12-31")

# 120 months = 10 years. With START=2002-04, the first prediction lands at
# 2012-04-30, extending long-OOS to ~12.75 years.
TRAIN_WINDOW = 120
TEST_WINDOW = 12  # block-gated refit cadence under engine v0.3.0

LONG_QUANTILE = 0.8
SHORT_QUANTILE = 0.2
TRANSACTION_COST_BPS = 10.0

RESULTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "results" / "24_canonical_with_chmom"
)
# k=20: same as Phase 23g for direct apples-to-apples comparison.
K_PER_SECTOR = 20
TARGET_KIND = "sector_relative"
# 13 -> 14 features: chmom added. (maxret blocked on Bowen's daily SEP.)
# chmom = mom6m_recent - mom6m_prior (momentum acceleration). GKX 2020
# ranks it #4 in feature importance. Orthogonality check showed correlations
# under 0.06 with all existing features.
INCLUDE_FEATURES = ("mom", "rev", "mvol", "ivol", "log_mktcap",
                    "bm", "ep", "dvol",
                    "roe", "roa", "de", "asset_growth", "accruals",
                    "chmom")
# Pre-computed: returns from Bowen's Block B/C; features extended by
# notebooks/personb/compute_chmom_maxret_features.py
RETURNS_FILE = (
    Path(__file__).resolve().parents[2] / "data" / "processed"
    / "returns_broad_sharadar_2002_2024.parquet"
)
FEATURES_FILE = (
    Path(__file__).resolve().parents[2] / "data" / "processed"
    / "features_broad_sharadar_with_chmom_maxret.parquet"
)


# --------------------------------------------------------------------------
# Recording wrapper: capture predictions during walk-forward
# --------------------------------------------------------------------------

class RecordingModel:
    """Wraps a CrossSectionalModel and records every prediction batch.

    The backtest engine calls fit/predict during walk-forward but only
    keeps portfolio-level outputs. To compute prediction-quality metrics
    (R^2, IC) we need the raw scores at every rebalance. This wrapper
    appends each `predict` output to ``self.predictions`` -- concat at
    the end to get a tidy (date, ticker) Series. Also prints per-iteration
    progress so a stuck loop is obvious.
    """

    def __init__(self, inner, *, log_every: int = 1, label: str = "") -> None:
        import time
        self.inner = inner
        self.predictions: list[pd.Series] = []
        self._iter = 0
        self._t_start = time.time()
        self._t_last = self._t_start
        self._log_every = log_every
        self._label = label

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RecordingModel":
        import time
        self._iter += 1
        t0 = time.time()
        self.inner.fit(X, y)
        dt = time.time() - t0
        if (self._iter % self._log_every) == 0:
            elapsed = time.time() - self._t_start
            print(f"  [{self._label}] fit {self._iter} done "
                  f"in {dt:.2f}s (n_train={len(X):,}, total elapsed {elapsed:.1f}s)",
                  flush=True)
        self._t_last = time.time()
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        out = self.inner.predict(X)
        if not isinstance(out.index, pd.MultiIndex):
            out = pd.Series(out.to_numpy(), index=X.index, name=out.name)
        self.predictions.append(out.copy())
        return out

    @property
    def predictions_panel(self) -> pd.Series:
        """All captured predictions concatenated into one Series."""
        if not self.predictions:
            return pd.Series(dtype=float, name="pred")
        s = pd.concat(self.predictions).sort_index()
        s.name = "pred"
        return s


# --------------------------------------------------------------------------
# Realised next-period returns aligned to (date, ticker) for metric calcs
# --------------------------------------------------------------------------

def build_realised_panel(returns_wide: pd.DataFrame) -> pd.Series:
    """Convert wide returns to a long Series indexed by (date, ticker).

    For each rebalance date t, the *target* is the return realised at
    t+1 (next month). So we shift the wide returns by -1 along the date
    axis BEFORE stacking. The result at index (t, ticker) is the actual
    next-month return for that stock, which is what predictions are
    forecasting.
    """
    shifted = returns_wide.shift(-1)  # next-period return
    stacked = shifted.stack(future_stack=True).rename("y_true")
    stacked.index = stacked.index.set_names(["date", "ticker"])
    return stacked


# --------------------------------------------------------------------------
# Per-window metric computation
# --------------------------------------------------------------------------

@dataclass
class ModelEval:
    name: str
    metrics: dict
    backtest_result: object  # BacktestResult
    predictions: pd.Series


def evaluate_model(
    name: str,
    model,
    returns_wide: pd.DataFrame,
    features: pd.DataFrame,
    realised: pd.Series,
) -> ModelEval:
    """Run a single model's walk-forward + capture metrics on two windows."""
    print(f"\n[{name}] starting walk-forward (train_window={TRAIN_WINDOW})...")
    # Log every 10 fits for Lasso/XGBoost (119 total), every 5 for NN so we
    # can see the failure if it deadlocks again.
    log_every = 5 if name == "NN" else 10
    recorder = RecordingModel(model, log_every=log_every, label=name)

    # Layer 3: constant-K-per-sector regime function. Every rebalance emits
    # the same `k_per_sector`, leaving leverage and quantile defaults
    # unset (the engine falls back to 1.0x and the static quantiles, which
    # are ignored anyway under k_per_sector mode).
    def regime_constant_k(date) -> RegimeParams:
        return {"k_per_sector": K_PER_SECTOR}

    # Fix 2 (audit 2026-05-23): derive the sector_map from the feature panel
    # rather than load_sector_map() directly. The feature panel applies a SIC
    # fallback in factors.get_sector() for delisted tickers that aren't in the
    # current S&P 500 GICS sheet; passing load_sector_map() alone makes the
    # engine bucket those tickers as "UNKNOWN" for Layer-3 selection (so they
    # all compete in one synthetic pool instead of in their real sector).
    sector_map = (
        features.reset_index()
        .groupby("ticker")["sector"]
        .first()
        .to_dict()
    )

    # Phase 23: PIT universe IMPLICIT in the features panel. Bowen's
    # Block C feature freezer only writes rows for tickers that were in
    # the top-2000-by-mcap PIT universe at each date. So eligible @ d =
    # set of tickers with a row at (d, *) in the features index.
    # No look-ahead (universe was computed PIT). No survivorship leak
    # (delisted names included until their last-price-date).
    feat_dates = features.index.get_level_values("date")
    feat_tickers = features.index.get_level_values("ticker")
    _universe_map = {
        d: set(feat_tickers[feat_dates == d].unique())
        for d in features.index.get_level_values("date").unique()
    }

    def universe_at(date) -> set[str]:
        return _universe_map.get(pd.Timestamp(date), set())

    res = run_walk_forward_backtest(
        returns=returns_wide,
        features=features,
        model=recorder,
        train_window=TRAIN_WINDOW,
        test_window=TEST_WINDOW,
        long_quantile=LONG_QUANTILE,
        short_quantile=SHORT_QUANTILE,
        transaction_cost_bps=TRANSACTION_COST_BPS,
        regime_fn=regime_constant_k,
        sector_map=sector_map,
        eligible_universe_fn=universe_at,
    )
    # Fix 3 (audit 2026-05-23): assert the engine that produced this result
    # is the v0.3.0+ block-gated-refit engine. If a future engine version
    # changes semantics, we want a loud failure here, not silent number drift.
    iv = res.metadata.get("interface_version", "0.0.0")
    if not iv.startswith("0.5."):
        raise RuntimeError(
            f"[{name}] backtest engine returned interface_version={iv!r}; "
            f"Phase 22 requires v0.5.x (apply_pit_to_training flag)."
        )
    print(
        f"[{name}] finished: {res.metadata['n_rebalances']} rebalances, "
        f"{len(recorder.predictions_panel):,} prediction rows  "
        f"(engine v{iv})"
    )

    preds = recorder.predictions_panel
    # Align predictions with realised next-period returns (same MultiIndex)
    aligned = pd.concat([preds, realised], axis=1, join="inner").dropna()
    y_pred = aligned["pred"]
    y_true = aligned["y_true"]

    metrics_by_window: dict[str, dict] = {}
    for window_name, mask in [
        ("full_oos", pd.Series(True, index=aligned.index)),
        ("test_only",
         (aligned.index.get_level_values("date") >= TEST_START)
         & (aligned.index.get_level_values("date") <= TEST_END)),
    ]:
        y_t = y_true[mask]
        y_p = y_pred[mask]
        n_dates = y_t.index.get_level_values("date").nunique() if len(y_t) else 0

        # Prediction-quality metrics
        ic = information_coefficient(y_t, y_p)

        # Portfolio metrics on the matching subset of res.portfolio_returns
        port_dates = res.portfolio_returns.index
        if window_name == "test_only":
            port_mask = (port_dates >= TEST_START) & (port_dates <= TEST_END)
            port_rets = res.portfolio_returns[port_mask]
            gross_rets = res.gross_returns[port_mask]
        else:
            port_rets = res.portfolio_returns
            gross_rets = res.gross_returns
        port_stats = summary_stats(port_rets)

        metrics_by_window[window_name] = {
            "model": name,
            "window": window_name,
            "n_dates": int(n_dates),
            "n_obs": int(len(y_t)),
            "oos_r2_vs_zero": float(oos_r2(y_t, y_p, benchmark="zero")),
            "oos_r2_vs_mean": float(oos_r2(y_t, y_p, benchmark="mean")),
            "ic_mean": ic["ic_mean"],
            "ic_std": ic["ic_std"],
            "ic_ir": ic["ic_ir"],
            "sharpe_net": port_stats["sharpe_ratio"],
            "ann_return_net": port_stats["annualised_return"],
            "ann_vol_net": port_stats["annualised_volatility"],
            "max_drawdown": port_stats["max_drawdown"],
            "calmar": port_stats["calmar_ratio"],
            "hit_rate": port_stats["hit_rate"],
            "sharpe_gross": float(summary_stats(gross_rets)["sharpe_ratio"]),
            "avg_turnover": float(res.metadata["avg_monthly_turnover"]),
        }

    return ModelEval(
        name=name,
        metrics=metrics_by_window,
        backtest_result=res,
        predictions=preds,
    )


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def plot_cumulative_returns(evals: list[ModelEval], out_path: Path) -> None:
    """Cumulative net return per model on the test window (2019-2024)."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for ev in evals:
        rets = ev.backtest_result.portfolio_returns
        rets = rets[(rets.index >= TEST_START) & (rets.index <= TEST_END)]
        cum = (1.0 + rets).cumprod() - 1.0
        ax.plot(cum.index, cum.values * 100, label=ev.name, lw=1.6)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_title("Cumulative net return on the 2019-2024 test window")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return (%)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_drawdowns(evals: list[ModelEval], out_path: Path) -> None:
    """Drawdown curves per model on the test window."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for ev in evals:
        rets = ev.backtest_result.portfolio_returns
        rets = rets[(rets.index >= TEST_START) & (rets.index <= TEST_END)]
        wealth = (1.0 + rets).cumprod()
        dd = (wealth / wealth.cummax() - 1.0) * 100
        ax.plot(dd.index, dd.values, label=ev.name, lw=1.6)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_title("Drawdown on the 2019-2024 test window")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data --------------------------------------------------------
    print("=" * 72)
    print("Phase 23e: FINAL HONEST CANONICAL (Q-filtered + retuned + k=20)")
    print("=" * 72)
    print(f"Reading returns panel: {RETURNS_FILE.name}")
    returns_wide = pd.read_parquet(RETURNS_FILE)
    print(f"  shape: {returns_wide.shape[0]} months x "
          f"{returns_wide.shape[1]} tickers")

    print(f"Reading pre-computed features panel: {FEATURES_FILE.name}")
    features = pd.read_parquet(FEATURES_FILE)
    print(f"  features shape (raw): {features.shape}")

    # Q-FILTER: drop bankrupt-ticker rows from features AND returns BEFORE
    # passing to the engine. Now the model is trained on, predicts on, and
    # trades only legitimately-tradable names.
    tickers = features.index.get_level_values("ticker")
    is_q = pd.Series([is_q_suffix_bankruptcy(t) for t in tickers],
                     index=features.index)
    features = features.loc[~is_q]
    print(f"  Q-filter dropped {int(is_q.sum()):,} bankrupt-ticker rows "
          f"({100*is_q.mean():.1f}%)")
    print(f"  features shape (Q-filtered): {features.shape}")

    # Also drop Q-tickers from the returns wide panel
    q_cols = [c for c in returns_wide.columns if is_q_suffix_bankruptcy(c)]
    returns_wide = returns_wide.drop(columns=q_cols)
    print(f"  returns: dropped {len(q_cols)} Q-ticker columns; "
          f"new shape {returns_wide.shape}")

    print(f"  retuned XGBoost params (from Phase 23d Q-filtered): {RETUNED_XGB_PARAMS}")
    print(f"  retuned Lasso params:                                {RETUNED_LASSO_PARAMS}")
    print(f"  retuned NN params:                                   {RETUNED_NN_PARAMS}")
    print(f"  k_per_sector = {K_PER_SECTOR}")

    # Keep the sector column when passing features to the engine: the
    # models need it for the Layer-2 sector-relative target demean step
    # inside fit(). Each model's _split_X drops sector before the
    # underlying estimator sees it, so it never leaks into the prediction.
    # 2. Run each model --------------------------------------------------
    # XGBoost uses the Phase-8 13-feature tuned defaults from models.py.
    # All three models get target_kind="sector_relative" -- Layer 2 of
    # the framework's 3-layer sector neutrality stack.
    model_factories = [
        ("Lasso",
         lambda: LassoModel(
             alphas=[RETUNED_LASSO_PARAMS.get("alpha", 1e-4)],
             cv=3, target_kind=TARGET_KIND,
         )),
        ("XGBoost",
         lambda: XGBoostModel(
             target_kind=TARGET_KIND,
             n_estimators=RETUNED_XGB_PARAMS.get("n_estimators", 200),
             max_depth=RETUNED_XGB_PARAMS.get("max_depth", 3),
             learning_rate=RETUNED_XGB_PARAMS.get("learning_rate", 0.0115),
             subsample=RETUNED_XGB_PARAMS.get("subsample", 0.717),
             colsample_bytree=RETUNED_XGB_PARAMS.get("colsample_bytree", 0.890),
             min_child_weight=RETUNED_XGB_PARAMS.get("min_child_weight", 11),
             reg_alpha=RETUNED_XGB_PARAMS.get("reg_alpha", 0.794),
             reg_lambda=RETUNED_XGB_PARAMS.get("reg_lambda", 2.305),
         )),
        ("NN",
         lambda: NNModel(
             hidden_dim=RETUNED_NN_PARAMS.get("hidden_dim", 32),
             n_layers=RETUNED_NN_PARAMS.get("n_layers", 3),
             dropout=RETUNED_NN_PARAMS.get("dropout", 0.2),
             max_epochs=30, patience=5,
             learning_rate=RETUNED_NN_PARAMS.get("lr", 1e-3),
             target_kind=TARGET_KIND,
         )),
    ]

    realised = build_realised_panel(returns_wide)
    evals: list[ModelEval] = []
    for name, factory in model_factories:
        ev = evaluate_model(
            name=name,
            model=factory(),
            returns_wide=returns_wide,
            features=features,
            realised=realised,
        )
        evals.append(ev)

    # 3. Assemble metrics table -----------------------------------------
    rows = []
    for ev in evals:
        for _, m in ev.metrics.items():
            rows.append(m)
    metrics_df = pd.DataFrame(rows)
    metrics_path = RESULTS_DIR / "metrics.parquet"
    metrics_df.to_parquet(metrics_path)
    print(f"\nWrote {metrics_path.relative_to(RESULTS_DIR.parent.parent)}")

    # 4. Print headline table --------------------------------------------
    print("\n" + "=" * 72)
    print("Headline metrics (test window 2019-2024)")
    print("=" * 72)
    test_subset = metrics_df[metrics_df["window"] == "test_only"].copy()
    headline_cols = [
        "model", "n_dates", "oos_r2_vs_zero", "oos_r2_vs_mean", "ic_mean",
        "ic_ir", "sharpe_net", "ann_return_net", "max_drawdown",
        "avg_turnover",
    ]
    # Format floats nicely
    fmt = test_subset[headline_cols].copy()
    for c in fmt.columns:
        if fmt[c].dtype.kind == "f":
            fmt[c] = fmt[c].map(lambda x: f"{x:+.4f}" if abs(x) < 1
                                else f"{x:+.3f}")
    print(fmt.to_string(index=False))

    # 5. Plots -----------------------------------------------------------
    plot_cumulative_returns(evals, RESULTS_DIR / "cumulative_returns.png")
    plot_drawdowns(evals, RESULTS_DIR / "drawdowns.png")
    print(f"\nWrote plots to {RESULTS_DIR.name}/")

    # 6. Persist objects for downstream use -----------------------------
    with open(RESULTS_DIR / "per_model_results.pkl", "wb") as f:
        pickle.dump({ev.name: ev.backtest_result for ev in evals}, f)

    pred_df = pd.concat(
        [ev.predictions.rename(ev.name) for ev in evals], axis=1
    )
    pred_df.to_parquet(RESULTS_DIR / "predictions.parquet")
    print(f"Wrote per_model_results.pkl and predictions.parquet")

    print(f"\nDone. See {RESULTS_DIR.name}/ for all artefacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
