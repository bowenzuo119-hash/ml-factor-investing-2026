"""Phase 22: HONEST canonical -- retuned XGBoost + relaxed-PIT trading.

This is the candidate honest canonical that addresses:
  * the survivorship leak fixed by engine v0.4.0 (strict PIT)
  * the universe narrowness of strict PIT (~500 names) by using
    relaxed PIT instead (~1000 names by 2024 -- cumulative ever-S&P
    members, no future-joiner look-ahead)
  * the hyperparameter overfit by using Phase 19's retuned XGBoost
    params (60 Optuna trials on the relaxed-PIT validation window)

Engine flags:
  * eligible_universe_fn = load_sp500_ever_membership  (relaxed PIT)
  * apply_pit_to_training = True (default; train + trade both on
    relaxed-PIT universe for internal consistency)
  * sector_map from features['sector'] (Fix 2: real GICS/SIC sectors)
  * regime_fn: constant k=5

XGBoost hyperparameters (from results/19_retune_all_relaxed_pit/best_params.json):
  n_estimators=50, max_depth=5, learning_rate=0.00886, subsample=0.948,
  colsample_bytree=0.65, min_child_weight=12, reg_alpha=0.643,
  reg_lambda=5.426 (much more regularized than Phase 3's tune)

NN and Lasso use their existing defaults for now (Phase 19's NN retune
hung; Lasso converged to R^2=+0.0032 but exact alpha not captured).

Output to ``results/22_canonical_relaxed_pit_retuned/``.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import run_walk_forward_backtest, RegimeParams
from src.data_loader import load_sp500_ever_membership
from src.factors import build_feature_panel, load_sector_map
from src.metrics import (
    information_coefficient,
    oos_r2,
    summary_stats,
)
from src.models import LassoModel, NNModel, XGBoostModel


# Retuned XGBoost hyperparameters from Phase 19 (Optuna on relaxed-PIT
# validation window). Much more regularized than Phase 3's tune --
# min_child_weight 12 (was 1), reg_lambda 5.4 (was 1), n_estimators 50
# (was 200). Best val R^2 = +0.002699.
RETUNED_XGB_PARAMS = {
    "n_estimators": 50,
    "max_depth": 5,
    "learning_rate": 0.00886,
    "subsample": 0.94837,
    "colsample_bytree": 0.65052,
    "min_child_weight": 12,
    "reg_alpha": 0.64281,
    "reg_lambda": 5.42577,
}


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
    / "results" / "22_canonical_relaxed_pit_retuned"
)
# Layer 3: k longs and k shorts within EACH GICS sector.
# k=5 is the empirical optimum from Phase 13's sensitivity sweep.
K_PER_SECTOR = 5
# Layer 2: target = next-period return minus (date, sector) mean.
TARGET_KIND = "sector_relative"
INCLUDE_FEATURES = ("mom", "rev", "mvol", "ivol", "log_mktcap",
                    "bm", "ep", "dvol",
                    "roe", "roa", "de", "asset_growth", "accruals")
PANEL_FILE = Path(__file__).resolve().parents[2] / "data" / "processed" / "returns_spliced_2002_2024.parquet"


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

    # Phase 22: RELAXED PIT. At each rebalance, eligible universe =
    # all tickers that were S&P 500 members at any point UP TO that date
    # (cumulative watchlist, no future joiners). Roughly 2x larger than
    # strict PIT (~1000 names by 2024 vs ~500), restores former-member
    # tradability without re-introducing look-ahead bias.
    def universe_at(date) -> set[str]:
        return set(load_sp500_ever_membership(asof=pd.Timestamp(date)))

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
    print("Phase 22: HONEST canonical (relaxed PIT + retuned XGBoost)")
    print("=" * 72)
    print(f"Reading returns panel: {PANEL_FILE.name}")
    returns_wide = pd.read_parquet(PANEL_FILE)
    print(f"  shape: {returns_wide.shape[0]} months x "
          f"{returns_wide.shape[1]} tickers")

    print(f"Building feature panel ({len(INCLUDE_FEATURES)} features)...")
    features = build_feature_panel(
        start=START, end=END,
        include=INCLUDE_FEATURES,
        sector_rank=True,
    )
    print(f"  features shape: {features.shape}")

    # Keep the sector column when passing features to the engine: the
    # models need it for the Layer-2 sector-relative target demean step
    # inside fit(). Each model's _split_X drops sector before the
    # underlying estimator sees it, so it never leaks into the prediction.
    features_for_engine = features
    sector_col = features["sector"]

    realised = build_realised_panel(returns_wide)

    # 2. Run each model --------------------------------------------------
    # XGBoost uses the Phase-8 13-feature tuned defaults from models.py.
    # All three models get target_kind="sector_relative" -- Layer 2 of
    # the framework's 3-layer sector neutrality stack.
    model_factories = [
        ("Lasso",
         lambda: LassoModel(alphas=20, cv=5, target_kind=TARGET_KIND)),
        ("XGBoost",
         # Phase 22: use the relaxed-PIT-retuned hyperparameters from
         # Phase 19's Optuna study. Much more regularised than the
         # Phase 3 tune (min_child_weight 12 vs 1, reg_lambda 5.4 vs 1).
         lambda: XGBoostModel(
             target_kind=TARGET_KIND,
             n_estimators=RETUNED_XGB_PARAMS["n_estimators"],
             max_depth=RETUNED_XGB_PARAMS["max_depth"],
             learning_rate=RETUNED_XGB_PARAMS["learning_rate"],
             subsample=RETUNED_XGB_PARAMS["subsample"],
             colsample_bytree=RETUNED_XGB_PARAMS["colsample_bytree"],
             min_child_weight=RETUNED_XGB_PARAMS["min_child_weight"],
             reg_alpha=RETUNED_XGB_PARAMS["reg_alpha"],
             reg_lambda=RETUNED_XGB_PARAMS["reg_lambda"],
         )),
        ("NN",
         lambda: NNModel(hidden_dim=32, n_layers=3, dropout=0.2,
                         max_epochs=30, patience=5,
                         target_kind=TARGET_KIND)),
    ]

    evals: list[ModelEval] = []
    for name, factory in model_factories:
        ev = evaluate_model(
            name=name,
            model=factory(),
            returns_wide=returns_wide,
            features=features_for_engine,
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
