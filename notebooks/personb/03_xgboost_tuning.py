"""Phase 3: tune XGBoost hyperparameters on the 2016-2018 validation window.

Uses Optuna's TPE sampler over the canonical GKX-style hyperparameter grid.
Objective: OOS R^2 vs zero on the 2016-2018 validation window, using a
single train (2005-2015) -> predict (2016-2018) split per trial -- the
walk-forward refit happens only at FINAL-evaluation time, not inside the
tuning loop (each tuning trial would be 10+ minutes otherwise).

The chosen hyperparameters are then pinned as the new XGBoostModel
defaults (manually copied into src/models.py after this script reports
its best trial), and Phase 1.5's evaluation gets re-run with them.

Why R^2 not Sharpe as the objective: the framework section 8.4 calls
R^2 the model-quality metric; Sharpe is the strategy's. R^2 is also
much smoother than Sharpe over a small validation window, so the
optimisation surface Optuna sees is less noisy.

Run with:
    .venv/bin/python -m notebooks.personb.03_xgboost_tuning
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from src.factors import build_feature_panel
from src.metrics import oos_r2


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

PANEL_START = "2005-01-01"
PANEL_END = "2018-12-31"  # tuning never sees 2019+
TRAIN_END = pd.Timestamp("2015-12-31")
VAL_START = pd.Timestamp("2016-01-31")
VAL_END = pd.Timestamp("2018-12-31")

N_TRIALS = 60
TIMEOUT_SEC = 30 * 60  # 30 min hard cap

INCLUDE_FEATURES = ("mom", "rev", "mvol", "ivol", "log_mktcap",
                    "bm", "ep", "dvol",
                    "roe", "roa", "de", "asset_growth", "accruals")

RESULTS_DIR = (
    Path(__file__).resolve().parents[2] / "results" / "03_xgboost_tuning"
)
PANEL_FILE = (
    Path(__file__).resolve().parents[2]
    / "data" / "processed" / "returns_spliced_2005_2024.parquet"
)


# --------------------------------------------------------------------------
# Build the once-only training/validation panels
# --------------------------------------------------------------------------

def realised_next_period(returns_wide: pd.DataFrame) -> pd.Series:
    """Long-format (date, ticker) -> realised next-month return."""
    shifted = returns_wide.shift(-1)
    stacked = shifted.stack(future_stack=True).rename("y_true")
    stacked.index = stacked.index.set_names(["date", "ticker"])
    return stacked


def make_train_val(features: pd.DataFrame, returns_wide: pd.DataFrame
                   ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Slice the feature panel + realised returns into (X_train, y_train, X_val, y_val).

    Single split (not walk-forward) so each Optuna trial is fast.
    The walk-forward refit happens later, at FINAL-evaluation time.
    """
    realised = realised_next_period(returns_wide)
    # Drop the 'sector' column -- not a predictive feature.
    X = features.drop(columns=["sector"]) if "sector" in features.columns else features
    # Align X and y on intersection
    joint = pd.concat([X, realised.rename("__y__")], axis=1, join="inner").dropna(
        subset=["__y__"]
    )
    y = joint["__y__"]
    Xa = joint.drop(columns=["__y__"])

    dates = Xa.index.get_level_values("date")
    train_mask = dates <= TRAIN_END
    val_mask = (dates >= VAL_START) & (dates <= VAL_END)

    return Xa[train_mask], y[train_mask], Xa[val_mask], y[val_mask]


# --------------------------------------------------------------------------
# Optuna objective
# --------------------------------------------------------------------------

def make_objective(X_train, y_train, X_val, y_val):
    def objective(trial: optuna.Trial) -> float:
        import xgboost as xgb

        # Suggest hyperparameters
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        }
        model = xgb.XGBRegressor(
            tree_method="hist",
            random_state=42,
            n_jobs=1,
            **params,
        )

        # XGBoost handles NaNs natively -- no imputation needed.
        model.fit(X_train.to_numpy(), y_train.to_numpy())
        y_pred = pd.Series(model.predict(X_val.to_numpy()), index=X_val.index)

        # OOS R^2 vs zero on the validation slice.
        return oos_r2(y_val, y_pred, benchmark="zero")

    return objective


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Phase 3: Optuna tuning of XGBoost on 2016-2018 validation window")
    print("=" * 72)
    print(f"  features: {INCLUDE_FEATURES}")
    print(f"  trials:   {N_TRIALS}")
    print(f"  timeout:  {TIMEOUT_SEC // 60} min")
    print(f"  objective: OOS R^2 vs zero (higher is better)")

    print(f"\n[1/4] Loading panels...")
    returns_wide = pd.read_parquet(PANEL_FILE)
    features = build_feature_panel(
        start=PANEL_START, end=PANEL_END, include=INCLUDE_FEATURES,
        sector_rank=True,
    )
    print(f"  features shape: {features.shape}, returns shape: {returns_wide.shape}")

    print(f"\n[2/4] Slicing train (<=2015-12) / val (2016-01..2018-12)...")
    X_train, y_train, X_val, y_val = make_train_val(features, returns_wide)
    print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}")

    print(f"\n[3/4] Running Optuna study (TPE, seed=42)...")
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="xgboost_phase3",
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    t0 = time.time()
    progress: list[tuple[int, float, float]] = []  # (trial, value, elapsed)

    def progress_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        elapsed = time.time() - t0
        best = study.best_value
        progress.append((trial.number, trial.value, elapsed))
        if trial.number % 5 == 0 or trial.number == N_TRIALS - 1:
            print(f"  trial {trial.number:3d}: R^2={trial.value:+.5f}, "
                  f"best so far={best:+.5f}, elapsed {elapsed:.0f}s",
                  flush=True)

    objective = make_objective(X_train, y_train, X_val, y_val)
    study.optimize(
        objective,
        n_trials=N_TRIALS,
        timeout=TIMEOUT_SEC,
        callbacks=[progress_cb],
        show_progress_bar=False,
    )

    elapsed = time.time() - t0
    print(f"\n[4/4] Tuning done in {elapsed:.0f}s ({len(study.trials)} trials)")
    print(f"  best R^2 = {study.best_value:+.5f}")
    print(f"  best params:")
    for k, v in study.best_params.items():
        v_str = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"    {k:18s} = {v_str}")

    # Persist outputs
    best_params = dict(study.best_params)
    best_summary = {
        "best_value": float(study.best_value),
        "best_params": best_params,
        "n_trials": len(study.trials),
        "elapsed_sec": float(elapsed),
    }
    with open(RESULTS_DIR / "best_params.json", "w") as f:
        json.dump(best_summary, f, indent=2)
    print(f"\nWrote {RESULTS_DIR / 'best_params.json'}")

    # Trial history for the report
    trials_df = pd.DataFrame([
        {"trial": t.number, "value": t.value, **t.params}
        for t in study.trials if t.value is not None
    ])
    trials_df.to_parquet(RESULTS_DIR / "trials.parquet")
    print(f"Wrote {RESULTS_DIR / 'trials.parquet'}")

    print("\nNext step: copy these hyperparameters into XGBoostModel's "
          "__init__ defaults in src/models.py, then re-run "
          "01b_with_value_factors.py to get tuned-XGBoost numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
