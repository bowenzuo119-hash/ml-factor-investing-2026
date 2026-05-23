"""Phase 18: re-tune XGBoost hyperparameters on the PIT-filtered panel.

Background: Phase 15 (with engine-v0.4.0 PIT filter) collapsed XGBoost
Sharpe from +1.50 to -0.31, and Phase 17 (train-on-full / trade-on-PIT)
confirmed the collapse is from the trading-universe restriction, not
from training-data shrinkage. But the model's IC also dropped from
+0.018 -> +0.004 under PIT — much sharper than expected for a 33%
training-data reduction. The hypothesis: the previously-tuned
hyperparameters (depth=3, n_estimators=200, lr=0.0115) were Optuna-tuned
on the 941-ticker no-PIT panel in Phase 3, and may be miscalibrated for
the smaller PIT-filtered training distribution.

This phase reruns Phase 3's Optuna study but with:
  * PIT-filtered training panel (only S&P 500 members at each
    training feature's date)
  * 2002-2024 canonical panel
  * Same hyperparameter grid (n_estimators, max_depth, lr, subsample,
    colsample_bytree, min_child_weight, reg_alpha, reg_lambda)
  * 60 trials, 30 min timeout, TPE sampler seed=42 for reproducibility
  * Objective: OOS R^2 vs zero on the 2017-2018 validation window

Walk-forward is NOT used inside the tuning loop (each trial = single
train -> val split, ~20s/trial). Walk-forward refit happens at FINAL
evaluation time in Phase 21.

Output: results/18_retune_xgboost_pit/best_params.json
        results/18_retune_xgboost_pit/trials.parquet

Run with:
    .venv/bin/python -m notebooks.personb.18_retune_xgboost_pit
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from src.data_loader import load_sp500_membership
from src.factors import build_feature_panel
from src.metrics import oos_r2


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

PANEL_START = "2002-04-01"
PANEL_END = "2018-12-31"  # tuning never sees 2019+
TRAIN_END = pd.Timestamp("2016-12-31")
VAL_START = pd.Timestamp("2017-01-31")
VAL_END = pd.Timestamp("2018-12-31")

N_TRIALS = 60
TIMEOUT_SEC = 30 * 60

INCLUDE_FEATURES = ("mom", "rev", "mvol", "ivol", "log_mktcap",
                    "bm", "ep", "dvol",
                    "roe", "roa", "de", "asset_growth", "accruals")

RESULTS_DIR = (
    Path(__file__).resolve().parents[2] / "results" / "18_retune_xgboost_pit"
)
PANEL_FILE = (
    Path(__file__).resolve().parents[2]
    / "data" / "processed" / "returns_spliced_2002_2024.parquet"
)


# --------------------------------------------------------------------------
# Build PIT-filtered training panel
# --------------------------------------------------------------------------

def realised_next_period(returns_wide: pd.DataFrame) -> pd.Series:
    shifted = returns_wide.shift(-1)
    stacked = shifted.stack(future_stack=True).rename("y_true")
    stacked.index = stacked.index.set_names(["date", "ticker"])
    return stacked


def make_pit_train_val(features: pd.DataFrame, returns_wide: pd.DataFrame
                       ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Slice into (X_train, y_train, X_val, y_val), filtered to PIT membership.

    A (date, ticker) row is kept only if `ticker` was an S&P 500 member
    at `date` — exactly what the v0.4.0 engine would feed the model
    during a Phase 15 walk-forward.
    """
    realised = realised_next_period(returns_wide)
    X = features.drop(columns=["sector"]) if "sector" in features.columns else features
    joint = pd.concat([X, realised.rename("__y__")], axis=1, join="inner").dropna(
        subset=["__y__"]
    )
    y = joint["__y__"]
    Xa = joint.drop(columns=["__y__"])

    dates = Xa.index.get_level_values("date")
    tickers = Xa.index.get_level_values("ticker")

    # PIT filter: build {date -> set(members at that date)} once.
    unique_dates = sorted(set(dates))
    pit_map: dict[pd.Timestamp, set[str]] = {
        d: set(load_sp500_membership(asof=d)) for d in unique_dates
    }
    keep = np.array([
        t in pit_map[d] for d, t in zip(dates, tickers)
    ])

    train_mask = (dates <= TRAIN_END) & keep
    val_mask = (dates >= VAL_START) & (dates <= VAL_END) & keep

    return Xa[train_mask], y[train_mask], Xa[val_mask], y[val_mask]


# --------------------------------------------------------------------------
# Optuna objective
# --------------------------------------------------------------------------

def make_objective(X_train, y_train, X_val, y_val):
    def objective(trial: optuna.Trial) -> float:
        import xgboost as xgb

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 800, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        }
        model = xgb.XGBRegressor(
            tree_method="hist",
            random_state=42,
            n_jobs=1,
            **params,
        )
        model.fit(X_train.to_numpy(), y_train.to_numpy())
        y_pred = pd.Series(model.predict(X_val.to_numpy()), index=X_val.index)
        return oos_r2(y_val, y_pred, benchmark="zero")
    return objective


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Phase 18: Optuna XGBoost retune on PIT-filtered 2017-2018 val window")
    print("=" * 72)
    print(f"  features: {len(INCLUDE_FEATURES)} ({INCLUDE_FEATURES})")
    print(f"  trials:   {N_TRIALS}")
    print(f"  timeout:  {TIMEOUT_SEC // 60} min")
    print(f"  objective: OOS R^2 vs zero (higher is better)")
    print(f"  search:   wider than Phase 3 (depth 2-7, lr 0.005-0.3, n_est 50-800)")

    print("\n[1/4] Loading panels...")
    returns_wide = pd.read_parquet(PANEL_FILE)
    features = build_feature_panel(
        start=PANEL_START, end=PANEL_END, include=INCLUDE_FEATURES,
        sector_rank=True,
    )
    print(f"  features shape: {features.shape}, returns shape: {returns_wide.shape}")

    print("\n[2/4] Slicing train (PIT, <= 2016-12) / val (PIT, 2017-01..2018-12)...")
    X_train, y_train, X_val, y_val = make_pit_train_val(features, returns_wide)
    print(f"  X_train: {X_train.shape} (PIT-filtered)")
    print(f"  X_val:   {X_val.shape} (PIT-filtered)")

    print("\n[3/4] Running Optuna study (TPE, seed=42)...")
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="xgboost_phase18",
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    t0 = time.time()

    def progress_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        elapsed = time.time() - t0
        best = study.best_value
        if trial.number % 5 == 0 or trial.number == N_TRIALS - 1:
            print(f"  trial {trial.number:3d}: R^2={trial.value:+.6f}, "
                  f"best so far={best:+.6f}, elapsed {elapsed:.0f}s",
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
    print(f"  best R^2 = {study.best_value:+.6f}")
    print(f"  best params:")
    for k, v in study.best_params.items():
        v_str = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"    {k:18s} = {v_str}")

    best_summary = {
        "best_value": float(study.best_value),
        "best_params": dict(study.best_params),
        "n_trials": len(study.trials),
        "elapsed_sec": float(elapsed),
        "panel": "2002-2024 PIT-filtered",
        "objective": "oos_r2_vs_zero on 2017-2018 validation",
    }
    with open(RESULTS_DIR / "best_params.json", "w") as f:
        json.dump(best_summary, f, indent=2)
    print(f"\nWrote {RESULTS_DIR / 'best_params.json'}")

    trials_df = pd.DataFrame([
        {"trial": t.number, "value": t.value, **t.params}
        for t in study.trials if t.value is not None
    ])
    trials_df.to_parquet(RESULTS_DIR / "trials.parquet")
    print(f"Wrote {RESULTS_DIR / 'trials.parquet'}")

    print("\nNext: copy these hyperparameters into Phase 21's driver "
          "(or as XGBoostModel defaults) and run the full PIT canonical.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
