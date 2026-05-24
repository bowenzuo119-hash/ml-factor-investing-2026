"""Phase 23d: re-tune XGBoost on Q-suffix-FILTERED training panel.

Phase 23a's tune was on the full broad panel which included bankrupt
Q-suffix tickers (LEHMQ, ENRNQ, SIVBQ, INTEQ, ...). The XGBoost we
got from 23a learned to exploit those patterns. When we apply the
Q-filter at trade time (Phase 23c), we use a model that's been
optimized for a slightly different distribution.

This phase re-runs Optuna with Q-tickers EXCLUDED from training, so
the model is properly tuned for the universe it'll actually trade.

Same Optuna search as 23a; just changes the input data.

Run with:
    .venv/bin/python -m notebooks.personb.23d_retune_xgb_qfiltered
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from src.metrics import oos_r2


PANEL_START = pd.Timestamp("2002-04-01")
PANEL_END = pd.Timestamp("2018-12-31")
TRAIN_END = pd.Timestamp("2016-12-31")
VAL_START = pd.Timestamp("2017-01-31")
VAL_END = pd.Timestamp("2018-12-31")

N_TRIALS = 60
TIMEOUT_SEC = 30 * 60

RESULTS_DIR = (
    Path(__file__).resolve().parents[2] / "results" / "23d_retune_xgb_qfiltered"
)
RETURNS_FILE = (
    Path(__file__).resolve().parents[2] / "data" / "processed"
    / "returns_broad_sharadar_2002_2024.parquet"
)
FEATURES_FILE = (
    Path(__file__).resolve().parents[2] / "data" / "processed"
    / "features_broad_sharadar_2002_2024.parquet"
)


def is_q_suffix_bankruptcy(t: str) -> bool:
    t = str(t).upper().strip()
    return len(t) >= 4 and t.endswith("Q")


def realised_next_period(returns_wide):
    shifted = returns_wide.shift(-1)
    stacked = shifted.stack(future_stack=True).rename("y_true")
    stacked.index = stacked.index.set_names(["date", "ticker"])
    return stacked


def make_train_val_qfiltered(features, returns_wide):
    realised = realised_next_period(returns_wide)
    X = features.drop(columns=["sector"]) if "sector" in features.columns else features

    # Q-filter at the (date, ticker) level
    tickers_lvl = X.index.get_level_values("ticker")
    is_q = pd.Series([is_q_suffix_bankruptcy(t) for t in tickers_lvl],
                     index=X.index)
    X = X.loc[~is_q]
    print(f"  Q-filter dropped {is_q.sum():,} of {len(is_q):,} rows "
          f"({100*is_q.mean():.1f}%)")

    joint = pd.concat([X, realised.rename("__y__")], axis=1,
                      join="inner").dropna(subset=["__y__"])
    y = joint["__y__"]
    Xa = joint.drop(columns=["__y__"])
    dates = Xa.index.get_level_values("date")
    train_mask = (dates >= PANEL_START) & (dates <= TRAIN_END)
    val_mask = (dates >= VAL_START) & (dates <= VAL_END)
    return Xa[train_mask], y[train_mask], Xa[val_mask], y[val_mask]


def xgb_objective(X_train, y_train, X_val, y_val):
    def _obj(trial):
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
        model = xgb.XGBRegressor(tree_method="hist", random_state=42, n_jobs=1, **params)
        model.fit(X_train.to_numpy(), y_train.to_numpy())
        y_pred = pd.Series(model.predict(X_val.to_numpy()), index=X_val.index)
        return oos_r2(y_val, y_pred, benchmark="zero")
    return _obj


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("Phase 23d: XGBoost Optuna retune on Q-FILTERED Sharadar panel")
    print("=" * 72)

    print("\n[1/3] Loading panels...")
    returns_wide = pd.read_parquet(RETURNS_FILE)
    features = pd.read_parquet(FEATURES_FILE)
    print(f"  returns: {returns_wide.shape}, features: {features.shape}")

    print("\n[2/3] Building Q-FILTERED train/val splits...")
    X_train, y_train, X_val, y_val = make_train_val_qfiltered(features, returns_wide)
    print(f"  X_train: {X_train.shape}  X_val: {X_val.shape}")

    print(f"\n[3/3] Optuna ({N_TRIALS} trials, timeout {TIMEOUT_SEC//60}min)...")
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                 study_name="xgboost_phase23d")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t0 = time.time()

    def cb(study, trial):
        if trial.number % 5 == 0 or trial.number == N_TRIALS - 1:
            elapsed = time.time() - t0
            best = study.best_value if study.best_trial else float("nan")
            val = trial.value if trial.value is not None else float("nan")
            print(f"  trial {trial.number:3d}: R^2={val:+.6f}, best={best:+.6f}, "
                  f"elapsed {elapsed:.0f}s", flush=True)

    study.optimize(xgb_objective(X_train, y_train, X_val, y_val),
                   n_trials=N_TRIALS, timeout=TIMEOUT_SEC,
                   callbacks=[cb], show_progress_bar=False)
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. Best R^2 = {study.best_value:+.6f}")
    print("Best params:")
    for k, v in study.best_params.items():
        v_str = f"{v:.5f}" if isinstance(v, float) else str(v)
        print(f"  {k:18s} = {v_str}")

    with open(RESULTS_DIR / "best_params.json", "w") as f:
        json.dump({
            "panel": "Sharadar broad top-2000 2002-2024 (Q-FILTERED, PIT)",
            "objective": "oos_r2_vs_zero on 2017-2018 validation",
            "train_rows": int(len(X_train)),
            "val_rows": int(len(X_val)),
            "best_value": float(study.best_value),
            "best_params": dict(study.best_params),
            "n_trials": len(study.trials),
            "elapsed_sec": float(elapsed),
        }, f, indent=2)
    print(f"Wrote {RESULTS_DIR / 'best_params.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
