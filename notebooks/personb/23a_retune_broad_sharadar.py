"""Phase 23a: Optuna retune of XGBoost + Lasso + NN on the broad Sharadar panel.

Inputs (committed by Bowen, Block B+C):
  * data/processed/returns_broad_sharadar_2002_2024.parquet  (276 x 5897)
  * data/processed/features_broad_sharadar_2002_2024.parquet (1.24M x 13 + sector)

Universe = top 2000 by market cap per date, PIT-filtered via TICKERS + DAILY.
Pre-applied by Bowen: every row in the features panel is PIT-eligible at its
date (the universe is implicit in the panel index). So no PIT filter needed here.

This rescues the model retune from the smaller-S&P-500 strict-PIT panel of
Phase 19. Wider universe ~2x bigger -> more cross-sectional dispersion ->
Optuna has more signal to lock onto.

Differences from Phase 19:
  * Different panel (5897 vs 941 tickers, 1.24M vs 56K train rows after PIT)
  * NN has torch.set_num_threads(1) to avoid the multithread hang we hit in P19
  * Fewer NN trials (20) with smaller models to stay within timeout

Run with:
    .venv/bin/python -m notebooks.personb.23a_retune_broad_sharadar
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
PANEL_END = pd.Timestamp("2018-12-31")  # tuning never sees 2019+
TRAIN_END = pd.Timestamp("2016-12-31")
VAL_START = pd.Timestamp("2017-01-31")
VAL_END = pd.Timestamp("2018-12-31")

XGB_N_TRIALS = 60
LASSO_N_TRIALS = 30
NN_N_TRIALS = 20
TIMEOUT_SEC_PER_MODEL = 30 * 60

RESULTS_DIR = (
    Path(__file__).resolve().parents[2] / "results" / "23a_retune_broad_sharadar"
)
RETURNS_FILE = (
    Path(__file__).resolve().parents[2]
    / "data" / "processed" / "returns_broad_sharadar_2002_2024.parquet"
)
FEATURES_FILE = (
    Path(__file__).resolve().parents[2]
    / "data" / "processed" / "features_broad_sharadar_2002_2024.parquet"
)


def realised_next_period(returns_wide: pd.DataFrame) -> pd.Series:
    shifted = returns_wide.shift(-1)
    stacked = shifted.stack(future_stack=True).rename("y_true")
    stacked.index = stacked.index.set_names(["date", "ticker"])
    return stacked


def make_train_val(features: pd.DataFrame, returns_wide: pd.DataFrame):
    """Build (X_train, y_train, X_val, y_val) on the broad Sharadar panel.

    Universe is implicit in features (PIT-eligible rows only). Just slice by date.
    """
    realised = realised_next_period(returns_wide)
    X = features.drop(columns=["sector"]) if "sector" in features.columns else features
    joint = pd.concat([X, realised.rename("__y__")], axis=1,
                      join="inner").dropna(subset=["__y__"])
    y = joint["__y__"]
    Xa = joint.drop(columns=["__y__"])

    dates = Xa.index.get_level_values("date")
    train_mask = (dates >= PANEL_START) & (dates <= TRAIN_END)
    val_mask = (dates >= VAL_START) & (dates <= VAL_END)
    return Xa[train_mask], y[train_mask], Xa[val_mask], y[val_mask]


# --------------------- Objectives ---------------------

def xgb_objective(X_train, y_train, X_val, y_val):
    def _obj(trial: optuna.Trial) -> float:
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
            tree_method="hist", random_state=42, n_jobs=1, **params,
        )
        model.fit(X_train.to_numpy(), y_train.to_numpy())
        y_pred = pd.Series(model.predict(X_val.to_numpy()), index=X_val.index)
        return oos_r2(y_val, y_pred, benchmark="zero")
    return _obj


def lasso_objective(X_train, y_train, X_val, y_val):
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="median").fit(X_train.to_numpy())
    Xtr_imp = imputer.transform(X_train.to_numpy())
    Xva_imp = imputer.transform(X_val.to_numpy())
    scaler = StandardScaler().fit(Xtr_imp)
    Xtr = scaler.transform(Xtr_imp)
    Xva = scaler.transform(Xva_imp)
    ytr = y_train.to_numpy()

    def _obj(trial: optuna.Trial) -> float:
        alpha = trial.suggest_float("alpha", 1e-6, 1e-1, log=True)
        max_iter = trial.suggest_int("max_iter", 1000, 10000, step=1000)
        model = Lasso(alpha=alpha, max_iter=max_iter, random_state=42)
        model.fit(Xtr, ytr)
        y_pred = pd.Series(model.predict(Xva), index=X_val.index)
        return oos_r2(y_val, y_pred, benchmark="zero")
    return _obj


def nn_objective(X_train, y_train, X_val, y_val):
    """PyTorch MLP with torch.set_num_threads(1) to avoid Phase 19's hang."""
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    # Phase 19 hung at NN trial 0 — likely a thread-pool deadlock interacting
    # with Optuna + the broader panel. Force single-thread + smaller model.
    torch.set_num_threads(1)

    imputer = SimpleImputer(strategy="median").fit(X_train.to_numpy())
    Xtr_imp = imputer.transform(X_train.to_numpy())
    Xva_imp = imputer.transform(X_val.to_numpy())
    scaler = StandardScaler().fit(Xtr_imp)
    Xtr = torch.tensor(scaler.transform(Xtr_imp), dtype=torch.float32)
    Xva = torch.tensor(scaler.transform(Xva_imp), dtype=torch.float32)
    ytr = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
    n_features = Xtr.shape[1]

    def _obj(trial: optuna.Trial) -> float:
        torch.manual_seed(42)
        hidden_dim = trial.suggest_int("hidden_dim", 16, 64, step=16)
        n_layers = trial.suggest_int("n_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        max_epochs = 20
        batch_size = 4096

        layers = []
        in_dim = n_features
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        layers += [nn.Linear(in_dim, 1)]
        model = nn.Sequential(*layers)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()

        n = Xtr.shape[0]
        for _ in range(max_epochs):
            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                yhat = model(Xtr[idx]).squeeze(-1)
                loss = loss_fn(yhat, ytr[idx])
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            yhat_val = model(Xva).squeeze(-1).numpy()
        y_pred = pd.Series(yhat_val, index=X_val.index)
        return oos_r2(y_val, y_pred, benchmark="zero")
    return _obj


def retune(name, n_trials, objective, X_train, y_train, X_val, y_val, log_every=5):
    print(f"\n{'=' * 72}")
    print(f"Retune {name} (n_trials={n_trials}, timeout {TIMEOUT_SEC_PER_MODEL//60}min)")
    print('=' * 72)
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        direction="maximize", sampler=sampler, study_name=f"{name}_phase23a",
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t0 = time.time()

    def progress_cb(study, trial):
        elapsed = time.time() - t0
        best = study.best_value if study.best_trial else float("nan")
        if trial.number % log_every == 0 or trial.number == n_trials - 1:
            val = trial.value if trial.value is not None else float("nan")
            print(f"  trial {trial.number:3d}: R^2={val:+.6f}, "
                  f"best={best:+.6f}, elapsed {elapsed:.0f}s",
                  flush=True)

    obj_fn = objective(X_train, y_train, X_val, y_val)
    study.optimize(obj_fn, n_trials=n_trials, timeout=TIMEOUT_SEC_PER_MODEL,
                   callbacks=[progress_cb], show_progress_bar=False)
    elapsed = time.time() - t0
    print(f"\n  {name} done in {elapsed:.0f}s ({len(study.trials)} trials)")
    print(f"  best R^2 = {study.best_value:+.6f}")
    print(f"  best params:")
    for k, v in study.best_params.items():
        v_str = f"{v:.5f}" if isinstance(v, float) else str(v)
        print(f"    {k:18s} = {v_str}")
    return {
        "model": name,
        "best_value": float(study.best_value),
        "best_params": dict(study.best_params),
        "n_trials": len(study.trials),
        "elapsed_sec": float(elapsed),
    }


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("Phase 23a: Optuna retune on BROAD SHARADAR panel (top-2000)")
    print("=" * 72)

    print("\n[1/3] Loading broad panel + features (already PIT-filtered)...")
    returns_wide = pd.read_parquet(RETURNS_FILE)
    features = pd.read_parquet(FEATURES_FILE)
    print(f"  returns: {returns_wide.shape} (months x tickers)")
    print(f"  features: {features.shape}")

    print("\n[2/3] Slicing train (PIT, <=2016-12) / val (PIT, 2017-01..2018-12)...")
    X_train, y_train, X_val, y_val = make_train_val(features, returns_wide)
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")

    print("\n[3/3] Three Optuna studies...")
    summary: dict[str, dict] = {}
    summary["xgboost"] = retune("XGBoost", XGB_N_TRIALS, xgb_objective,
                                 X_train, y_train, X_val, y_val, log_every=5)
    summary["lasso"] = retune("Lasso", LASSO_N_TRIALS, lasso_objective,
                               X_train, y_train, X_val, y_val, log_every=5)
    summary["nn"] = retune("NN", NN_N_TRIALS, nn_objective,
                           X_train, y_train, X_val, y_val, log_every=3)

    with open(RESULTS_DIR / "best_params.json", "w") as f:
        json.dump({
            "panel": "Sharadar broad top-2000 2002-2024 (PIT)",
            "objective": "oos_r2_vs_zero on 2017-2018 validation",
            "train_rows": int(len(X_train)),
            "val_rows": int(len(X_val)),
            "by_model": summary,
        }, f, indent=2)
    print(f"\nWrote {RESULTS_DIR / 'best_params.json'}")

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for m in ("xgboost", "lasso", "nn"):
        s = summary[m]
        print(f"  {m:8s}  best R^2 = {s['best_value']:+.6f}  "
              f"(n_trials={s['n_trials']}, elapsed {s['elapsed_sec']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
