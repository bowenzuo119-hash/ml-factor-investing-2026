"""Phase 4: pairwise Diebold-Mariano model comparison.

Framework section 8.4 prescribes the adapted DM test for declaring whether
two model forecasts have significantly different squared-error performance.
We run it on the predictions produced by the canonical 8-feature backtest
(Phase 3c) -- one DM test per pair of models, then a 3x3 significance
table for the final report.

The implementation lives in `src.metrics.diebold_mariano`: per-rebalance
average squared-error differential, Newey-West HAC variance, two-sided
p-value from standard normal. See its docstring for the full spec.

Run with:
    .venv/bin/python -m notebooks.personb.04_model_comparison
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import diebold_mariano


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

# Use Phase 3c predictions (8 features, tuned XGBoost). If Phase 3c hasn't
# been run yet, fall back to Phase 3b (7 features) so this notebook still
# runs end-to-end during development.
SOURCE_PHASE_DIR = (
    Path(__file__).resolve().parents[2] / "results" / "03c_tuned_xgboost_8features"
)
FALLBACK_PHASE_DIR = (
    Path(__file__).resolve().parents[2] / "results" / "03b_tuned_xgboost"
)
RESULTS_DIR = (
    Path(__file__).resolve().parents[2] / "results" / "04_model_comparison"
)
PANEL_FILE = (
    Path(__file__).resolve().parents[2] / "data" / "processed"
    / "returns_spliced_2005_2024.parquet"
)

# Test window per Framework section 7.2
TEST_START = pd.Timestamp("2019-01-01")
TEST_END = pd.Timestamp("2024-12-31")

# Newey-West lag for HAC variance. 12 = annual seasonality buffer on
# monthly data; standard choice in the GKX paper.
NEWEY_WEST_LAGS = 12

MODELS = ["Lasso", "XGBoost", "NN"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def realised_next_period(returns_wide: pd.DataFrame) -> pd.Series:
    """Long-format (date, ticker) -> realised next-month return."""
    shifted = returns_wide.shift(-1)
    stacked = shifted.stack(future_stack=True).rename("y_true")
    stacked.index = stacked.index.set_names(["date", "ticker"])
    return stacked


def pick_source() -> Path:
    if SOURCE_PHASE_DIR.exists() and (SOURCE_PHASE_DIR / "predictions.parquet").exists():
        return SOURCE_PHASE_DIR
    if FALLBACK_PHASE_DIR.exists() and (FALLBACK_PHASE_DIR / "predictions.parquet").exists():
        print(f"  [warn] {SOURCE_PHASE_DIR.name} not found, falling back to "
              f"{FALLBACK_PHASE_DIR.name}")
        return FALLBACK_PHASE_DIR
    raise FileNotFoundError(
        "Neither Phase 3c nor Phase 3b predictions exist. "
        "Run a tuned-XGBoost backtest first."
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Phase 4: Diebold-Mariano model comparison")
    print("=" * 72)

    src = pick_source()
    print(f"  source: {src.relative_to(Path(__file__).resolve().parents[2])}")
    print(f"  test window: {TEST_START.date()} -> {TEST_END.date()}")
    print(f"  Newey-West lags: {NEWEY_WEST_LAGS}")

    # Load predictions and align with realised next-month returns
    preds = pd.read_parquet(src / "predictions.parquet")
    # predictions.parquet has columns = model names; index = (date, ticker)
    print(f"\n  predictions shape: {preds.shape}, columns: {list(preds.columns)}")

    returns_wide = pd.read_parquet(PANEL_FILE)
    realised = realised_next_period(returns_wide)

    # Restrict to the test window
    test_mask_pred = (
        (preds.index.get_level_values("date") >= TEST_START)
        & (preds.index.get_level_values("date") <= TEST_END)
    )
    preds_test = preds[test_mask_pred]
    print(f"  test-window predictions: {len(preds_test):,} rows, "
          f"{preds_test.index.get_level_values('date').nunique()} rebalance dates")

    # --- Per-model summary first ---
    print("\nPer-model squared-error MSE on the test window:")
    per_model_mse: dict[str, float] = {}
    for m in MODELS:
        if m not in preds_test.columns:
            continue
        joint = pd.concat(
            [preds_test[m].rename("pred"), realised.rename("y")],
            axis=1, join="inner",
        ).dropna()
        mse = float(((joint["y"] - joint["pred"]) ** 2).mean())
        per_model_mse[m] = mse
        print(f"  {m:8s}  MSE = {mse:.5f}  (n={len(joint):,})")

    # --- Pairwise Diebold-Mariano ---
    pairs = [(a, b) for i, a in enumerate(MODELS)
             for b in MODELS[i + 1:]]
    print(f"\nPairwise Diebold-Mariano (HAC lags = {NEWEY_WEST_LAGS}, "
          f"two-sided p-values):")

    rows = []
    for a, b in pairs:
        if a not in preds_test.columns or b not in preds_test.columns:
            continue
        res = diebold_mariano(
            preds_test[a], preds_test[b], realised,
            newey_west_lags=NEWEY_WEST_LAGS,
        )
        # Interpretation: negative dm_stat means A has SMALLER MSE
        # (model A is better). Positive means B is better.
        better = a if res["dm_stat"] < 0 else b if res["dm_stat"] > 0 else "tie"
        if res["p_value"] < 0.01:
            sig = "*** (p<0.01)"
        elif res["p_value"] < 0.05:
            sig = "**  (p<0.05)"
        elif res["p_value"] < 0.10:
            sig = "*   (p<0.10)"
        else:
            sig = "    (n.s.)"
        print(f"  {a} vs {b:8s}  "
              f"DM = {res['dm_stat']:+7.3f}  "
              f"p = {res['p_value']:.4f}  {sig}  "
              f"-> {better} has smaller MSE")
        rows.append({
            "model_a": a,
            "model_b": b,
            "dm_stat": res["dm_stat"],
            "p_value": res["p_value"],
            "mean_diff": res["mean_diff"],
            "n_dates": res["n_dates"],
            "winner": better,
            "significance": sig.strip(),
        })

    # 3x3 significance table for the report
    sig_matrix = pd.DataFrame(
        np.nan, index=MODELS, columns=MODELS, dtype=float,
    )
    for row in rows:
        # Mirror the test across the diagonal: stat for (a, b) is the
        # negative of (b, a); p-value is symmetric.
        sig_matrix.loc[row["model_a"], row["model_b"]] = row["dm_stat"]
        sig_matrix.loc[row["model_b"], row["model_a"]] = -row["dm_stat"]
    print("\nDM statistic matrix (negative = row model better than col model):")
    print(sig_matrix.round(3).to_string())

    # Persist
    dm_df = pd.DataFrame(rows)
    dm_df.to_parquet(RESULTS_DIR / "dm_results.parquet")
    sig_matrix.to_parquet(RESULTS_DIR / "dm_statistic_matrix.parquet")

    summary = {
        "source_phase": str(src.name),
        "test_start": str(TEST_START.date()),
        "test_end": str(TEST_END.date()),
        "newey_west_lags": NEWEY_WEST_LAGS,
        "per_model_mse": per_model_mse,
        "pairs": rows,
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nWrote {RESULTS_DIR.name}/dm_results.parquet, "
          f"dm_statistic_matrix.parquet, summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
