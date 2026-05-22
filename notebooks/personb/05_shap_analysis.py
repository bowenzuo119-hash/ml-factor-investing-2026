"""Phase 5a: SHAP feature-importance analysis on the canonical 8-feature
tuned XGBoost.

XGBoost's built-in `feature_importances_` is gain-based and rolled-up to a
single number per feature -- useful but cannot show direction, interactions,
or per-row attribution. SHAP (SHapley Additive exPlanations) values give us:

  * Mean |SHAP| per feature -- the more honest importance metric.
  * Summary plot: for each feature, distribution of SHAP values colored by
    feature value. Shows whether HIGH momentum -> POSITIVE or NEGATIVE
    contribution to predicted return.
  * Dependence plots: how each feature's effect changes with its value.

We fit one tuned XGBoost on the full pre-test panel (2005-2018), then
explain a random subsample of test-window rows.

Run with:
    .venv/bin/python -m notebooks.personb.05_shap_analysis
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
# Skip the `shap` package and use XGBoost's native pred_contribs=True
# (which is the exact mechanism shap.TreeExplainer wraps). Reason: shap
# 0.46 (the version compatible with our numpy 1.26) chokes on XGBoost
# 3.2's JSON-array `base_score` field; upgrading shap needs numpy 2.0+
# which would break the venv. Native pred_contribs returns the same
# values and we plot with plain matplotlib.

from src.factors import build_feature_panel


RESULTS_DIR = (
    Path(__file__).resolve().parents[2] / "results" / "05_shap_analysis"
)
PANEL_FILE = (
    Path(__file__).resolve().parents[2] / "data" / "processed"
    / "returns_spliced_2005_2024.parquet"
)

TRAIN_END = pd.Timestamp("2018-12-31")
TEST_START = pd.Timestamp("2019-01-01")
TEST_END = pd.Timestamp("2024-12-31")

INCLUDE_FEATURES = ("mom", "rev", "mvol", "ivol", "log_mktcap",
                    "bm", "ep", "dvol",
                    "roe", "roa", "de", "asset_growth", "accruals")

# Sample size for the SHAP explanation -- 5000 (date, ticker) rows
# from the test window. Enough for stable distribution plots, fast
# enough to run in seconds.
SHAP_SAMPLE_SIZE = 5000
RANDOM_STATE = 42


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("Phase 5a: SHAP analysis on the canonical 8-feature tuned XGBoost")
    print("=" * 72)

    # 1. Build features ----------------------------------------------
    print("\n[1/5] Building feature panel...")
    panel = build_feature_panel(
        start="2005-01-01", end="2024-12-31",
        include=INCLUDE_FEATURES, sector_rank=True,
    )
    X_all = panel.drop(columns=["sector"])
    print(f"  panel: {X_all.shape}")

    # 2. Realised returns -> y ----------------------------------------
    print("\n[2/5] Loading realised next-month returns...")
    returns = pd.read_parquet(PANEL_FILE)
    realised = returns.shift(-1).stack(future_stack=True).rename("y")
    realised.index = realised.index.set_names(["date", "ticker"])

    joint = pd.concat([X_all, realised], axis=1, join="inner").dropna()
    y = joint["y"]
    X = joint.drop(columns=["y"])

    dates = X.index.get_level_values("date")
    train_mask = dates <= TRAIN_END
    test_mask = (dates >= TEST_START) & (dates <= TEST_END)
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    print(f"  train rows: {len(X_train):,}, test rows: {len(X_test):,}")

    # 3. Fit canonical tuned XGBoost ----------------------------------
    print("\n[3/5] Fitting tuned XGBoost (13-feature defaults) on 2005-2018...")
    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.0115,
        subsample=0.717, colsample_bytree=0.890, min_child_weight=11,
        reg_alpha=0.794, reg_lambda=2.305, tree_method="hist",
        random_state=RANDOM_STATE, n_jobs=1,
    )
    model.fit(X_train.to_numpy(), y_train.to_numpy())
    print(f"  fit done. n_estimators={model.n_estimators}")

    # 4. SHAP values on a test-window subsample ---------------------
    print(f"\n[4/5] Computing SHAP values on {SHAP_SAMPLE_SIZE:,} test rows...")
    rng = np.random.default_rng(RANDOM_STATE)
    if len(X_test) > SHAP_SAMPLE_SIZE:
        idx = rng.choice(len(X_test), size=SHAP_SAMPLE_SIZE, replace=False)
        X_sample = X_test.iloc[idx]
    else:
        X_sample = X_test

    # XGBoost's native SHAP: model.predict(X, pred_contribs=True) returns a
    # (n_samples, n_features + 1) array. Last column is the bias term;
    # the first n_features columns are the per-feature SHAP contributions.
    # Tree-exact, sub-second on 5000 rows.
    booster = model.get_booster()
    dmat = xgb.DMatrix(X_sample.to_numpy(), feature_names=list(X_sample.columns))
    shap_full = booster.predict(dmat, pred_contribs=True)
    shap_values = shap_full[:, :-1]  # drop the bias column
    bias = float(shap_full[0, -1])
    print(f"  shap_values shape: {shap_values.shape}, bias = {bias:+.5f}")

    # Mean |SHAP| per feature -- the standard "feature importance" metric
    mean_abs = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X_sample.columns,
    ).sort_values(ascending=False)
    total = mean_abs.sum()
    print("\nMean |SHAP| per feature (share of total):")
    for f, v in mean_abs.items():
        pct = 100 * v / total
        bar = "#" * int(pct / 2)
        print(f"  {f:12s}  {v:.5f}  {pct:5.1f}%  {bar}")

    # 5. Plots --------------------------------------------------------
    print("\n[5/5] Saving plots...")

    # 5a. Bar plot: mean |SHAP| per feature
    fig, ax = plt.subplots(figsize=(9, 5))
    colors_bar = plt.cm.viridis(np.linspace(0.2, 0.8, len(mean_abs)))
    ax.barh(mean_abs.index[::-1], mean_abs.values[::-1],
            color=colors_bar[::-1], edgecolor="black", linewidth=0.4)
    ax.set_title("Mean |SHAP| value per feature\n"
                 "Tuned XGBoost, 8 features, 5,000 test-window rows",
                 fontsize=11, weight="bold")
    ax.set_xlabel("Mean |SHAP value|  (units of return)")
    for i, v in enumerate(mean_abs.values[::-1]):
        ax.text(v, i, f"  {100*v/total:.1f}%", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "shap_importance_bar.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)

    # 5b. Summary plot in the shap-beeswarm style, but written by hand
    # since we are not importing shap. For each feature: a horizontal
    # scatter where x = SHAP value, y = feature row (jittered), and
    # colour = feature value. Standard "does high momentum push UP or
    # DOWN?" view.
    ordered_features = mean_abs.index.tolist()
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, feat in enumerate(reversed(ordered_features)):
        feat_idx = list(X_sample.columns).index(feat)
        y_pos = i + rng.uniform(-0.35, 0.35, size=len(X_sample))
        sc = ax.scatter(
            shap_values[:, feat_idx], y_pos,
            c=X_sample[feat].to_numpy(),
            cmap="coolwarm", s=4, alpha=0.55,
            vmin=0, vmax=1,
        )
    cb = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.04)
    cb.set_label("Sector-relative rank of the feature\n"
                 "(0 = lowest in sector, 1 = highest)", fontsize=9)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(range(len(ordered_features)))
    ax.set_yticklabels(list(reversed(ordered_features)))
    ax.set_xlabel("SHAP value  (effect on predicted next-month return)")
    ax.set_title("SHAP summary — how each feature pushes the prediction\n"
                 "Red dots = high feature value in sector; blue = low",
                 fontsize=11, weight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "shap_summary_beeswarm.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)

    # 5c. Per-feature dependence plots for the top-4
    top4 = mean_abs.head(4).index.tolist()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, feat in zip(axes.ravel(), top4):
        feat_idx = list(X_sample.columns).index(feat)
        ax.scatter(
            X_sample[feat].to_numpy(),
            shap_values[:, feat_idx],
            s=3, alpha=0.35, color="#1F3864",
        )
        # Local mean to show the shape
        bins = np.linspace(0, 1, 11)
        x_centres = 0.5 * (bins[:-1] + bins[1:])
        vals = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (X_sample[feat].to_numpy() >= lo) & (
                X_sample[feat].to_numpy() < hi)
            vals.append(np.nan if not mask.any() else
                        shap_values[mask, feat_idx].mean())
        ax.plot(x_centres, vals, color="#DC2626", lw=2,
                label="Decile-binned mean", marker="o")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(f"{feat}", fontsize=10, weight="bold")
        ax.set_xlabel(f"{feat} (sector-relative rank)")
        ax.set_ylabel("SHAP value")
        ax.grid(alpha=0.3)
    fig.suptitle("Per-feature dependence — top 4 features\n"
                 "(red curve = decile-binned mean SHAP)",
                 fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "shap_dependence_top4.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)

    # Persist the raw SHAP frame so anyone can reproduce / re-plot
    shap_df = pd.DataFrame(shap_values, index=X_sample.index,
                           columns=[f"shap_{c}" for c in X_sample.columns])
    shap_df.to_parquet(RESULTS_DIR / "shap_values.parquet")
    mean_abs.to_frame("mean_abs_shap").to_parquet(
        RESULTS_DIR / "feature_importance_shap.parquet"
    )

    print(f"\nWrote 3 PNG plots, shap_values.parquet, "
          f"feature_importance_shap.parquet")
    print(f"  -> {RESULTS_DIR.relative_to(Path.cwd())}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
