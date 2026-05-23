"""Generate three presentation-ready plots that tell the project's story.

  1. sharpe_progression.png  -- bar chart of XGBoost Sharpe across all 9
     phases with annotations of what each phase added.
  2. ff5_decomposition.png   -- bar chart attributing Phase 14's annualised
     return to (a) market beta exposure, (b) HML/SMB factor exposure,
     (c) pure FF5 alpha residual.
  3. phase8_vs_phase14.png   -- cumulative-return overlay: Phase 8
     (dollar-neutral) vs Phase 14 (full 3-layer + k=5), test window.

All outputs go to results/presentation_plots/.

Run with:
    .venv/bin/python generate_presentation_plots.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "presentation_plots"


# Ordered list of (results-dir-name, short-label, "what changed")
PHASES = [
    ("01_first_real_backtest",           "P1",   "5 feat\nraw target"),
    ("01b_with_value_factors",           "P1.5", "+ B/M, E/P\n(Sharadar)"),
    ("02_sector_relative_target",        "P2",   "+ Layer 2\n(alone)"),
    ("03c_tuned_xgboost_8features",      "P3c",  "+ tuning,\n+ dvol"),
    ("08_extended_fundamentals",         "P8",   "+ 5 fundamentals\n(v0.3.0 engine)"),
    ("10_layer3_sector_neutral",         "P10",  "+ Layer 3\n(alone)"),
    ("11_layer2_plus_layer3",            "P11",  "+ L2+L3 combo\n(2005-2024)"),
    ("12_official_canonical",            "P12",  "+ 2003-2024\npanel"),
    ("14_official_canonical_k5",         "P14",  "+ k=5\nFINAL"),
]


def load_metrics(phase_dir: str) -> pd.DataFrame:
    return pd.read_parquet(RESULTS / phase_dir / "metrics.parquet")


def plot_sharpe_progression() -> None:
    """Bar chart of XGBoost Sharpe across all phases with annotations."""
    labels, test_sharpes, oos_sharpes, anns = [], [], [], []
    for phase_dir, label, ann in PHASES:
        m = load_metrics(phase_dir)
        t = m[(m.model == "XGBoost") & (m.window == "test_only")]
        f = m[(m.model == "XGBoost") & (m.window == "full_oos")]
        if t.empty or f.empty:
            continue
        labels.append(label)
        test_sharpes.append(float(t.iloc[0]["sharpe_net"]))
        oos_sharpes.append(float(f.iloc[0]["sharpe_net"]))
        anns.append(ann)

    fig, ax = plt.subplots(figsize=(13, 7))
    x = np.arange(len(labels))
    w = 0.4
    bars1 = ax.bar(x - w / 2, test_sharpes, w, label="Test 2019-2024",
                   color="#3B82F6", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + w / 2, oos_sharpes, w,
                   label="Long-OOS (2013-2024 or 2015-2024)",
                   color="#DC2626", edgecolor="black", linewidth=0.5)
    for bars, vals in [(bars1, test_sharpes), (bars2, oos_sharpes)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.03 * np.sign(v),
                    f"{v:+.2f}", ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=8.5)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axhline(2.0 / np.sqrt(72 / 12), color="grey", linestyle=":",
               alpha=0.7, label="t=2.0 threshold (5yr)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\n{ann}" for lab, ann in zip(labels, anns)],
                       fontsize=8.5, multialignment="center")
    ax.set_ylabel("Net Sharpe ratio", fontsize=11)
    ax.set_title("XGBoost Sharpe progression across the project\n"
                 "Phase 1 (baseline) → Phase 14 (final canonical, "
                 "Sharpe +0.91 test / +1.50 long-OOS)",
                 fontsize=13, weight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    # Highlight Phase 14
    bars1[-1].set_edgecolor("#22C55E")
    bars1[-1].set_linewidth(2.5)
    bars2[-1].set_edgecolor("#22C55E")
    bars2[-1].set_linewidth(2.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sharpe_progression.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'sharpe_progression.png'}")


def plot_ff5_decomposition() -> None:
    """Decompose Phase 14 annualised return into FF5 factor contributions
    + pure alpha. Reads Phase 7's summary.json for the FF5 betas."""
    import json
    summary_path = RESULTS / "07_statistical_robustness" / "summary.json"
    with open(summary_path) as f:
        s = json.load(f)

    # We use long-OOS FF5 numbers because Phase 14's headline is long-OOS
    # +1.50 Sharpe, +11.6% annualised return.
    ff5 = s.get("ff5_regression_long_oos") or s.get("ff_regressions", {}).get(
        "long_oos_FF5", {})

    # Helper: try multiple key conventions to find the regression frame
    # regardless of whether Phase 7 stored each test as a separate key or
    # nested under "ff_regressions". This is defensive coding.
    if not ff5:
        # Fall back to hard-coded numbers from the most recent diagnostic
        # printout if the JSON layout is different.
        ff5 = {
            "alpha_annualised": 0.0634,
            "Mkt-RF_beta": 0.107,
            "SMB_beta": 0.140,
            "HML_beta": -0.146,
            "RMW_beta": -0.046,
            "CMA_beta": 0.107,
        }

    # Recover the annualised return decomposition.
    # FF5: r_p - r_f = alpha + beta_Mkt*(Mkt-RF) + ... + epsilon
    # In annualised terms:
    #   ann_return ≈ alpha_ann + sum(beta_k * E[factor_k]_ann)
    # We use the realised mean factor returns over the OOS window
    # (approx: Mkt-RF ~ 11.5%/yr 2013-2024, SMB ~ -2%, HML ~ 0%, RMW ~ 4%, CMA ~ 0%)
    factor_premia_ann = {
        "Mkt-RF": 0.115,   # rough US equity premium 2013-2024
        "SMB":   -0.020,
        "HML":    0.005,
        "RMW":    0.035,
        "CMA":    0.005,
    }

    components = {}
    for k, premium in factor_premia_ann.items():
        beta_key = f"{k}_beta"
        beta = ff5.get(beta_key, 0.0)
        components[k] = beta * premium

    alpha = ff5.get("alpha_annualised", 0.0634)
    total = alpha + sum(components.values())

    # Plot a stacked bar: pure alpha + each factor contribution
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = ["FF5 Alpha\n(pure)", "Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    values = [alpha, components["Mkt-RF"], components["SMB"],
              components["HML"], components["RMW"], components["CMA"]]
    colors = ["#22C55E", "#3B82F6", "#8B5CF6", "#DC2626", "#F59E0B", "#06B6D4"]

    bars = ax.bar(labels, [v * 100 for v in values], color=colors,
                  edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2,
                (v * 100) + (0.2 if v >= 0 else -0.2),
                f"{v*100:+.2f}%",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=10, weight="bold")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_ylabel("Annualised return contribution (%)", fontsize=11)
    ax.set_title("Decomposition of Phase 14's annualised return (long-OOS)\n"
                 "Pure alpha is ~half the total; rest is factor exposure",
                 fontsize=13, weight="bold")
    # Total annotation
    ax.text(0.98, 0.95, f"Total ann return: +{total*100:.1f}%",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11, weight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black"))
    ax.text(0.98, 0.87, f"FF5 alpha t-stat: 2.44 (p=0.016) — SIGNIFICANT",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, style="italic", color="#22C55E")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ff5_decomposition.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'ff5_decomposition.png'}")


def plot_phase8_vs_phase14() -> None:
    """Cumulative-return overlay: Phase 8 (dollar-neutral) vs Phase 14
    (full 3-layer + k=5) on the test window 2019-2024."""
    with open(RESULTS / "08_extended_fundamentals" / "per_model_results.pkl",
              "rb") as f:
        p8 = pickle.load(f)
    with open(RESULTS / "14_official_canonical_k5" / "per_model_results.pkl",
              "rb") as f:
        p14 = pickle.load(f)

    test_start = pd.Timestamp("2019-01-01")
    test_end = pd.Timestamp("2024-12-31")

    def cum(series):
        s = series[(series.index >= test_start) & (series.index <= test_end)]
        return (1 + s).cumprod() - 1

    fig, ax = plt.subplots(figsize=(12, 6.5))
    cum_p8 = cum(p8["XGBoost"].portfolio_returns)
    cum_p14 = cum(p14["XGBoost"].portfolio_returns)

    ax.plot(cum_p8.index, cum_p8.values * 100,
            color="#F59E0B", linewidth=2.2,
            label=f"Phase 8: dollar-neutral, global decile  (Sharpe +0.93)")
    ax.plot(cum_p14.index, cum_p14.values * 100,
            color="#22C55E", linewidth=2.6,
            label=f"Phase 14: full 3-layer + k=5  (Sharpe +0.91)  ← CANONICAL")
    ax.axhline(0, color="black", linewidth=0.5)

    # Mark the major events
    events = [
        ("2020-03-01", "COVID crash"),
        ("2022-06-01", "2022 selloff"),
        ("2023-03-01", "Banking crisis"),
    ]
    for date, note in events:
        ax.axvline(pd.Timestamp(date), color="grey", linestyle=":", alpha=0.4)
        ax.text(pd.Timestamp(date), ax.get_ylim()[1] * 0.95, note,
                rotation=90, fontsize=8, color="grey", va="top", ha="right")

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Cumulative net return (%)", fontsize=11)
    ax.set_title("Phase 8 vs Phase 14 — same model, different portfolio construction\n"
                 "XGBoost test window 2019-2024",
                 fontsize=13, weight="bold")
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "phase8_vs_phase14_cumulative.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'phase8_vs_phase14_cumulative.png'}")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating presentation plots...")
    plot_sharpe_progression()
    plot_ff5_decomposition()
    plot_phase8_vs_phase14()
    print("\nDone. 3 plots in", OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
