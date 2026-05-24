"""qa_figures.py - generate Q&A defense figures for the presentation.

For each high-likelihood examiner question (per `report/QA_PREP.md`), produces
a single-panel chart that supports the spoken answer with a visible artefact.
Saves into `results/qa_figures/` so they can be pasted into slides during
prep or shown in the Q&A itself.

Four figures generated:

  1. placebo_vs_real.png - REAL +1.15 Sharpe vs SHUFFLED -0.94 (Q1).
     The cleanest visual answer to "is this just a backtest artefact?"

  2. model_comparison.png - Lasso vs XGBoost vs NN, Sharpe + FF5 alpha t-stat
     across the 3 reporting windows (Q2, Q5).

  3. where_alpha_lives.png - Broad survivorship-free (~4,400) vs strict
     top-2000 universe, FF5 alpha + t-stat side by side (Q8).
     The down-cap-concentration finding.

  4. momentum_control.png - FF5 alpha vs FF5+UMD (Carhart) alpha, with UMD
     loading annotated (Q9). Proves the alpha is NOT repackaged momentum.

Run with:
    .venv/bin/python -m notebooks.personb.qa_figures
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "qa_figures"

# Color palette (consistent across figures)
COL_REAL = "#1F3864"      # dark navy -- canonical / authoritative
COL_PLACEBO = "#DC2626"   # red -- placebo / null
COL_GOOD = "#22C55E"      # green -- positive / confirming
COL_NEUTRAL = "#94A3B8"   # grey -- baseline / reference
COL_WARN = "#F59E0B"      # amber -- caveat


def fig_placebo() -> Path:
    """Q1: placebo shuffle — REAL features vs SHUFFLED features Sharpe."""
    labels = ["REAL features\n(canonical)", "SHUFFLED\nseed=0", "SHUFFLED\nseed=1", "SHUFFLED\nmean"]
    sharpes = [1.153, -1.034, -0.847, -0.940]
    colors = [COL_REAL, COL_PLACEBO, COL_PLACEBO, COL_PLACEBO]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, sharpes, color=colors, edgecolor="white", lw=1.5)
    # Annotate each bar
    for b, v in zip(bars, sharpes):
        ha = "center"
        va = "bottom" if v >= 0 else "top"
        offset = 0.04 if v >= 0 else -0.04
        ax.text(b.get_x() + b.get_width() / 2, v + offset, f"{v:+.2f}",
                ha=ha, va=va, fontsize=13, weight="bold")
    ax.axhline(0, color="black", lw=1.0)
    ax.axhline(1.153, color=COL_REAL, ls=":", lw=0.7, alpha=0.6)
    ax.annotate("", xy=(0, 1.153), xytext=(3, -0.94),
                arrowprops=dict(arrowstyle="<->", color="#475569", lw=1.2, alpha=0.5))
    ax.text(1.5, 0.1, "Δ Sharpe ≈ 2.09\n(REAL − shuffled mean)",
            ha="center", fontsize=10.5, color="#475569",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#94A3B8", alpha=0.9))
    ax.set_ylabel("Full-OOS Sharpe ratio (2012-2024)", fontsize=12)
    ax.set_title("Q1 — Placebo: shuffling feature → ticker mapping kills the edge\n"
                 "+1.15 Sharpe collapses to −0.94 → genuine ML feature content, not backtest leakage",
                 fontsize=12.5, weight="bold", pad=12)
    ax.set_ylim(-1.5, 1.6)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "placebo_vs_real.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_model_comparison() -> Path:
    """Q2 / Q5: Lasso vs XGBoost vs NN, Sharpe + FF5 alpha t-stat per window."""
    models = ["Lasso", "XGBoost", "NN"]
    sharpe = {
        "Full-OOS\n2012-2024":  [0.71, 1.15, 0.62],
        "Long-OOS\n2015-2024":  [0.67, 0.97, 0.53],
        "Test-OOS\n2019-2024":  [0.65, 1.00, 0.49],
    }
    alpha_t = {
        "Full-OOS\n2012-2024":  [2.40, 6.85, 1.37],
        "Long-OOS\n2015-2024":  [2.78, 6.00, 1.69],
        "Test-OOS\n2019-2024":  [2.38, 5.00, 0.96],
    }
    colors = {"Lasso": COL_NEUTRAL, "XGBoost": COL_REAL, "NN": COL_WARN}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    bar_w = 0.27
    windows = list(sharpe.keys())
    xs = np.arange(len(windows))
    for i, m in enumerate(models):
        ax1.bar(xs + (i - 1) * bar_w, [sharpe[w][i] for w in windows],
                bar_w, label=m, color=colors[m], edgecolor="white", lw=1.2)
        ax2.bar(xs + (i - 1) * bar_w, [alpha_t[w][i] for w in windows],
                bar_w, label=m, color=colors[m], edgecolor="white", lw=1.2)

    for ax, ylabel, title, ref_line, ref_label in [
        (ax1, "Sharpe ratio (10 bps/side)",
         "(a) Sharpe ratio by model × window", 0, None),
        (ax2, "FF5 alpha t-stat (Newey-West)",
         "(b) FF5 alpha t-stat by model × window", 2.0, "t=2 (5% sig)"),
    ]:
        ax.set_xticks(xs)
        ax.set_xticklabels(windows, fontsize=10.5)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11.5, weight="bold")
        ax.axhline(ref_line, color="grey", lw=0.6)
        if ref_label:
            ax.axhline(ref_line, color="grey", lw=0.7, ls="--")
            ax.text(2.5, ref_line + 0.15, ref_label, fontsize=9, color="grey", ha="right")
        ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Q2/Q5 — Model comparison: XGBoost wins on Sharpe AND FF5 α t-stat across every window\n"
                 "Diebold-Mariano: XGBoost > Lasso (p<0.01), XGBoost > NN (p<0.05)",
                 fontsize=12, weight="bold", y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "model_comparison.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_where_alpha_lives() -> Path:
    """Q8: broad survivorship-free vs strict top-2000 — the down-cap finding."""
    universes = ["Broad survivorship-free\n(~4,400 names/mo, canonical)",
                 "Strict rolling top-2,000\n(large/mid-cap end only)"]
    alphas = [18.73, 1.80]   # %/yr
    t_stats = [6.85, 0.96]
    mkt_b = [1.29, 0.28]
    smb_b = [1.26, 0.15]

    fig, axes = plt.subplots(1, 4, figsize=(15, 5.5))

    def _bar(ax, vals, title, fmt, colors=None, ref=None, ref_label=None):
        if colors is None:
            colors = [COL_REAL, COL_NEUTRAL]
        bars = ax.bar(["Broad", "Top-2,000"], vals, color=colors,
                      edgecolor="white", lw=1.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + abs(v) * 0.05 if v != 0 else 0.05,
                    fmt.format(v), ha="center", va="bottom",
                    fontsize=12, weight="bold")
        if ref is not None:
            ax.axhline(ref, color="grey", ls="--", lw=0.7, alpha=0.6)
            if ref_label:
                ax.text(1.4, ref + (max(vals) - min(vals)) * 0.05,
                        ref_label, fontsize=9, color="grey", ha="right")
        ax.set_title(title, fontsize=11, weight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(vals) * 1.25)

    _bar(axes[0], alphas, "FF5 α (%/yr)", "{:+.1f}%")
    _bar(axes[1], t_stats, "FF5 α t-stat", "{:+.2f}", ref=2.0, ref_label="t=2 (5% sig)")
    _bar(axes[2], mkt_b, "Mkt-β", "{:+.2f}")
    _bar(axes[3], smb_b, "SMB-β", "{:+.2f}")

    # Annotate the n.s. on top-2,000
    axes[1].annotate("n.s. ↓", xy=(1, 0.96), xytext=(1.2, 3.5),
                     arrowprops=dict(arrowstyle="->", color=COL_PLACEBO, lw=1.5),
                     fontsize=11, color=COL_PLACEBO, weight="bold")

    fig.suptitle("Q8 — Where the alpha lives: the headline +18.7%/yr is a DOWN-CAP effect (GKX 2020 §IV.D)\n"
                 "On the strict rolling top-2,000 alone, the FF5 alpha is +1.8%/yr at t=0.96 (not significant)",
                 fontsize=12, weight="bold", y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "where_alpha_lives.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_momentum_control() -> Path:
    """Q9: FF5 alpha vs FF5+UMD alpha + UMD loading."""
    specs = ["FF5\n(5-factor)", "FF5 + UMD\n(Carhart 6F)"]
    alphas = [17.7, 20.1]    # %/yr
    t_stats = [6.11, 7.40]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: alpha bars
    bars = ax1.bar(specs, alphas, color=[COL_NEUTRAL, COL_REAL],
                   edgecolor="white", lw=1.5)
    for b, v, t in zip(bars, alphas, t_stats):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.5,
                 f"α = {v:+.1f}%/yr\nt = {t:+.2f}",
                 ha="center", va="bottom", fontsize=11.5, weight="bold")
    ax1.annotate("", xy=(1, 20.1), xytext=(0, 17.7),
                 arrowprops=dict(arrowstyle="->", color=COL_GOOD, lw=2.0))
    ax1.text(0.5, 22, "α RISES by +2.4 pp\nwhen UMD is added", ha="center",
             fontsize=10, color=COL_GOOD, weight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0FDF4",
                       edgecolor=COL_GOOD, lw=1.0))
    ax1.set_ylabel("FF-adjusted alpha (%/yr, Newey-West)", fontsize=11)
    ax1.set_title("(a) Adding UMD to the regression RAISES alpha", fontsize=11.5, weight="bold")
    ax1.set_ylim(0, 26)
    ax1.grid(axis="y", alpha=0.3)

    # Right: UMD loading -- the "smoking gun"
    ax2.bar(["UMD loading\n(Carhart 6F)"], [-0.43], color=COL_PLACEBO,
            edgecolor="white", lw=1.5, width=0.45)
    ax2.text(0, -0.43 - 0.04, "β = −0.43\nt = −4.61", ha="center", va="top",
             fontsize=12.5, weight="bold")
    ax2.axhline(0, color="black", lw=1.0)
    ax2.text(0, -0.18, "MOMENTUM-AVERSE\n(short loading on UMD)",
             ha="center", fontsize=11, color="white", weight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor=COL_PLACEBO, alpha=0.85))
    ax2.set_ylabel("Factor loading (β)", fontsize=11)
    ax2.set_title("(b) UMD coefficient: portfolio is momentum-AVERSE", fontsize=11.5, weight="bold")
    ax2.set_ylim(-0.6, 0.1)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Q9 — Momentum control: the +18.7% FF5 α is NOT repackaged momentum premium\n"
                 "Carhart-6F α = +20.1%/yr (rises), UMD β = −0.43 (momentum-averse) ⇒ alpha is genuine cross-sectional skill",
                 fontsize=11.5, weight="bold", y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "momentum_control.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 66)
    print("Generating Q&A defense figures for the presentation")
    print("=" * 66)

    figs = [
        ("Q1 - placebo shuffle", fig_placebo),
        ("Q2/Q5 - model comparison", fig_model_comparison),
        ("Q8 - where the alpha lives", fig_where_alpha_lives),
        ("Q9 - momentum control", fig_momentum_control),
    ]
    for label, fn in figs:
        out = fn()
        print(f"  [{label}] wrote {out}")

    print(f"\nAll {len(figs)} figures in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
