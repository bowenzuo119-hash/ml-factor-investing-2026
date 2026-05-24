"""report_figure_project_arc.py - one-pager presentation summary.

A single landscape figure: "Project arc - 5 weeks, 24 phases, 1 honest
canonical", composed of
  - top:    the audit -> rebuild -> final timeline (text arrow)
  - mid-L:  headline Sharpe across the key phases (bar)
  - mid-R:  leaky vs honest equity curves
  - bottom: the final-canonical headline numbers table

    python -m notebooks.persona.report_figure_project_arc
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.metrics import sharpe_ratio

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
FIG = RES / "persona_figures" / "project_arc_one_pager.png"

C_LEAK, C_HONEST, C_GREY, C_AMBER, C_PURPLE = "#ef4444", "#2563eb", "#9ca3af", "#f59e0b", "#a78bfa"

# phase dir, short label, colour
PHASES = [
    ("14_official_canonical_k5", "P14\nleaky S&P", C_LEAK),
    ("15_canonical_2002", "P15\nS&P + PIT", C_GREY),
    ("22_canonical_relaxed_pit_retuned", "P22\nretuned PIT", C_AMBER),
    ("23g_canonical_qfiltered_orig_tune", "P23g\nbroad ★", C_HONEST),
    ("24b_canonical_all_gkx", "P24b\n+GKX", C_PURPLE),
]


def _sharpe(pkl):
    return sharpe_ratio(pickle.load(open(pkl, "rb"))["XGBoost"].portfolio_returns.dropna())


def main() -> int:
    # gather phase Sharpes
    bars = []
    for d, lab, c in PHASES:
        p = RES / d / "per_model_results.pkl"
        if p.exists():
            bars.append((lab, _sharpe(p), c))
    leaky = pickle.load(open(RES / "14_official_canonical_k5" / "per_model_results.pkl", "rb"))["XGBoost"].portfolio_returns.dropna()
    collapse = pickle.load(open(RES / "15_canonical_2002" / "per_model_results.pkl", "rb"))["XGBoost"].portfolio_returns.dropna()
    honest = pickle.load(open(RES / "23g_canonical_qfiltered_orig_tune" / "per_model_results.pkl", "rb"))["XGBoost"].portfolio_returns.dropna()

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.7, 2.0, 1.1], hspace=0.55, wspace=0.22)
    fig.suptitle("Project arc — 5 weeks, 24 phases, 1 honest canonical",
                 fontsize=16, fontweight="bold", y=0.98)

    # --- top: timeline arrow ---
    axt = fig.add_subplot(gs[0, :]); axt.axis("off")
    steps = [
        ("Baseline\n+1.50 Sharpe", C_LEAK),
        ("PIT audit\nleak found", "#111827"),
        ("S&P collapses\n−0.31", C_GREY),
        ("Broad Sharadar\nrebuild", C_AMBER),
        ("Honest canonical\n+1.07 (β-adj α t=5.6)", C_HONEST),
    ]
    n = len(steps)
    for i, (txt, c) in enumerate(steps):
        x = (i + 0.5) / n
        axt.text(x, 0.55, txt, ha="center", va="center", fontsize=10,
                 fontweight="bold", color="white",
                 bbox=dict(boxstyle="round,pad=0.4", fc=c, ec="none"))
        if i < n - 1:
            axt.annotate("", xy=((i + 1) / n, 0.55), xytext=((i + 0.92) / n, 0.55),
                         arrowprops=dict(arrowstyle="-|>", color="#374151", lw=2))
    axt.set_xlim(0, 1); axt.set_ylim(0, 1)

    # --- mid-left: phase Sharpe bar ---
    axb = fig.add_subplot(gs[1, 0])
    labs = [b[0] for b in bars]; vals = [b[1] for b in bars]; cols = [b[2] for b in bars]
    axb.bar(range(len(bars)), vals, color=cols, edgecolor="white")
    axb.axhline(0, color="black", lw=0.6)
    for i, v in enumerate(vals):
        axb.text(i, v + (0.04 if v >= 0 else -0.10), f"{v:+.2f}", ha="center",
                 fontsize=9, fontweight="bold")
    axb.set_xticks(range(len(bars))); axb.set_xticklabels(labs, fontsize=8)
    axb.set_ylabel("net Sharpe (full-OOS)")
    axb.set_title("Headline Sharpe across key phases", fontsize=11, fontweight="bold")
    axb.grid(alpha=0.2, axis="y")

    # --- mid-right: leaky vs honest equity ---
    axe = fig.add_subplot(gs[1, 1])
    for r, c, ls, lab in [(leaky, C_LEAK, "--", f"leaky S&P {sharpe_ratio(leaky):+.2f}"),
                          (collapse, C_GREY, "-", f"S&P+PIT {sharpe_ratio(collapse):+.2f}"),
                          (honest, C_HONEST, "-", f"broad {sharpe_ratio(honest):+.2f}")]:
        cum = (1 + r).cumprod()
        axe.plot(cum.index, cum.values, color=c, ls=ls, lw=1.8, label=lab)
    axe.set_yscale("log"); axe.set_ylabel("growth of $1 (log)")
    axe.set_title("Leaky vs survivorship-corrected equity", fontsize=11, fontweight="bold")
    axe.legend(fontsize=8, loc="upper left"); axe.grid(alpha=0.2, which="both")

    # --- bottom: headline table ---
    axn = fig.add_subplot(gs[2, :]); axn.axis("off")
    axn.text(0.0, 1.18, "Final canonical — headline numbers", transform=axn.transAxes,
             fontsize=11, fontweight="bold", va="bottom")
    rows = [
        ["Canonical", "XGBoost · broad top-2000 PIT · 13 features · k=20/sector · 10 bps/side"],
        ["Net Sharpe", "+1.07 full-OOS (2012–24) · +1.00 test (2019–24)"],
        ["FF5 alpha", "+17.7%/yr, t = +5.58 (Newey-West) — survives SMB control"],
        ["At 30 bps/side", "Sharpe 0.94 · FF5 alpha +13.5%/yr, t = 4.28 (robust to ~50 bps)"],
        ["Out-of-time", "static 2002–18 → 2019–24 Sharpe +1.04 (vs WF +0.996) — not a refit artifact"],
        ["Character", "high-beta directional L/S (Mkt-β ≈ 1.4, vol 32%, DD −34%) + significant α — NOT market-neutral"],
    ]
    tbl = axn.table(cellText=rows, colWidths=[0.16, 0.84], cellLoc="left", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1, 1.6)
    for (r, _), cell in tbl.get_celld().items():
        cell.set_edgecolor("#e5e7eb")
        if r % 2 == 0:
            cell.set_facecolor("#f9fafb")

    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"phase Sharpes: {[(l, round(v,2)) for l,v,_ in bars]}")
    print(f"wrote {FIG.name}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
