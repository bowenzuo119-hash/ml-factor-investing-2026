"""presentation_diagrams.py - hand-built schematic diagrams for the talk.

Six clean, presentation-ready diagrams (not data plots -- conceptual
flowcharts and timelines) that illustrate the project's process at
different levels:

  1. pipeline_overview.png       -- the 3-workstream architecture + data flow
  2. walk_forward_cv.png         -- sliding-window CV illustration
  3. universe_construction.png   -- funnel from raw Sharadar to broad panel
  4. audit_journey.png           -- Sharpe timeline showing the +1.49 -> -0.31 -> +1.15 story
  5. robustness_battery.png      -- waterfall of robustness checks
  6. return_decomposition.png    -- how the +4,600% breaks down honestly

All saved under results/diagrams/ at 200 dpi (presentation-ready).

Run with:
    .venv/bin/python -m notebooks.personb.presentation_diagrams
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "diagrams"

# Consistent palette (matches our existing figures)
COL_DATA = "#1F3864"     # navy -- Person A (data lane)
COL_MODEL = "#22C55E"    # green -- Person B (alpha model)
COL_REGIME = "#A855F7"   # purple -- Person C (regime overlay)
COL_ENGINE = "#F59E0B"   # amber -- Bowen's engine
COL_OUTPUT = "#DC2626"   # red -- final outputs / honest findings
COL_LIGHT = "#E5E7EB"    # light grey background
COL_TEXT = "#111827"     # near-black text


# ============================================================
# 1. Pipeline overview
# ============================================================
def fig_pipeline_overview(out):
    fig, ax = plt.subplots(figsize=(15, 7.5))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def box(x, y, w, h, label, sublabel=None, color="#1F3864", text_color="white"):
        bb = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15",
                             facecolor=color, edgecolor="white", lw=2)
        ax.add_patch(bb)
        ax.text(x + w/2, y + h/2 + (0.18 if sublabel else 0), label,
                ha="center", va="center", fontsize=11.5, weight="bold", color=text_color)
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.25, sublabel,
                    ha="center", va="center", fontsize=8.5, color=text_color, style="italic")

    def arrow(x1, y1, x2, y2, label=None):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=2.0, color="#374151"))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.18, label, ha="center", fontsize=8.5,
                    color="#374151", style="italic",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="#D1D5DB", alpha=0.9))

    # Title
    ax.text(7.5, 7.6, "Project pipeline — 3 workstreams, single shared interface",
            ha="center", fontsize=14, weight="bold")

    # === Row 1: Inputs ===
    # Person A: Data lane
    box(0.3, 5.3, 4.4, 1.5, "PERSON A (BOWEN) — Data Lane",
        "Sharadar SF1 / SEP / DAILY / TICKERS / SP500 / ACTIONS\n"
        "PIT filter • Q-filter • broad survivorship-free panel",
        COL_DATA)

    # Person C: Regime overlay (separate input)
    box(10.3, 5.3, 4.4, 1.5, "PERSON C (ANDREA) — Regime Overlay",
        "Walk-forward HMM on 6 macro features\n"
        "(VIX, vol, term spread, credit spread, S&P 3mo)",
        COL_REGIME)

    # === Middle: Features panel ===
    box(2.2, 3.3, 4.0, 1.3, "Features panel",
        "1.24M rows, ~4,400 names/mo\n14 GKX-style features",
        "#475569", text_color="white")

    # === Middle: Returns panel ===
    box(6.5, 3.3, 2.0, 1.3, "Returns panel",
        "276 mo × 5,897 tickers\nSEP closeadj (split/div adj.)",
        "#475569", text_color="white")

    # === Middle: Regime overlay CSV ===
    box(8.8, 3.3, 4.0, 1.3, "Regime overlay CSV",
        "180 monthly labels (Jan 2010+)\nLeverage: 1.0 calm / 0.4 crisis",
        "#475569", text_color="white")

    # === Center: Person B (alpha model) ===
    box(3.0, 1.3, 5.0, 1.4, "PERSON B (NICOLAS) — Alpha Model",
        "Lasso / XGBoost / NN • Optuna-tuned\nSector-relative target, k=20/sector",
        COL_MODEL)

    # === Center: Backtest engine ===
    box(9.0, 1.3, 4.0, 1.4, "ENGINE (Bowen, v0.5.0)",
        "Walk-forward (120-mo window)\nPIT-correct, 10 bps/side, sanity 3/3",
        COL_ENGINE)

    # === Bottom: Outputs ===
    box(3.5, -0.2, 8.0, 1.0, "OUTPUT: Phase 24-RT canonical",
        "Sharpe +1.15 • FF5 α +18.73%/yr at t=+6.85 • beats S&P up to 25 bps/side",
        COL_OUTPUT)

    # === Arrows ===
    # A -> features + returns
    arrow(2.5, 5.3, 3.0, 4.6)  # A -> features
    arrow(3.5, 5.3, 7.5, 4.6)  # A -> returns
    # C -> overlay csv
    arrow(12.5, 5.3, 11.0, 4.6)
    # features + returns -> Person B
    arrow(4.2, 3.3, 5.0, 2.7)  # features -> B
    arrow(7.0, 3.3, 6.5, 2.7)  # returns -> B
    # B -> engine
    arrow(8.0, 2.0, 9.0, 2.0)
    # overlay csv -> engine
    arrow(10.8, 3.3, 10.8, 2.7)
    # engine -> output
    arrow(11.0, 1.3, 9.0, 0.8)

    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ============================================================
# 2. Walk-forward CV
# ============================================================
def fig_walk_forward_cv(out):
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.set_xlim(2002, 2026)
    ax.set_ylim(0, 12)
    ax.axis("off")

    # Title positioned in data coordinates (xlim mid = 2014)
    ax.text(2014, 11.3, "Walk-forward cross-validation: 120-mo training window, 12-mo test, block-gated refit",
            ha="center", fontsize=13.5, weight="bold")
    ax.text(2014, 10.55,
            "Each row = one walk-forward iteration. Train window slides forward 12 months at each step.",
            ha="center", fontsize=10, style="italic", color="#475569")

    # Draw timeline at the bottom
    timeline_y = 0.5
    ax.axhline(timeline_y, color="#374151", lw=1.2)
    for y in [2002, 2005, 2010, 2015, 2020, 2024]:
        ax.plot([y, y], [timeline_y-0.1, timeline_y+0.1], color="#374151", lw=1.2)
        ax.text(y, timeline_y - 0.5, str(y), ha="center", fontsize=10)

    # Draw 8 iterations (showing 8 of the 13 actual walk-forward steps)
    iterations = [
        (2002, 2012),  # 1st iter: train 2002-2011 (120 mo), test 2012
        (2003, 2013),
        (2004, 2014),
        (2005, 2015),
        (2009, 2019),
        (2012, 2022),
        (2013, 2023),
        (2014, 2024),
    ]
    # Reverse for top-to-bottom plotting
    iterations = list(reversed(iterations))
    iter_labels = ["Iter 8 (latest)", "Iter 7", "Iter 6", "Iter 5", "...", "Iter 3", "Iter 2", "Iter 1 (first)"]

    for i, ((train_start, test_start), label) in enumerate(zip(iterations, iter_labels)):
        y = 1.5 + i * 1.05
        # Training window (10 years)
        train_rect = Rectangle((train_start, y), test_start - train_start, 0.7,
                               facecolor=COL_MODEL, alpha=0.6, edgecolor=COL_MODEL, lw=1.2)
        ax.add_patch(train_rect)
        # Test window (1 year)
        test_rect = Rectangle((test_start, y), 1, 0.7,
                              facecolor=COL_OUTPUT, alpha=0.7, edgecolor=COL_OUTPUT, lw=1.2)
        ax.add_patch(test_rect)
        # Label
        ax.text(2001.5, y + 0.35, label, ha="right", va="center", fontsize=9, color="#374151")
        # Test year label
        ax.text(test_start + 0.5, y + 0.35, f"{test_start}", ha="center", va="center",
                fontsize=8, color="white", weight="bold")

    # Legend
    train_patch = mpatches.Patch(color=COL_MODEL, alpha=0.6, label="Training window (120 months = 10 years)")
    test_patch = mpatches.Patch(color=COL_OUTPUT, alpha=0.7, label="Test window (12 months = 1 year)")
    ax.legend(handles=[train_patch, test_patch], loc="upper left", fontsize=11,
              framealpha=0.95, bbox_to_anchor=(0.02, 0.95))

    # Annotation
    ax.text(2014, 10.0,
            "Result: 155 monthly test predictions (Feb 2012 → Dec 2024), all genuinely out-of-sample.",
            ha="center", fontsize=10.5, color="#1F3864", weight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F0F9FF", edgecolor=COL_DATA, lw=1.2))

    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ============================================================
# 3. Universe construction funnel
# ============================================================
def fig_universe_construction(out):
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(6.5, 7.5, "Universe construction — Sharadar raw tables → broad survivorship-free panel",
            ha="center", fontsize=13.5, weight="bold")

    # Funnel stages
    stages = [
        (1.5, 6.0, 10.0, "ALL Sharadar SEP tickers (any time, any exchange)",
         "~30,000 ticker-history rows", COL_DATA),
        (2.0, 4.8, 9.0, "Common stock on major US exchanges (NYSE/NASDAQ/ARCA/BATS)",
         "~17,689 tickers", "#3B6EBA"),
        (2.6, 3.6, 7.8, "PIT-eligible: firstpricedate ≤ asof ≤ lastpricedate",
         "Median ~6,500 alive names/month", "#5B92D8"),
        (3.2, 2.4, 6.6, "Top-2000 by mcap (rolling, union over time)",
         "~5,897 unique tickers (2002-2024)", "#7CB3E8"),
        (3.6, 1.2, 5.8, "Q-filter (drop endswith('Q') AND isdelisted=='Y')",
         "~1,114 bankruptcy tickers dropped", "#94C5F0"),
        (3.0, 0.0, 7.0, "FINAL: ~4,400 names/month median",
         "5,500 unique tickers across 2012-2024 trade history", COL_OUTPUT),
    ]

    for (x, y, w, label, sublabel, color) in stages:
        h = 1.0
        bb = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                             facecolor=color, edgecolor="white", lw=2)
        ax.add_patch(bb)
        ax.text(x + w/2, y + h*0.62, label, ha="center", va="center",
                fontsize=10.5, weight="bold", color="white")
        ax.text(x + w/2, y + h*0.25, sublabel, ha="center", va="center",
                fontsize=9, color="white", style="italic")

    # Arrows down
    for i in range(len(stages)-1):
        x = stages[i][0] + stages[i][2]/2
        y_top = stages[i][1]
        y_bot = stages[i+1][1] + 1.0
        ax.annotate("", xy=(x, y_bot), xytext=(x, y_top),
                    arrowprops=dict(arrowstyle="->", lw=2.0, color="#374151"))

    # Key insight box
    ax.text(11.5, 4.5, "Key:", fontsize=11, weight="bold", color="#1F3864")
    ax.text(11.5, 4.0, "• Survivorship-free", fontsize=9, color="#1F3864")
    ax.text(11.5, 3.6, "  (no backfilled history)", fontsize=8, color="#475569", style="italic")
    ax.text(11.5, 3.1, "• Point-in-time", fontsize=9, color="#1F3864")
    ax.text(11.5, 2.7, "  (no look-ahead)", fontsize=8, color="#475569", style="italic")
    ax.text(11.5, 2.2, "• ~4,400 not 2,000", fontsize=9, color="#1F3864")
    ax.text(11.5, 1.8, "  (alive set of historical", fontsize=8, color="#475569", style="italic")
    ax.text(11.5, 1.5, "   top-2,000 union)", fontsize=8, color="#475569", style="italic")

    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ============================================================
# 4. Audit journey timeline
# ============================================================
def fig_audit_journey(out):
    fig, ax = plt.subplots(figsize=(15, 7))

    phases = [
        ("Phase 14\n(pre-audit)", 1.49, "S&P 500 union\nk=5", "#94A3B8", "LEAKY"),
        ("Phase 15\n(PIT applied)", -0.31, "S&P-only,\nstrict PIT", "#EF4444", "−1.80\nfrom audit"),
        ("Phase 22\n(S&P honest)", 0.31, "Relaxed PIT\nretuned", "#F59E0B", "α not sig"),
        ("Phase 23g\n(broad rebuild)", 1.05, "Broad ~4,400\nuniverse + 13ft", "#22C55E", "first sig α"),
        ("Phase 24-RT\n(FINAL)", 1.15, "+ chmom\n14 features", "#1F3864", "α t=+6.85"),
    ]

    # Add S&P benchmark
    sp_sharpe = 0.99

    xs = np.arange(len(phases))
    sharpes = [p[1] for p in phases]
    colors = [p[3] for p in phases]
    labels = [p[0] for p in phases]
    sublabels = [p[2] for p in phases]
    annotations = [p[4] for p in phases]

    # Connecting line first (behind bars)
    ax.plot(xs, sharpes, "-", color="#475569", lw=1.5, alpha=0.5, zorder=1)

    bars = ax.bar(xs, sharpes, color=colors, edgecolor="white", lw=2, zorder=2, width=0.65)

    # Labels on each bar
    for x, sh, label, sublabel, annot, col in zip(xs, sharpes, labels, sublabels, annotations, colors):
        y_text = sh + 0.08 if sh >= 0 else sh - 0.20
        va = "bottom" if sh >= 0 else "top"
        ax.text(x, y_text, f"Sh = {sh:+.2f}", ha="center", va=va,
                fontsize=12, weight="bold", color=col)
        ax.text(x, -1.0, label, ha="center", fontsize=10.5, weight="bold")
        ax.text(x, -1.45, sublabel, ha="center", fontsize=8.5, style="italic", color="#475569")
        # Annotation badge
        ax.text(x, sh/2 if sh >= 0 else sh/2, annot, ha="center", va="center",
                fontsize=8.5, color="white", weight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=col, alpha=0.85, edgecolor="white"))

    # S&P 500 benchmark line
    ax.axhline(sp_sharpe, color="#94A3B8", ls="--", lw=1.8, alpha=0.7)
    ax.text(4.5, sp_sharpe + 0.04, f"S&P 500 Sharpe = +{sp_sharpe:.2f}",
            ha="right", fontsize=10, color="#475569", style="italic")
    ax.axhline(0, color="black", lw=0.8)

    # The "audit shock" annotation
    ax.annotate("", xy=(1, -0.31), xytext=(0, 1.49),
                arrowprops=dict(arrowstyle="->", color="#EF4444", lw=3, alpha=0.8))
    ax.text(0.5, 0.85, "Survivorship leak\ncaught + corrected", ha="center", fontsize=10,
            color="#EF4444", weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEE2E2", edgecolor="#EF4444"))

    # The "rebuild" annotation
    ax.annotate("", xy=(4, 1.15), xytext=(2, 0.31),
                arrowprops=dict(arrowstyle="->", color="#22C55E", lw=3, alpha=0.8))
    ax.text(3.0, 0.6, "Broad universe rebuild\n+ feature engineering\n+ bug fixes", ha="center", fontsize=10,
            color="#16A34A", weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#DCFCE7", edgecolor="#22C55E"))

    ax.set_ylim(-2.0, 2.0)
    ax.set_xticks([])
    ax.set_ylabel("Long-OOS Sharpe ratio", fontsize=12)
    ax.set_title("The audit journey: +1.49 (leaky) → −0.31 (PIT applied) → +1.15 (honest, final)",
                 fontsize=14, weight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ============================================================
# 5. Robustness battery waterfall
# ============================================================
def fig_robustness_battery(out):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(7.5, 9.6, "Robustness battery — alpha survives every check",
            ha="center", fontsize=15, weight="bold")

    checks = [
        ("Engine sanity gate", "Random / Oracle / Uniform on synthetic panel",
         "3/3 PASS\n(random Sh −0.51, oracle +99, uniform 0)", True),
        ("Feature-shuffle placebo", "Permute feature→ticker mapping; Sh should collapse",
         "+1.15 → −0.94\n(2.1 Sh swing — genuine ML content)", True),
        ("Diebold-Mariano (model selection)", "Lasso vs XGBoost vs NN on MSE + Sharpe",
         "XGBoost > Lasso (p<0.01),\nXGBoost > NN (p<0.05)", True),
        ("Block bootstrap (Sharpe CI)", "6-mo blocks, 10k iterations, P(SR≤0)",
         "P(SR≤0) = 0.0002 long-OOS\nCI: [+0.54, +1.44]", True),
        ("Deflated Sharpe Ratio", "Bailey-LdP 2014, N=25 trials", "DSR = 0.85–0.88\n(comfortably above 0.5)", True),
        ("Carhart 6F momentum control", "FF5 + UMD; is the alpha just momentum?",
         "α RISES to +20.1%/yr (t=+7.4)\nUMD β=−0.43 (momentum-averse)", True),
        ("Cost-grid stress", "10 / 15 / 30 / 50 / 75 bps/side",
         "Sig up to ~50 bps/side\nbeats S&P up to 25 bps/side", True),
        ("Dense k-sweep + bootstrap CIs", "k∈[1,100], plateau-zoom k∈[10,20]",
         "k=20 indistinguishable from\nevery k∈[10,20] (11/11 CIs overlap)", True),
    ]

    y_start = 8.7
    row_h = 0.95
    for i, (check, what, result, passed) in enumerate(checks):
        y = y_start - i * row_h
        # Check name
        bb_check = FancyBboxPatch((0.3, y - 0.35), 4.2, 0.7,
                                   boxstyle="round,pad=0.05,rounding_size=0.1",
                                   facecolor=COL_DATA, edgecolor="white", lw=1.5)
        ax.add_patch(bb_check)
        ax.text(2.4, y, check, ha="center", va="center", fontsize=10.5,
                weight="bold", color="white")
        # What it tests
        ax.text(4.8, y, what, ha="left", va="center", fontsize=9,
                color="#475569", style="italic")
        # Pass badge
        ax.text(10.0, y, "✓", ha="center", va="center", fontsize=24,
                weight="bold", color=COL_MODEL)
        # Result
        ax.text(10.4, y, result, ha="left", va="center", fontsize=9.5,
                color=COL_TEXT)

    # Bottom summary
    bb_sum = FancyBboxPatch((1.0, 0.1), 13, 0.95,
                             boxstyle="round,pad=0.1,rounding_size=0.15",
                             facecolor=COL_OUTPUT, edgecolor="white", lw=2)
    ax.add_patch(bb_sum)
    ax.text(7.5, 0.58, "Headline survives: full-OOS Sharpe +1.15 / FF5 α +18.73%/yr at t=+6.85 (p<0.001)",
            ha="center", va="center", fontsize=13.5, weight="bold", color="white")

    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ============================================================
# 6. Return decomposition
# ============================================================
def fig_return_decomposition(out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7),
                                    gridspec_kw={"width_ratios": [1.2, 1]})

    # ============== Left: cumulative wealth funnel ==============
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis("off")
    ax1.set_title("Cumulative wealth ($1 → $X over 13 yrs)\nhonest decomposition",
                  fontsize=12.5, weight="bold")

    # Three stacked bars showing decomposition
    items = [
        ("S&P 500 passive baseline", 463, 5.63, COL_LIGHT, "#475569",
         "+14.3%/yr CAGR\n(2012-24 bull market)"),
        ("β-hedged pure alpha (uncorrelated)", 614, 7.13, COL_MODEL, "white",
         "+16.5%/yr CAGR\n1.26× S&P, but corr ≈ 0 with S&P\n→ uncorrelated diversifier"),
        ("Gross XGBoost (Mkt-β=+1.5)", 4600, 47.00, COL_DATA, "white",
         "+34.7%/yr CAGR\nML-emergent leverage\n+ alpha + 13yr compounding"),
    ]
    y_pos = [8, 5, 2]
    for (label, cum, val, color, text_col, sub), y in zip(items, y_pos):
        # Width proportional to log-scale of multiple
        w = 1 + 7 * np.log10(val) / np.log10(50)
        bb = FancyBboxPatch((1, y - 0.5), w, 1.0,
                             boxstyle="round,pad=0.05,rounding_size=0.12",
                             facecolor=color, edgecolor="white", lw=2)
        ax1.add_patch(bb)
        # Inside label
        ax1.text(1.2, y + 0.18, label, ha="left", va="center", fontsize=10,
                 weight="bold", color=text_col)
        ax1.text(1.2, y - 0.18, sub, ha="left", va="center", fontsize=7.5,
                 color=text_col, style="italic")
        # Outside value (escape $ so matplotlib doesn't enter math mode)
        ax1.text(w + 1.2, y + 0.18, f"+{cum:,}%", ha="left", va="center",
                 fontsize=12, weight="bold", color=color if color != COL_LIGHT else "#475569")
        ax1.text(w + 1.2, y - 0.18, rf"\${val:.2f} from \$1", ha="left", va="center",
                 fontsize=10, color=color if color != COL_LIGHT else "#475569")

    ax1.text(5, 0.5, "We defend the β-hedged +614% as the deployable headline.",
             ha="center", fontsize=10, weight="bold", color=COL_OUTPUT,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEE2E2", edgecolor=COL_OUTPUT))

    # ============== Right: waterfall of annual return decomposition ==============
    ax2.set_title("Waterfall: how the annual return is built\n"
                  "Gross → minus costs → = Net (decomposed by FF5 source)",
                  fontsize=12, weight="bold")

    # Waterfall steps from gross to net, then FF5 decomposition of net
    # GROSS = +37.6%/yr
    # NET = +34.7%/yr (after -2.9 cost drag)
    # NET = FF5 alpha +18.7 + Mkt-beta contribution +13.5 + SMB +2.5 + small
    steps = [
        ("Gross return", 37.6, COL_DATA, "total"),
        ("Cost drag\n(10 bps × 179% turnover)", -2.9, "#EF4444", "delta"),
        ("Net return", 34.7, COL_MODEL, "total"),
        ("FF5 pure α", 18.7, COL_MODEL, "decomp"),
        ("Mkt-β × Mkt-RF\n(β=+1.5, leveraged market)", 13.5, "#5B92D8", "decomp"),
        ("SMB exposure\n(small-cap premium)", 2.5, "#7CB3E8", "decomp"),
    ]

    ys = np.arange(len(steps))
    for i, (label, val, color, kind) in enumerate(steps):
        bar_color = color
        if kind == "delta" and val < 0:
            bar_color = "#EF4444"  # red for negative
        ax2.barh(len(steps) - 1 - i, val, color=bar_color, edgecolor="white", lw=1.5)
        # Label inside or outside bar
        ax2.text(val + (0.5 if val >= 0 else -0.5), len(steps) - 1 - i,
                 f"{val:+.1f}%/yr", ha="left" if val >= 0 else "right",
                 va="center", fontsize=10.5, weight="bold",
                 color=COL_TEXT)
        # Bar label
        ax2.text(-12, len(steps) - 1 - i, label, ha="right", va="center",
                 fontsize=9.5, color=COL_TEXT)
        # Separator between gross-net section and FF5 decomposition
        if i == 2:
            ax2.axhline(len(steps) - 1 - i - 0.5, color="#94A3B8",
                         ls="--", lw=1, alpha=0.7)
            ax2.text(20, len(steps) - 1 - i - 0.5,
                      "  ↓ Net return decomposed by source", ha="left",
                      va="center", fontsize=8.5, style="italic", color="#475569")

    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_xlim(-15, 45)
    ax2.set_ylim(-0.7, len(steps) - 0.3)
    ax2.set_xlabel("Annualised return contribution (%/yr)", fontsize=10.5)
    ax2.set_yticks([])
    ax2.set_xticks([-5, 0, 10, 20, 30, 40])
    ax2.grid(axis="x", alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    # Bottom text
    ax2.text(15, -0.5,
             "FF5 pure α = ~54% of net return; leveraged-factor exposure = ~46%",
             ha="center", fontsize=10, weight="bold", color=COL_TEXT,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F9FF",
                       edgecolor=COL_DATA, lw=1))

    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("Building 6 presentation diagrams")
    print("=" * 70)

    diagrams = [
        ("pipeline_overview.png", fig_pipeline_overview, "3-workstream architecture + data flow"),
        ("walk_forward_cv.png", fig_walk_forward_cv, "sliding-window CV illustration"),
        ("universe_construction.png", fig_universe_construction, "Sharadar funnel to broad panel"),
        ("audit_journey.png", fig_audit_journey, "+1.49 → −0.31 → +1.15 Sharpe story"),
        ("robustness_battery.png", fig_robustness_battery, "8 robustness checks all passed"),
        ("return_decomposition.png", fig_return_decomposition, "where the +4,600% really comes from"),
    ]

    for fname, fn, desc in diagrams:
        out = OUT_DIR / fname
        fn(out)
        print(f"  [{fname}] {desc}  ->  {out}")

    print(f"\nAll 6 diagrams in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
