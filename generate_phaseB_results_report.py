"""Generate PHASE_B_RESULTS_REPORT.pdf - the cross-phase comparison report.

Reads results/{01_first_real_backtest, 01b_with_value_factors,
02_sector_relative_target}/ and produces a single PDF with:

  * narrative summary of each phase
  * combined cumulative-returns + drawdown plots across all 3 phases x 3 models
  * IC, Sharpe, and R^2 bar charts for cross-phase comparison
  * side-by-side metric tables
  * decisions taken and what's next

Run with:
    .venv/bin/python generate_phaseB_results_report.py
"""
from __future__ import annotations

import io
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
OUT_PATH = ROOT / "PHASE_B_RESULTS_REPORT.pdf"

# Test window per Framework section 7.2
TEST_START = pd.Timestamp("2019-01-01")
TEST_END = pd.Timestamp("2024-12-31")

PHASES = [
    ("01_first_real_backtest", "Phase 1", "5 features, raw target"),
    ("01b_with_value_factors", "Phase 1.5", "+ B/M and E/P (7 features)"),
    ("02_sector_relative_target", "Phase 2", "+ sector-relative target"),
    ("03b_tuned_xgboost", "Phase 3b", "tuned XGBoost (7 feat)"),
    ("03c_tuned_xgboost_8features", "Phase 3c", "tuned + dvol (8 feat)"),
    ("08_extended_fundamentals", "Phase 8", "+ ROE/ROA/D/E/AG/Acc (13 feat)"),
]

MODELS = ["Lasso", "XGBoost", "NN"]

# Distinct colours per (phase, model). Same model uses same colour across
# phases but with shading; same phase uses related colours across models.
PHASE_COLORS = {
    "Phase 1":   "#9CA3AF",  # grey
    "Phase 1.5": "#3B82F6",  # blue
    "Phase 2":   "#10B981",  # green
    "Phase 3b":  "#DC2626",  # red - tuned XGBoost, 7 features
    "Phase 3c":  "#7C3AED",  # purple - tuned + 8 features
    "Phase 8":   "#F59E0B",  # amber - canonical: tuned + 13 features
}
MODEL_LINESTYLES = {"Lasso": ":", "XGBoost": "-", "NN": "--"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_phase(phase_dir: str) -> dict:
    """Return {model_name: BacktestResult} for a phase, plus its metrics frame."""
    d = RESULTS / phase_dir
    with open(d / "per_model_results.pkl", "rb") as f:
        models = pickle.load(f)
    metrics = pd.read_parquet(d / "metrics.parquet")
    return {"models": models, "metrics": metrics}


def test_window(s: pd.Series) -> pd.Series:
    return s[(s.index >= TEST_START) & (s.index <= TEST_END)]


# ---------------------------------------------------------------------------
# Plot helpers - matplotlib figs returned as PNG bytes for reportlab
# ---------------------------------------------------------------------------

def fig_to_image(fig, *, width=16 * cm, height=9 * cm):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=width, height=height)


def plot_combined_cumulative(phases_data) -> Image:
    fig, ax = plt.subplots(figsize=(11, 6))
    for phase_dir, phase_label, _ in PHASES:
        models = phases_data[phase_dir]["models"]
        for m in MODELS:
            if m not in models:
                continue
            rets = test_window(models[m].portfolio_returns)
            cum = (1.0 + rets).cumprod() - 1.0
            ax.plot(
                cum.index, cum.values * 100,
                color=PHASE_COLORS[phase_label],
                linestyle=MODEL_LINESTYLES[m],
                linewidth=1.4 if m == "XGBoost" else 1.0,
                alpha=0.95 if m == "XGBoost" else 0.75,
                label=f"{phase_label} - {m}",
            )
    ax.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax.set_title("Cumulative net return on the 2019-2024 test window — 9 strategies",
                 fontsize=12, weight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return (%)")
    ax.legend(loc="upper left", ncol=3, fontsize=7.5, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig_to_image(fig)


def plot_combined_drawdowns(phases_data) -> Image:
    fig, ax = plt.subplots(figsize=(11, 6))
    for phase_dir, phase_label, _ in PHASES:
        models = phases_data[phase_dir]["models"]
        for m in MODELS:
            if m not in models:
                continue
            rets = test_window(models[m].portfolio_returns)
            wealth = (1.0 + rets).cumprod()
            dd = (wealth / wealth.cummax() - 1.0) * 100
            ax.plot(
                dd.index, dd.values,
                color=PHASE_COLORS[phase_label],
                linestyle=MODEL_LINESTYLES[m],
                linewidth=1.4 if m == "XGBoost" else 1.0,
                alpha=0.95 if m == "XGBoost" else 0.75,
                label=f"{phase_label} - {m}",
            )
    ax.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax.set_title("Drawdown on the 2019-2024 test window",
                 fontsize=12, weight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left", ncol=3, fontsize=7.5, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig_to_image(fig)


def bar_chart(metric_key: str, title: str, ylabel: str,
              phases_data, fmt_percent=False) -> Image:
    """Grouped bar chart: x-axis = model, group = phase."""
    fig, ax = plt.subplots(figsize=(10, 5))
    n_models = len(MODELS)
    n_phases = len(PHASES)
    x = np.arange(n_models)
    width = 0.8 / n_phases

    for i, (phase_dir, phase_label, _) in enumerate(PHASES):
        df = phases_data[phase_dir]["metrics"]
        df = df[df["window"] == "test_only"].set_index("model")
        vals = [df.loc[m, metric_key] if m in df.index else 0 for m in MODELS]
        if fmt_percent:
            vals = [v * 100 for v in vals]
        bars = ax.bar(x + (i - n_phases / 2 + 0.5) * width, vals,
                      width, label=phase_label,
                      color=PHASE_COLORS[phase_label],
                      edgecolor="black", linewidth=0.4)
        for b, v in zip(bars, vals):
            txt = f"{v:+.3f}" if not fmt_percent else f"{v:+.1f}%"
            ax.text(b.get_x() + b.get_width() / 2, v,
                    txt, ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig_to_image(fig)


# ---------------------------------------------------------------------------
# Build the document
# ---------------------------------------------------------------------------

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=18, leading=22,
                    textColor=colors.HexColor("#1F3864"),
                    spaceBefore=14, spaceAfter=8)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, leading=17,
                    textColor=colors.HexColor("#2E5496"),
                    spaceBefore=8, spaceAfter=4)
BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=11,
                      leading=16, spaceAfter=8, alignment=TA_LEFT)
LEAD = ParagraphStyle("Lead", parent=BODY, fontSize=11.5, leading=17,
                      textColor=colors.HexColor("#222222"), spaceAfter=10)
CAPTION = ParagraphStyle("Cap", parent=BODY, fontSize=9, leading=12,
                         textColor=colors.HexColor("#666666"),
                         spaceAfter=12, alignment=TA_LEFT)
NOTE = ParagraphStyle("Note", parent=BODY, fontSize=10.5, leading=14,
                      leftIndent=10, rightIndent=10,
                      backColor=colors.HexColor("#FFF8E1"),
                      borderColor=colors.HexColor("#F2C744"),
                      borderWidth=0.6, borderPadding=8,
                      spaceAfter=10)


def metric_table(phases_data, metric_keys: list[tuple[str, str, str]]) -> Table:
    """metric_keys = list of (column_name, display_label, fmt)."""
    header = ["Metric"] + [
        f"{lbl}\n({desc})" for _, lbl, desc in PHASES
    ]
    rows = [header]
    for model in MODELS:
        rows.append([f"--- {model} ---"] + [""] * len(PHASES))
        for col, label, fmt in metric_keys:
            row = [f"  {label}"]
            for phase_dir, _, _ in PHASES:
                df = phases_data[phase_dir]["metrics"]
                df = df[(df["window"] == "test_only")
                        & (df["model"] == model)]
                if not df.empty:
                    v = df.iloc[0][col]
                    row.append(fmt.format(v))
                else:
                    row.append("-")
            rows.append(row)
    t = Table(rows, hAlign="LEFT",
              colWidths=[3.5 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm,
                         2.1 * cm, 2.1 * cm])
    style = [
        ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#888888")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CCCCCC")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3864")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        # Bold the model section headers
        *[("BACKGROUND", (0, 1 + i * (len(metric_keys) + 1)),
           (-1, 1 + i * (len(metric_keys) + 1)),
           colors.HexColor("#E8EEF7"))
          for i in range(len(MODELS))],
        *[("FONT", (0, 1 + i * (len(metric_keys) + 1)),
           (-1, 1 + i * (len(metric_keys) + 1)), "Helvetica-Bold", 9.5)
          for i in range(len(MODELS))],
    ]
    t.setStyle(TableStyle(style))
    return t


def main() -> int:
    print(f"Loading phases from {RESULTS}...")
    phases_data = {p[0]: load_phase(p[0]) for p in PHASES}
    for phase_dir, phase_label, _ in PHASES:
        n = len(phases_data[phase_dir]["models"])
        print(f"  {phase_label}: loaded {n} models")

    story = []

    # ---- cover ------------------------------------------------------
    story += [
        Paragraph("Phase B Results — three iterations on the alpha model",
                  ParagraphStyle("T", parent=styles["Title"], fontSize=20,
                                 textColor=colors.HexColor("#1F3864"))),
        Paragraph("Project ml-factor-investing-2026 &nbsp;|&nbsp; "
                  "Person B (Alpha Model) &nbsp;|&nbsp; 2026-05-22",
                  ParagraphStyle("Sub", parent=styles["Normal"],
                                 fontSize=11, leading=14,
                                 textColor=colors.HexColor("#666666"),
                                 spaceAfter=14)),
        Paragraph(
            "Three end-to-end backtests on the Framework's 2005-2024 sample "
            "with three different setups. Test window 2019-2024 in every chart.",
            LEAD,
        ),
        Paragraph(
            "<b>The headline.</b> Tuned XGBoost on the 13-feature panel "
            "(Phase 8), now running on backtest engine v0.3.0 with "
            "block-gated refit: <b>+0.94 net Sharpe</b>, "
            "<b>+7.9% annualised</b>, max drawdown <b>-5.5%</b> on the "
            "2019-2024 test window. v0.3.0 refits only at the start of "
            "each test_window block (not every period as the buggy v0.2.0 "
            "engine did), which removed a layer of recency noise-fitting "
            "and gave a 41% Sharpe lift over the same predictions. "
            "Deflated Sharpe now clears 0.95 on both 5yr and 10yr OOS "
            "windows; FF5 alpha = +3.83%/yr is borderline significant "
            "(t=1.94, p=0.055). See Section 5 for the per-phase narrative.",
            NOTE,
        ),
    ]

    # ---- chart 1: cumulative returns ---------------------------------
    story += [
        Paragraph("1. Cumulative net return — all 9 strategies", H1),
        plot_combined_cumulative(phases_data),
        Paragraph(
            "Solid = XGBoost (the framework's primary model). "
            "Dashed = NN. Dotted = Lasso. "
            "Grey = Phase 1, blue = Phase 1.5, green = Phase 2. "
            "Phase 1.5 XGBoost (solid blue) is the cleanest upward "
            "curve from late 2020 onward; Phase 2 (solid green) keeps "
            "most of that gain but with a flatter slope.",
            CAPTION,
        ),
    ]

    # ---- chart 2: drawdowns ------------------------------------------
    story += [
        Paragraph("2. Drawdown — same nine strategies", H1),
        plot_combined_drawdowns(phases_data),
        Paragraph(
            "Lower (more negative) is worse. The 2020 COVID drawdown "
            "hits everyone; the 2022-2023 drawdown is the more "
            "interesting test of the cross-sectional signal. Phase 2 "
            "XGBoost (solid green) has the shallowest drawdown of the "
            "tree-based strategies.",
            CAPTION,
        ),
    ]

    # ---- chart 3: Sharpe bar ----------------------------------------
    story += [
        PageBreak(),
        Paragraph("3. Key metrics, side-by-side", H1),
        Paragraph(
            "Each chart is grouped by model; the three bars in each group "
            "are Phase 1, 1.5, and 2.",
            BODY,
        ),
        Paragraph("3.1 Net Sharpe ratio (annualised)", H2),
        bar_chart("sharpe_net", "Net Sharpe on test window 2019-2024",
                  "Sharpe", phases_data),
        Paragraph(
            "XGBoost's Sharpe jumps from -0.03 (Phase 1) to +0.56 (Phase 1.5) "
            "when B/M and E/P are added, then drops to +0.43 in Phase 2 "
            "when the model has to predict sector-relative returns. Lasso "
            "stays effectively flat through all three phases; NN peaks at "
            "Phase 1 then degrades.",
            CAPTION,
        ),
    ]

    story += [
        Paragraph("3.2 Information coefficient (rank correlation, higher = better)", H2),
        bar_chart("ic_mean", "IC mean (Spearman, test window 2019-2024)",
                  "IC mean", phases_data),
        Paragraph(
            "Only XGBoost achieves a meaningfully positive IC in any phase. "
            "Phase 2 (sector-relative target) gives XGBoost a small further "
            "lift: +0.0062 → +0.0079 (~+30% relative). For the two "
            "underperforming models the IC is negative throughout — they "
            "are not learning useful cross-sectional signal yet.",
            CAPTION,
        ),
    ]

    story += [
        Paragraph("3.3 Annualised return — net of 10 bps transaction cost", H2),
        bar_chart("ann_return_net",
                  "Annualised net return on the test window",
                  "Annualised return", phases_data, fmt_percent=True),
        Paragraph(
            "Phase 1.5 XGBoost is the only configuration that produces "
            "annualised returns above 4%. Phase 2 retains most of the "
            "return at +4.0%; Phase 1 was effectively flat.",
            CAPTION,
        ),
    ]

    story += [
        Paragraph("3.4 Max drawdown (less negative = better)", H2),
        bar_chart("max_drawdown",
                  "Max drawdown on the test window",
                  "Max drawdown", phases_data, fmt_percent=True),
        Paragraph(
            "Phase 2's sector-relative target reduces drawdowns for "
            "XGBoost (−16% → −14%) and NN (−20% → −16%) — fewer "
            "concentrated sector bets means less violent peak-to-trough "
            "losses. Lasso barely moves.",
            CAPTION,
        ),
    ]

    # ---- big metrics table -------------------------------------------
    story += [
        PageBreak(),
        Paragraph("4. Full side-by-side metric table", H1),
        Paragraph(
            "Every metric the project reports for Person B's deliverables, "
            "for each model, across the three phases. Test window 2019-2024.",
            BODY,
        ),
        metric_table(phases_data, [
            ("oos_r2_vs_zero", "OOS R² vs zero",     "{:+.4f}"),
            ("oos_r2_vs_mean", "OOS R² vs mean",     "{:+.4f}"),
            ("ic_mean",        "IC mean",            "{:+.4f}"),
            ("ic_ir",          "IC IR (mean/std)",   "{:+.3f}"),
            ("sharpe_net",     "Net Sharpe",         "{:+.3f}"),
            ("ann_return_net", "Annualised return",  "{:+.2%}"),
            ("max_drawdown",   "Max drawdown",       "{:+.2%}"),
            ("avg_turnover",   "Avg turnover",       "{:.2f}"),
        ]),
        Spacer(1, 0.3 * cm),
        Paragraph(
            "<b>How to read the table.</b> Each model's eight metrics are "
            "shown across three phases. A negative R² vs zero is normal for "
            "individual-stock forecasts (Gu-Kelly-Xiu 2020 reports OOS R² in "
            "the +0.1% to +0.6% range across all their models). IC is the "
            "more reliable model-quality measure for a cross-sectional ranker.",
            CAPTION,
        ),
    ]

    # ---- narrative ---------------------------------------------------
    story += [
        PageBreak(),
        Paragraph("5. What each phase actually changed", H1),
        Paragraph(
            "Each phase is one focused experiment, with one variable "
            "moved at a time. All other knobs (training window, "
            "transaction cost, train/val/test split, RNG seed) are "
            "identical across the three phases.",
            BODY,
        ),

        Paragraph("Phase 1 — baseline (5 features, raw target)", H2),
        Paragraph(
            "Features: momentum (12-1), short-term reversal, monthly "
            "volatility, idiosyncratic volatility (24-month proxy), log "
            "market cap. Target: raw next-period return. This was the "
            "first walk-forward run on the full 2005-2024 sample.",
            BODY,
        ),
        Paragraph(
            "<b>Result:</b> all three models close to zero Sharpe. R² in "
            "the GKX ballpark (+0.7% to +0.9% for Lasso and NN against "
            "the zero benchmark). XGBoost slightly worse — surprising "
            "given the framework calls it the primary model.",
            BODY,
        ),

        Paragraph("Phase 1.5 — add B/M and E/P (Sharadar SF1)", H2),
        Paragraph(
            "Sourced point-in-time quarterly fundamentals from Sharadar "
            "SF1 via Bowen's new <font face='Courier'>load_fundamentals</font>. "
            "Computed B/M (book equity / market cap) from the ARQ "
            "dimension and E/P (TTM net income / market cap) from the "
            "ART dimension. Same 120-month sliding training window.",
            BODY,
        ),
        Paragraph(
            "<b>Result:</b> XGBoost's Sharpe explodes from −0.03 to "
            "+0.56. Lasso barely moves (L1 likely zeroes most of the new "
            "coefficients). NN drifts slightly down. R² vs zero gets "
            "worse for XGBoost (−0.003 → −0.027) while IC and Sharpe "
            "improve — the classic Gu-Kelly-Xiu phenomenon: tree "
            "models predict with higher variance once given richer "
            "features, so squared-error R² punishes them even though "
            "rank ordering is better.",
            BODY,
        ),

        Paragraph("Phase 2 — sector-relative target (Layer 2)", H2),
        Paragraph(
            "Same 7 features. Target changes from raw next-month return "
            "to (next-month return) minus (per-(date, sector) mean). "
            "This is Layer 2 of the Framework's three-layer "
            "sector-neutrality stack (section 3.2).",
            BODY,
        ),
        Paragraph(
            "<b>Result:</b> XGBoost's IC and IC IR improve "
            "(+0.0062 → +0.0079, +0.090 → +0.109). Drawdowns shrink "
            "for XGBoost and NN. But Sharpe drops for all three models "
            "— the backtest still uses GLOBAL top/bottom-decile "
            "selection, not per-sector top-k. A sector-neutral model "
            "feeding a sector-blind portfolio gives up the profitable "
            "sector tilts that raw-target models captured.",
            BODY,
        ),
        Paragraph(
            "<b>Decision:</b> keep <font face='Courier'>target_kind=\"raw\""
            "</font> as the canonical default. Layer 2 is in the code, "
            "ready to switch on the moment Bowen implements Layer 3 "
            "(sector-neutral portfolio, <font face='Courier'>k_per_sector"
            "</font> already exists in <font face='Courier'>"
            "backtest.py</font> as a warn-only stub).",
            NOTE,
        ),
    ]

    # ---- what next ---------------------------------------------------
    story += [
        Paragraph("6. What still needs doing", H1),
        Paragraph(
            "<b>Phase 3 — XGBoost hyperparameter tuning (next).</b> "
            "The current XGBoost is on out-of-the-box defaults "
            "(n_estimators=300, max_depth=4, lr=0.05). Tuning on the "
            "2016-2018 validation window via Optuna should give a "
            "further Sharpe lift. Will use validation OOS R² as the "
            "search objective and pin the chosen hyperparameters as the "
            "new XGBoostModel defaults.",
            BODY,
        ),
        Paragraph(
            "<b>Phase 4 — Diebold-Mariano model comparison.</b> "
            "Frameworks section 8.4 prescribes the adapted DM test "
            "for pairwise model ranking. Output: a 3x3 significance "
            "table for the final report.",
            BODY,
        ),
        Paragraph(
            "<b>Pending from Bowen — sector-neutral portfolio.</b> "
            "Once <font face='Courier'>k_per_sector</font> is wired "
            "through the backtest loop, re-run Phase 2 and re-evaluate "
            "Layer 2's contribution. Should recover the Sharpe drop.",
            BODY,
        ),
        Paragraph(
            "<b>Stretch — lag features and dynamics features.</b> "
            "Framework section 3.5. Add x_{t-1}, x_{t-2} and "
            "engineered columns like 3-month momentum change. Skip if "
            "time runs out before Week 5 report-writing.",
            BODY,
        ),
    ]

    # ---- where it lives ----------------------------------------------
    story += [
        Paragraph("7. Where everything lives", H1),
        Paragraph(
            "All commits are on the <font face='Courier'>personb-models"
            "</font> branch on GitHub and pushed to origin. Bowen and "
            "Person C can pull at any time.",
            BODY,
        ),
        Paragraph("&bull; Code: "
                  "<font face='Courier'>src/factors.py</font>, "
                  "<font face='Courier'>src/models.py</font>, "
                  "<font face='Courier'>src/metrics.py</font>", BODY),
        Paragraph("&bull; Drivers: "
                  "<font face='Courier'>notebooks/personb/01_first_real_backtest.py"
                  "</font>, "
                  "<font face='Courier'>01b_with_value_factors.py</font>, "
                  "<font face='Courier'>02_sector_relative_target.py</font>",
                  BODY),
        Paragraph("&bull; Artefacts: "
                  "<font face='Courier'>results/01_first_real_backtest/</font>, "
                  "<font face='Courier'>01b_with_value_factors/</font>, "
                  "<font face='Courier'>02_sector_relative_target/</font>",
                  BODY),
        Paragraph("&bull; Per-phase rationales in DECISIONS.md (entries "
                  "dated 2026-05-22).", BODY),
    ]

    # ---- build -------------------------------------------------------
    doc = SimpleDocTemplate(
        str(OUT_PATH),
        pagesize=A4,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        title="Phase B Results - cross-phase comparison",
        author="Person B (Nicolas)",
    )

    def on_page(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#888888"))
        canvas.drawString(1.8 * cm, 0.8 * cm,
                          "ml-factor-investing-2026 — Person B Phase B results — 2026-05-22")
        canvas.drawRightString(A4[0] - 1.8 * cm, 0.8 * cm,
                               f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"\nWrote {OUT_PATH.name} ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
