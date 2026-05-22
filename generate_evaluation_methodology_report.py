"""Generate EVALUATION_METHODOLOGY_REPORT.pdf - how we judge whether our model
is good. Plain-English walkthrough of the benchmark problem, the metrics,
and the time-window robustness checks.

Run with:
    .venv/bin/python generate_evaluation_methodology_report.py
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    ListFlowable, ListItem, PageBreak, Paragraph,
    SimpleDocTemplate, Spacer, Table, TableStyle,
)


OUT_PATH = Path(__file__).parent / "EVALUATION_METHODOLOGY_REPORT.pdf"

styles = getSampleStyleSheet()
TITLE = ParagraphStyle("Title", parent=styles["Title"], fontSize=21, leading=25,
                       textColor=colors.HexColor("#1F3864"), spaceAfter=2)
SUBTITLE = ParagraphStyle("Sub", parent=styles["Normal"], fontSize=11, leading=14,
                          textColor=colors.HexColor("#666666"), spaceAfter=14)
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16, leading=21,
                    textColor=colors.HexColor("#1F3864"),
                    spaceBefore=16, spaceAfter=8)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, leading=17,
                    textColor=colors.HexColor("#2E5496"),
                    spaceBefore=10, spaceAfter=4)
BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=11, leading=16,
                      spaceAfter=8, alignment=TA_LEFT)
LEAD = ParagraphStyle("Lead", parent=BODY, fontSize=11.5, leading=17,
                      textColor=colors.HexColor("#222222"), spaceAfter=10)
BULLET = ParagraphStyle("Bul", parent=BODY, fontSize=11, leading=16, leftIndent=4,
                        spaceAfter=4)
NOTE = ParagraphStyle("Note", parent=BODY, fontSize=10.5, leading=14,
                      leftIndent=10, rightIndent=10,
                      backColor=colors.HexColor("#FFF8E1"),
                      borderColor=colors.HexColor("#F2C744"),
                      borderWidth=0.6, borderPadding=8,
                      spaceAfter=10)
EXAMPLE = ParagraphStyle("Ex", parent=BODY, fontSize=10.5, leading=15,
                         leftIndent=10, rightIndent=10,
                         backColor=colors.HexColor("#EFF6FF"),
                         borderColor=colors.HexColor("#3B82F6"),
                         borderWidth=0.6, borderPadding=8,
                         spaceAfter=10)


def p(text, style=BODY):
    return Paragraph(text, style)


def bullets(items, style=BULLET):
    return ListFlowable(
        [ListItem(p(t, style), leftIndent=14,
                  bulletColor=colors.HexColor("#2E5496"))
         for t in items],
        bulletType="bullet", bulletFontSize=11, leftIndent=14,
        spaceBefore=2, spaceAfter=8,
    )


def make_table(rows, col_widths=None, header=True):
    t = Table(rows, colWidths=col_widths, hAlign="LEFT")
    style = [
        ("FONT", (0, 0), (-1, -1), "Helvetica", 10),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#888888")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CCCCCC")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3864")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 10),
        ]
    t.setStyle(TableStyle(style))
    return t


story = []

# ==================================================================
# COVER
# ==================================================================

story += [
    p("How we judge whether our model is good", TITLE),
    p("The benchmarking problem in long-short equity, what metrics we use, "
      "and how we test against the time-window cherry-picking critique.<br/>"
      "Project ml-factor-investing-2026 &nbsp;|&nbsp; 2026-05-22", SUBTITLE),

    p("The short answer", H2),
    p("A long-short equity strategy is a fundamentally different "
      "investment than the S&amp;P 500. Comparing their raw returns is the "
      "wrong question. We evaluate at three levels: <b>(1) does the model "
      "predict returns better than a coin flip</b>, <b>(2) does the strategy "
      "make money on a risk-adjusted basis</b>, and <b>(3) is the result "
      "robust to which 5-year window we picked</b>. Bootstrap CIs, deflated "
      "Sharpe, and Fama-French regressions answer each question in turn. "
      "Our canonical model (Phase 8) passes (1) and (2) cleanly on the "
      "10-year window; the more stringent factor-adjusted test for (2) is "
      "borderline.", LEAD),
]

# ==================================================================
# 1. THE BENCHMARK PROBLEM
# ==================================================================

story += [
    p("1. Why we can't just compare to the S&P 500's return", H1),

    p("1.1 What a long-short portfolio actually is", H2),
    p("Our strategy holds equal-dollar amounts long and short. If you put "
      "$1,000 into the strategy:", BODY),
    bullets([
        "$1,000 of cash sits in your account",
        "You borrow $1,000 of stocks and sell them (the short leg)",
        "You then own $1,000 of cash + $1,000 of bought stocks (long leg) - "
        "$1,000 of stocks owed (short leg). Net asset value = $1,000 of cash.",
        "The strategy's P/L over the month is (long-leg return) - "
        "(short-leg return), expressed as a percentage of NAV. The market's "
        "overall direction barely matters by construction.",
    ]),
    p(EXAMPLE_text := (
        "<b>If S&amp;P 500 rises 10% in a month and our long leg is up 12% "
        "while our short leg is up 9%:</b><br/>"
        "Long P/L = +12% × $1000 = +$120<br/>"
        "Short P/L = -9% × $1000 = -$90  (we LOSE when shorted stocks rise)<br/>"
        "Net P/L = +$30 = +3% on $1,000 NAV.<br/><br/>"
        "<b>If S&amp;P 500 falls 10% and long is down 12%, short is down 9%:</b><br/>"
        "Long P/L = -12% × $1000 = -$120<br/>"
        "Short P/L = +9% × $1000 = +$90  (we GAIN when shorted stocks fall)<br/>"
        "Net P/L = -$30 = -3% on NAV.<br/><br/>"
        "Symmetric. The strategy doesn't care about market direction - it "
        "cares about the SPREAD between winners and losers."
    ), EXAMPLE),

    p("1.2 Why comparing to S&P 500 returns is the wrong question", H2),
    p("The S&amp;P 500 in 2019-2024 returned ~13% per year. Our long-short "
      "strategy returned ~5%. A naive reader would say \"the strategy "
      "underperformed.\" That's wrong, for two reasons:", BODY),
    bullets([
        "<b>The strategies take different amounts of risk.</b> The S&amp;P "
        "500 has annualised volatility ~18% and a max drawdown of ~34% in "
        "2020. Our strategy has volatility ~9% and max drawdown ~9%. We "
        "took roughly half the risk and made roughly a third of the return. "
        "What matters is return PER UNIT of risk.",
        "<b>The strategies have different exposures.</b> The S&amp;P 500 "
        "is a pure long bet on the US market. Our strategy is approximately "
        "market-neutral (net beta of +0.05 to +0.10). They're not "
        "comparable as investments any more than a 30-year bond is "
        "comparable to a venture capital fund.",
    ]),

    p(NOTE_text := (
        "<b>The correct comparison is risk-adjusted.</b> An investor could "
        "lever our +0.66 Sharpe strategy 2x to roughly match the S&amp;P "
        "500's annualised return -- with comparable volatility but a much "
        "shorter maximum drawdown. Or they could blend our market-neutral "
        "return with their existing portfolio to add a non-correlated "
        "return stream. Neither of those uses the S&amp;P 500's raw return "
        "as the benchmark."
    ), NOTE),
]

story += [PageBreak()]

# ==================================================================
# 2. THE THREE LEVELS OF EVALUATION
# ==================================================================

story += [
    p("2. The three levels of evaluation we use (Framework section 8.2)", H1),
    p("The Project Framework prescribes a 3-level evaluation, each "
      "answering a different question.", LEAD),

    p("2.1 Level 1: Did the model predict returns better than chance?", H2),
    p("Question: ignoring the portfolio entirely, can our XGBoost model "
      "rank stocks better than a random number generator?", BODY),
    p("<b>Two metrics:</b>", BODY),
    bullets([
        "<b>Out-of-sample R² (Gu-Kelly-Xiu version):</b> "
        "1 - SUM((y - pred)²) / SUM(y²). Numerator is squared forecast "
        "error; denominator is total squared return. R² > 0 means we beat "
        "predicting zero. For individual stock returns, GKX (2020) reports "
        "OOS R² in the 0.1-0.6% range across all their models. Ours: "
        "Phase 8 = -2.0% (negative because our tree model makes large "
        "predictions; squared error punishes that even when ranking is "
        "good).",
        "<b>Information Coefficient (IC):</b> per-month rank correlation "
        "between predicted scores and realised returns, averaged across "
        "months. IC > 0 means we rank stocks correctly on average. "
        "Production cross-sectional models typically score 0.01-0.03. Ours: "
        "Phase 8 = +0.0123. Small but positive.",
    ]),

    p("2.2 Level 2: Did the strategy generate alpha?", H2),
    p("Question: when we actually trade on the predictions with realistic "
      "costs, does the portfolio make risk-adjusted money?", BODY),
    p("<b>Primary metric: Sharpe ratio.</b> "
      "(Annualised return / annualised volatility). Captures return per "
      "unit of risk. A Sharpe of 1.0 is considered very good for a "
      "long-short equity strategy; 0.5-0.8 is solid. Ours: <b>+0.66</b> on "
      "the canonical 2019-2024 test window, <b>+0.79</b> on the 2015-2024 "
      "long-OOS window.", BODY),
    p("<b>Secondary metrics:</b>", BODY),
    bullets([
        "<b>Maximum drawdown:</b> worst peak-to-trough loss. Lower is "
        "better. S&amp;P 500's 2019-2024 max drawdown was -34%; ours is -9%.",
        "<b>Calmar ratio:</b> annualised return / |max drawdown|. Another "
        "risk-adjusted measure that punishes deep drawdowns more harshly "
        "than Sharpe does.",
        "<b>Average monthly turnover:</b> measures transaction cost drag. "
        "Ours: ~1.77 (i.e. we replace ~88% of positions each month). High "
        "but absorbed by the 10 bps cost assumption.",
    ]),

    p("2.3 Level 3: Does the regime overlay add value?", H2),
    p("Question: do market-condition-aware leverage adjustments improve "
      "the strategy beyond running it at constant leverage?", BODY),
    p("Not yet evaluated in this project because Person C's GMM regime "
      "model isn't wired through Bowen's backtest engine yet (the "
      "<font face='Courier'>regime_fn</font> parameter exists but the "
      "current canonical run uses it as None). Once Layer 3 is wired, "
      "we compare Sharpe with and without the regime overlay on the "
      "same predictions.", BODY),
]

story += [PageBreak()]

# ==================================================================
# 3. THE TIME-WINDOW PROBLEM
# ==================================================================

story += [
    p("3. The time-window critique and how we handle it", H1),

    p("3.1 The COVID problem (and 2022 and 2020 banking crisis)", H2),
    p("The framework says \"test window 2019-2024.\" That window contains "
      "three unique market events:", BODY),
    bullets([
        "<b>Q1 2020 COVID crash:</b> S&amp;P 500 fell ~34% in 5 weeks. "
        "Volatility spiked to crisis levels. Cross-sectional patterns "
        "broke down (everything sold off together).",
        "<b>2022 bear market:</b> the largest tech selloff since the "
        "dot-com era. Growth dramatically underperformed value -- the "
        "opposite of the 2015-2019 regime.",
        "<b>March 2023 banking crisis:</b> SVB / Signature / First Republic "
        "collapsed. Financial sector specific.",
    ]),
    p("If our strategy works in this 5-year window, a skeptic could argue "
      "we got lucky -- maybe our model happens to do well during these "
      "specific events but would fail in a different regime. We address "
      "this two ways:", BODY),

    p("3.2 Robustness check #1: longer out-of-sample window", H2),
    p("Even within the data we have (2005-2024), we report metrics on "
      "<b>both</b> the framework's 5-year window AND a 10-year window:",
      BODY),
    make_table([
        ["Window", "Years", "n_months", "Includes", "Sharpe (canonical)"],
        ["2019-2024 (framework test)", "5", "60",
         "COVID + 2022 + SVB", "+0.66"],
        ["2015-2024 (long-OOS)", "10", "120",
         "2015 China selloff + 2016 elections + COVID + 2022 + SVB", "+0.79"],
    ], col_widths=[5.5 * cm, 1 * cm, 1.5 * cm, 5 * cm, 2 * cm]),
    Spacer(1, 0.3 * cm),
    p("Both windows give positive Sharpe. The long-OOS Sharpe (+0.79) is "
      "actually <b>HIGHER</b> than the short-window one (+0.66) -- the "
      "model performed even better on 2015-2018 than on 2019-2024. So the "
      "framework's 5-year test window is, if anything, a slightly hostile "
      "subset.", BODY),

    p("3.3 Robustness check #2: block bootstrap confidence intervals", H2),
    p("The framework section 8.3 explicitly prescribes block bootstrap. "
      "We take the 120 monthly returns of the long-OOS window, resample "
      "them in 6-month blocks (preserving short-horizon autocorrelation), "
      "and recompute the Sharpe 10,000 times. The 5-95% percentile of "
      "those 10,000 Sharpes is the bootstrap CI:", BODY),
    make_table([
        ["Window", "Observed Sharpe", "5-95% Bootstrap CI", "P(SR ≤ 0)"],
        ["2019-2024", "+0.69", "[+0.05, +1.09]", "3.7%"],
        ["2015-2024", "+0.79", "[+0.36, +1.13]", "0.14%"],
    ], col_widths=[3 * cm, 2.5 * cm, 4 * cm, 2 * cm]),
    Spacer(1, 0.3 * cm),
    p("On both windows the CI excludes zero -- the strategy is "
      "statistically distinguishable from random. On the 10-year window, "
      "the probability that the true Sharpe is ≤ 0 is 0.14% -- a very "
      "strong rejection.", BODY),

    p("3.4 Robustness check #3: deflated Sharpe (multi-test correction)", H2),
    p("Bailey &amp; López de Prado (2014) point out that we tried multiple "
      "model configurations along the way (Phase 1 → 1.5 → 2 → 3b → 3c "
      "→ 8). Each variant is a hypothesis test. The MAXIMUM Sharpe across "
      "6 attempts is biased upward -- some of it is just luck of "
      "picking the best of 6 noisy estimates.", BODY),
    p("The deflated Sharpe corrects for this:", BODY),
    bullets([
        "Compute expected maximum Sharpe given N=6 random configurations "
        "with the observed Sharpe-spread variance (Bonferroni-style)",
        "Subtract that expected-max from the observed Sharpe",
        "Penalise further for non-normal moments (skewness, kurtosis) of "
        "the return series",
        "Convert to a probability via the standard normal CDF -- the DSR.",
    ]),
    p("DSR > 0.95 means the strategy is significant at 5% AFTER adjusting "
      "for variant search:", BODY),
    make_table([
        ["Window", "DSR (Phase 8)", "Threshold", "Verdict"],
        ["2019-2024", "0.85", "0.95", "Borderline (not significant)"],
        ["2015-2024", "0.96", "0.95", "Significant ✓"],
    ], col_widths=[3 * cm, 2.5 * cm, 2 * cm, 5 * cm]),
    Spacer(1, 0.3 * cm),
    p("The 10-year DSR clears the threshold. The 5-year window is "
      "borderline but the bootstrap test on that window separately "
      "shows P(SR ≤ 0) < 5%.", BODY),
]

story += [PageBreak()]

# ==================================================================
# 4. FACTOR ADJUSTMENT
# ==================================================================

story += [
    p("4. The hardest test: is the Sharpe genuine skill, or factor exposure?", H1),

    p("4.1 The Fama-French question", H2),
    p("A long-short strategy can produce a positive Sharpe in three ways:",
      BODY),
    bullets([
        "<b>Genuine cross-sectional skill</b> -- the model picks winners "
        "and losers based on real, repeatable patterns.",
        "<b>Factor-tilt accidents</b> -- the strategy happens to short "
        "value stocks and long growth stocks during a period when growth "
        "dominated value. Looks like skill in-sample but is really "
        "captured by Fama-French factor premia.",
        "<b>Survivorship / data bias</b> -- the strategy worked because we "
        "didn't properly handle delisted stocks, banking crises, etc.",
    ]),
    p("We control for (3) by construction: CRSP price data includes delisted "
      "stocks with their bankruptcy returns; Sharadar SF1 covers delisted "
      "fundamentals; the fja05680 membership table is point-in-time. We "
      "test for (2) by regressing strategy returns on Fama-French factors:",
      BODY),

    p("4.2 The regression", H2),
    p("For each monthly portfolio return r_p,t in the test window, we run:",
      BODY),
    p("r_p,t - r_f,t = α + β₁(Mkt-RF)_t + β₂(SMB)_t + β₃(HML)_t + ε_t  "
      "(FF3)<br/>"
      "or, adding profitability (RMW) and investment (CMA):<br/>"
      "r_p,t - r_f,t = α + β₁(Mkt-RF)_t + β₂(SMB)_t + β₃(HML)_t + "
      "β₄(RMW)_t + β₅(CMA)_t + ε_t  (FF5)", BODY),
    p("The factor returns come from Ken French's data library (free, "
      "monthly, since 1926). Newey-West HAC standard errors handle "
      "autocorrelation. We focus on:", BODY),
    bullets([
        "<b>α (alpha):</b> the part of the strategy's return NOT explained "
        "by factor exposures. If α is significantly positive, we have "
        "skill beyond factor replication.",
        "<b>β coefficients on each factor:</b> tells us WHICH factors the "
        "strategy is effectively replicating.",
        "<b>R² of the regression:</b> how much of our return variance is "
        "explained by factors.",
    ]),

    p("4.3 The results — the most consequential numbers in the project", H2),
    make_table([
        ["", "FF3 (test)", "FF3 (long-OOS)", "FF5 (test)", "FF5 (long-OOS)"],
        ["α (annualised)", "+1.95%", "+2.88%", "+1.72%", "+2.68%"],
        ["α t-stat", "+0.46", "+1.17", "+0.44", "+1.14"],
        ["α p-value", "0.65", "0.24", "0.66", "0.25"],
        ["Mkt-RF β", "+0.11", "+0.10", "+0.16 ***", "+0.14 ***"],
        ["SMB β", "0.00", "+0.07", "-0.03", "+0.06"],
        ["HML β", "-0.17 ***", "-0.13 **", "-0.29 ***", "-0.26 ***"],
        ["RMW β", "", "", "-0.12", "-0.05"],
        ["CMA β", "", "", "+0.34", "+0.30 *"],
        ["R²", "0.12", "0.10", "0.19", "0.16"],
    ], col_widths=[3.5 * cm, 2.5 * cm, 3 * cm, 2.5 * cm, 3 * cm]),
    Spacer(1, 0.3 * cm),

    p(NOTE_text := (
        "<b>The honest story:</b> alpha is positive (+1.7 to +2.9% per year) "
        "but NOT statistically significant after factor adjustment (t-stat "
        "between 0.4 and 1.2 on all four specifications). The dominant "
        "factor exposure is the value factor: HML loading of -0.17 to "
        "-0.29 with t-stats of -2.7 to -4.6. The strategy is persistently "
        "<b>short value</b>, and value underperformed growth massively in "
        "2015-2024.<br/><br/>"
        "Interpretation: the ML model has empirically discovered the "
        "growth-over-value premium of this period, plus a smaller residual "
        "skill component. A passive HML-short ETF would have captured "
        "most of the same Sharpe at much lower complexity."
    ), NOTE),
    p("This is the kind of finding methodologically-careful financial-ML "
      "papers routinely report. It's not a failure -- it's a correctly-"
      "diagnosed limitation. Future work (Bowen's Layer 3 sector-neutral "
      "construction, or an explicit factor-hedge) could push alpha higher "
      "by removing the factor tilts.", BODY),
]

story += [PageBreak()]

# ==================================================================
# 5. SUMMARY TABLE
# ==================================================================

story += [
    p("5. The honest scorecard", H1),
    p("Compact summary of every claim we make, with the evidence behind it:",
      BODY),
    make_table([
        ["Claim", "Test", "Result", "Verdict"],
        ["The model predicts better than random",
         "IC > 0", "+0.0123", "weak but positive ✓"],
        ["The strategy makes money risk-adjusted",
         "Sharpe (5yr)", "+0.66", "decent ✓"],
        ["Holds on a longer window",
         "Sharpe (10yr)", "+0.79", "better, not worse ✓"],
        ["Statistically distinguishable from random",
         "Bootstrap CI excludes 0", "[+0.36, +1.13]",
         "P(SR<=0)=0.14% ✓"],
        ["Survives multi-test correction",
         "Deflated Sharpe > 0.95 (10yr)", "DSR = 0.96",
         "passes ✓"],
        ["Same on 5yr window",
         "DSR > 0.95 (5yr)", "DSR = 0.85",
         "borderline ✗"],
        ["Survives factor adjustment",
         "FF5 alpha significant", "t = 1.14, ns",
         "fails ✗"],
        ["Drawdown smaller than market",
         "max DD", "-9%", "vs S&P 500's -34% ✓"],
        ["Empirically market-neutral",
         "β to S&P 500", "+0.09 (t=1.99)",
         "small, borderline ✓"],
    ], col_widths=[4 * cm, 3.5 * cm, 3.5 * cm, 4 * cm]),
    Spacer(1, 0.3 * cm),
    p("Six green checks, one borderline, two open issues. The two open "
      "issues are the FF5 alpha (not significant after factor adjustment, "
      "suggesting much of the Sharpe is replicable by a HML-short) and "
      "the 5-year-window DSR (borderline). Both are honest limitations to "
      "disclose in the final report, not bugs.", BODY),
]

# ==================================================================
# Build
# ==================================================================

doc = SimpleDocTemplate(
    str(OUT_PATH), pagesize=A4,
    topMargin=1.6 * cm, bottomMargin=1.6 * cm,
    leftMargin=2.0 * cm, rightMargin=2.0 * cm,
    title="Evaluation methodology",
    author="Person B (Nicolas)",
)


def on_page(canvas, doc_):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.drawString(2.0 * cm, 0.9 * cm,
                      "ml-factor-investing-2026 — evaluation methodology — 2026-05-22")
    canvas.drawRightString(A4[0] - 2.0 * cm, 0.9 * cm,
                           f"Page {doc_.page}")
    canvas.restoreState()


doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
size_kb = OUT_PATH.stat().st_size / 1024
print(f"Wrote {OUT_PATH.name} ({size_kb:.1f} KB)")
