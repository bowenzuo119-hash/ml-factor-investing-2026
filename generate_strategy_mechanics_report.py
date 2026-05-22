"""Generate STRATEGY_MECHANICS_REPORT.pdf - plain-English walkthrough of HOW the
model picks stocks, decides longs vs shorts, sizes positions, and learns over time.

Run with:
    .venv/bin/python generate_strategy_mechanics_report.py
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


OUT_PATH = Path(__file__).parent / "STRATEGY_MECHANICS_REPORT.pdf"


styles = getSampleStyleSheet()
TITLE = ParagraphStyle("Title", parent=styles["Title"], fontSize=22, leading=26,
                       textColor=colors.HexColor("#1F3864"), spaceAfter=2)
SUBTITLE = ParagraphStyle("Sub", parent=styles["Normal"], fontSize=11, leading=14,
                          textColor=colors.HexColor("#666666"), spaceAfter=14)
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=17, leading=22,
                    textColor=colors.HexColor("#1F3864"),
                    spaceBefore=18, spaceAfter=8)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, leading=17,
                    textColor=colors.HexColor("#2E5496"),
                    spaceBefore=10, spaceAfter=4)
BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=11, leading=16,
                      spaceAfter=8, alignment=TA_LEFT)
LEAD = ParagraphStyle("Lead", parent=BODY, fontSize=11.5, leading=17,
                      textColor=colors.HexColor("#222222"), spaceAfter=10)
BULLET = ParagraphStyle("Bul", parent=BODY, fontSize=11, leading=16, leftIndent=4,
                        spaceAfter=4)
BOX = ParagraphStyle("Box", parent=BODY, fontSize=10.5, leading=15,
                     leftIndent=10, rightIndent=10,
                     backColor=colors.HexColor("#FFF8E1"),
                     borderColor=colors.HexColor("#F2C744"),
                     borderWidth=0.6, borderPadding=8,
                     spaceAfter=10, spaceBefore=4)
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


def num_table(rows, col_widths=None):
    t = Table(rows, colWidths=col_widths, hAlign="LEFT")
    style = [
        ("FONT", (0, 0), (-1, -1), "Helvetica", 10),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#888888")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CCCCCC")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3864")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    t.setStyle(TableStyle(style))
    return t


story = []

# =================================================================
# COVER
# =================================================================

story += [
    p("How Our Model Actually Works", TITLE),
    p("A plain-English walkthrough of every step the strategy takes, "
      "every month.<br/>"
      "Project ml-factor-investing-2026 &nbsp;|&nbsp; 2026-05-22",
      SUBTITLE),

    p("The one-paragraph version", H2),
    p("Every month-end we have ~500 stocks (the S&amp;P 500 of that "
      "day). For each stock we compute 8 numbers describing it — its "
      "momentum, its size, its volatility, etc. We feed those numbers "
      "into a machine-learning model (XGBoost) that has been trained "
      "on 10 years of past examples. The model spits back a score for "
      "each stock — a guess about how it'll do next month. We buy the "
      "top-scoring 20% of stocks and sell-short the bottom 20%. "
      "Equal money in each. We hold those positions for one month, "
      "then redo the whole process. The model gets re-trained each "
      "month using the most recent 10 years of data.", LEAD),

    p("That's it. Six paragraphs below explain each step.", BODY),
]

story += [PageBreak()]

# =================================================================
# STEP 1: THE UNIVERSE
# =================================================================

story += [
    p("Step 1. The investable universe", H1),
    p("At the start of every month, we ask: which stocks are even "
      "eligible to buy or short today?", BODY),

    p("Rule: a stock must satisfy ALL of these on the rebalance date:", BODY),
    bullets([
        "<b>It was an S&amp;P 500 member that day.</b> Not today — "
        "<i>that day</i>. So in 2008 we trade Lehman Brothers; in "
        "2020 we trade ExxonMobil even though oil has fallen out of "
        "favour by 2024. This is called \"point-in-time\" membership "
        "and it prevents survivorship bias.",
        "<b>It has a price that day</b> (i.e. it's still trading; "
        "not delisted yet).",
        "<b>It has values for every one of the 8 features.</b> No "
        "missing data — if even one feature is NaN, the stock drops "
        "out for that month.",
        "<b>The model produced a non-NaN score for it.</b>",
    ]),
    p(EXAMPLE_text := (
        "<b>Typical month:</b> ~500 stocks pass all four checks. Some "
        "months a few more (e.g., during the dot-com era, when index "
        "churn was high) and some months a few less. Over the 2005-2024 "
        "window the universe ranges from ~480 to ~510 names per month."
    ), EXAMPLE),
]

# =================================================================
# STEP 2: THE 8 FEATURES
# =================================================================

story += [
    p("Step 2. What we know about each stock", H1),
    p("For every eligible stock, we compute 8 features. Think of these "
      "as the model's input: an 8-dimensional snapshot of the stock "
      "at this moment.", BODY),

    num_table([
        ["#", "Feature", "What it captures", "How it's computed"],
        ["1", "Momentum (mom)",
         "Is the stock on a winning streak?",
         "Cumulative return from 12 months ago to 1 month ago (the spec's 12-1)"],
        ["2", "Short-term reversal (rev)",
         "Did the stock just over-react?",
         "Last month's return"],
        ["3", "Monthly volatility (mvol)",
         "How shaky is the price?",
         "Std dev of monthly returns over last 6 months"],
        ["4", "Idiosyncratic vol (ivol)",
         "How shaky after removing market moves?",
         "Std dev of OLS residuals over last 24 months"],
        ["5", "Log market cap (log_mktcap)",
         "How big is the company?",
         "log(shares outstanding × price)"],
        ["6", "Book-to-market (bm)",
         "Is the company \"cheap\" by accountants' yardstick?",
         "Equity / market cap (Sharadar quarterly)"],
        ["7", "Earnings yield (ep)",
         "Are earnings high relative to price?",
         "TTM net income / market cap (Sharadar trailing 12mo)"],
        ["8", "Dollar volume (dvol)",
         "How liquid is the stock?",
         "21-day trailing average of price × volume (yfinance daily)"],
    ], col_widths=[0.7 * cm, 3.5 * cm, 5.3 * cm, 7 * cm]),
    Spacer(1, 0.3 * cm),

    p("Critical step: <b>sector-relative ranking</b>", H2),
    p("Raw numbers aren't comparable. A utility stock's volatility "
      "looks low next to a tech stock's. So for every feature, we "
      "convert the raw value into a <b>percentile rank within the "
      "stock's sector</b>:", BODY),
    p(EXAMPLE_text := (
        "Apple has momentum = +30% over the past year. Other "
        "Information-Technology stocks that month range from -20% to "
        "+45%. Apple's <i>rank</i> inside IT is 0.85 — top 15% of its "
        "sector. That 0.85 is what the model sees, not the raw 0.30. "
        "This way a +30% return in a sleepy sector and a +30% return "
        "in a hot sector get fair treatment."
    ), EXAMPLE),
    p("Every one of the 8 features goes through this rank step. The "
      "model only ever sees numbers between 0 and 1.", BODY),
]

story += [PageBreak()]

# =================================================================
# STEP 3: PREDICTING THE FUTURE
# =================================================================

story += [
    p("Step 3. The model predicts next-month return", H1),
    p("Now we have, for each eligible stock, 8 ranks in [0, 1]. We "
      "feed those 8 numbers into our trained XGBoost model and it "
      "returns a single number per stock — its <b>predicted return</b> "
      "for the next month.", BODY),

    p("What XGBoost actually does (in 30 seconds)", H2),
    p("XGBoost is a collection of 150 small decision trees that were "
      "trained together. Each tree asks questions like \"is momentum "
      "above 0.7? then add 0.005 to the prediction\" → \"is "
      "book-to-market below 0.3? subtract 0.003\". You add up all 150 "
      "trees' contributions and that's the prediction.", BODY),
    p("It's not magic — it's a fancy weighted average of "
      "if-then-else rules that the training process built up automatically.",
      BODY),

    p(EXAMPLE_text := (
        "<b>Concrete example (made up but realistic-looking):</b><br/><br/>"
        "Stock: NVDA on 2023-06-30.<br/>"
        "Sector-relative ranks: mom 0.95 (high), rev 0.40, mvol 0.80, "
        "ivol 0.75, log_mktcap 0.98, bm 0.10 (low — \"expensive\"), "
        "ep 0.55, dvol 0.99.<br/><br/>"
        "Model output (the prediction): <b>+0.012</b> — \"NVDA should "
        "beat the median S&amp;P 500 stock by about 1.2% next month\".<br/><br/>"
        "Stock: T (AT&amp;T) on the same date.<br/>"
        "Ranks: mom 0.20, rev 0.60, mvol 0.30, ivol 0.25, log_mktcap "
        "0.85, bm 0.75, ep 0.65, dvol 0.70.<br/><br/>"
        "Model output: <b>-0.004</b> — \"AT&amp;T should under-perform "
        "the median by ~0.4% next month\"."
    ), EXAMPLE),
    p("These numbers aren't precise return forecasts — they're more "
      "like a <b>score</b>. What matters is the <b>ranking</b>: which "
      "stocks scored high, which scored low.", BODY),
]

# =================================================================
# STEP 4: WHO TO BUY AND WHO TO SHORT
# =================================================================

story += [
    p("Step 4. Decide who to buy, who to short, who to skip", H1),
    p("We take all ~500 scores and sort them from highest to lowest.", BODY),

    p("Then we draw two cut lines:", BODY),
    bullets([
        "<b>Long basket</b> — the top 20% of scores. We buy those "
        "stocks. ~100 of them.",
        "<b>Short basket</b> — the bottom 20% of scores. We "
        "sell-short those stocks. ~100 of them.",
        "<b>The middle 60%</b> — we hold no position. ~300 stocks "
        "are skipped that month.",
    ]),
    p("So out of ~500 eligible stocks each month, we typically have "
      "<b>~200 active positions</b>: half long, half short.", BODY),

    p(EXAMPLE_text := (
        "<b>Why skip the middle?</b><br/><br/>"
        "The model's confidence is highest at the extremes. If a stock "
        "scored just barely above average, the model's prediction is "
        "basically noise. Buying it would add transaction cost without "
        "adding meaningful information. So we only act on the "
        "high-conviction picks — the most-extreme 40% of names."
    ), EXAMPLE),
]

story += [PageBreak()]

# =================================================================
# STEP 5: SIZING & DOLLAR NEUTRALITY
# =================================================================

story += [
    p("Step 5. How much money goes into each position?", H1),
    p("<b>Equal weight within each basket.</b> If we have 100 longs, "
      "each one gets 1/100 of the long capital. Same for shorts. No "
      "stock is special — Apple gets the same dollar exposure as a "
      "tiny mid-cap that happened to make the top 20%.", BODY),

    p("<b>Dollar-neutral, gross exposure 2x.</b>", H2),
    num_table([
        ["", "Long basket", "Short basket", "Total"],
        ["Number of stocks", "~100", "~100", "~200 positions"],
        ["Weight per stock", "+1.0%", "-1.0%", ""],
        ["Sum of weights", "+100%", "-100%", "0% net market exposure"],
        ["Absolute exposure", "100%", "100%", "200% gross (2x leverage)"],
    ], col_widths=[4 * cm, 3.5 * cm, 3.5 * cm, 5 * cm]),
    Spacer(1, 0.3 * cm),

    p(EXAMPLE_text := (
        "<b>If you started with $1,000:</b><br/>"
        "Long $1,000 in 100 stocks = $10 per long position.<br/>"
        "Short $1,000 in 100 stocks = -$10 per short position.<br/>"
        "Cash position: still $1,000 (since shorting gives you cash).<br/>"
        "<i>You owe $1,000 of stock you've borrowed-and-sold, but you "
        "have $1,000 of stock you bought, plus your original $1,000 "
        "of cash — net assets unchanged. Profit/loss comes from "
        "the spread between the two baskets.</i>"
    ), EXAMPLE),

    p("<b>Why this design?</b>", BODY),
    bullets([
        "<b>Dollar-neutral by construction</b> (note: not exactly "
        "market-neutral -- see DOLLAR_VS_BETA_NEUTRAL.pdf for the full "
        "discussion). If the whole market goes up 5%, both baskets are "
        "up roughly 5%. Our P/L = (long leg's return) - (short leg's "
        "return). We make money only if our top-20% outperforms our "
        "bottom-20%, regardless of which direction the market went -- "
        "though the long leg systematically loads on higher-beta names, "
        "so a small positive net beta (around +0.2 to +0.4) leaks in.",
        "<b>No bet on direction.</b> We're not trying to time the "
        "market. We're betting we can pick winners <i>relative to "
        "losers</i>.",
        "<b>Transaction-cost-aware.</b> Equal-weight is robust to "
        "noisy predictions — you don't get whacked by a single big "
        "position turning bad.",
    ]),
]

# =================================================================
# STEP 6: TRADING + COSTS
# =================================================================

story += [
    p("Step 6. Trade and pay transaction cost", H1),
    p("From last month's portfolio to this month's portfolio, some "
      "stocks moved in (entered top/bottom 20%), some moved out, some "
      "are still there.", BODY),

    p("We compute <b>turnover</b>: how much of the portfolio actually "
      "changed. Then we charge a transaction cost of <b>10 basis points "
      "(0.10%) per dollar traded</b>. This covers the bid-ask spread "
      "plus market impact for liquid S&amp;P 500 names.", BODY),

    p(EXAMPLE_text := (
        "<b>Typical turnover, our strategy:</b> ~1.8 per month "
        "(i.e., we replace ~90% of our positions month over month). "
        "Transaction cost on each rebalance ≈ 1.8 × 10 bps = 0.18% "
        "of NAV. Over a year that's ~2.16% of return given to costs.<br/><br/>"
        "<b>What the cost charges in practice:</b> if last month we "
        "were long AAPL at +1% weight and this month AAPL drops out "
        "of the long basket, we pay 10 bps on the 1% closeout — "
        "0.0001 of NAV. Multiply by 200ish trades and you get ~0.2% "
        "of NAV in cost per month."
    ), EXAMPLE),
]

story += [PageBreak()]

# =================================================================
# STEP 7: HOW THE MODEL LEARNS
# =================================================================

story += [
    p("Step 7. How the model gets its rules in the first place", H1),
    p("So far we've described what happens at a single rebalance date. "
      "But how does the XGBoost model <i>know</i> which feature "
      "combinations predict positive returns? Training.", BODY),

    p("The training procedure", H2),
    bullets([
        "<b>Look back 10 years.</b> If we're rebalancing on 2020-01-31, "
        "the training data is every (stock, month) pair from 2010-01 "
        "to 2019-12. That's ~500 stocks × 120 months ≈ 60,000 "
        "training examples.",
        "<b>For each training example, we have the 8 features as of "
        "that month AND the realised next-month return.</b> The model "
        "learns: \"when features looked like X, the stock returned "
        "Y on average\".",
        "<b>XGBoost fits 150 decision trees to that data.</b> Each "
        "new tree learns to predict the prediction error of the "
        "previous trees, so the ensemble keeps improving.",
        "<b>The fitted model is used to predict at 2020-01-31</b>, "
        "and only that one date. Next month (2020-02-29) the whole "
        "process repeats: re-train on 2010-02 to 2020-01, predict at "
        "2020-02-29.",
    ]),
    p("This is called a <b>walk-forward backtest</b>. The model "
      "never sees data from the future — only the past. Each "
      "rebalance gets a fresh model trained on its own 10-year window.",
      BODY),

    p(EXAMPLE_text := (
        "<b>Why 10 years (and not, say, 3 or 20)?</b><br/>"
        "10 years is roughly one full business cycle: an expansion "
        "plus a recession. Shorter training windows miss recession "
        "patterns; longer ones include data so old it's no longer "
        "representative. 120 months is the conventional choice in "
        "the Gu-Kelly-Xiu (2020) paper this project replicates."
    ), EXAMPLE),
]

# =================================================================
# STEP 8: HOW WE PICKED THE MODEL'S DIALS
# =================================================================

story += [
    p("Step 8. How we set the model's hyperparameters", H1),
    p("XGBoost has lots of dials: how many trees, how deep, how fast "
      "to learn, how much to regularise, etc. Picking those by gut "
      "feel would be cheating (you'd inadvertently tune them to look "
      "good on the test data).", BODY),

    p("So we used a clean three-way split of the timeline:", BODY),
    num_table([
        ["Sample", "Dates", "What we used it for"],
        ["Training",
         "2005-01 → 2015-12",
         "The model fits its parameters on this window"],
        ["Validation",
         "2016-01 → 2018-12",
         "Optuna searched for the best XGBoost hyperparameters on this "
         "window (60 trials, ~85 seconds total). Only this slice "
         "decided the dials."],
        ["Testing",
         "2019-01 → 2024-12",
         "Sacred. Never touched until we're ready to compute the "
         "final report numbers. This is where the +0.53 Sharpe and "
         "+4.86% return live."],
    ], col_widths=[3 * cm, 4 * cm, 9 * cm]),
    Spacer(1, 0.3 * cm),

    p("The hyperparameters Optuna picked:", H2),
    num_table([
        ["Hyperparameter", "Textbook default", "Tuned value", "Effect"],
        ["n_estimators", "300 trees", "150 trees", "Half the size"],
        ["max_depth", "6", "4", "Shallower trees"],
        ["learning_rate", "0.10", "0.015", "Much slower"],
        ["min_child_weight", "1", "15", "Larger leaves"],
        ["reg_alpha (L1)", "0", "0.40", "L1 penalty added"],
        ["reg_lambda (L2)", "1", "2.85", "L2 tightened"],
    ], col_widths=[4.5 * cm, 3 * cm, 3 * cm, 5 * cm]),
    Spacer(1, 0.3 * cm),

    p("The whole pattern: <b>more regularised, slower learning, "
      "smaller forest</b>. This is what you'd expect for a "
      "signal-to-noise-low problem like stock-return prediction. "
      "Letting the model be aggressive would just overfit noise.",
      BODY),
]

story += [PageBreak()]

# =================================================================
# RESULT
# =================================================================

story += [
    p("The bottom line — what the strategy actually delivered", H1),
    p("Out-of-sample, on the 2019-2024 test window the model never "
      "trained on (or touched during hyperparameter tuning):", BODY),

    num_table([
        ["Metric", "Value", "Plain English"],
        ["Net Sharpe", "+0.53",
         "Return per unit of risk. >0 means we beat the risk-free "
         "rate on a risk-adjusted basis."],
        ["Annualised return", "+4.86%",
         "Net of 10 bps transaction cost. Compounded over 5 years = "
         "~+27% cumulative."],
        ["Max drawdown", "-14%",
         "Worst peak-to-trough loss. For comparison the S&amp;P 500 "
         "dropped ~35% in the same window."],
        ["Information coefficient", "+0.0067",
         "How well the model ranks stocks. Tiny but positive across "
         "5 years — most production cross-sectional models live in "
         "the 0.005 - 0.03 range."],
        ["Average turnover", "1.82",
         "We replace ~90% of positions each month. High turnover, "
         "but absorbed by the cost model."],
    ], col_widths=[4 * cm, 2 * cm, 9 * cm]),
    Spacer(1, 0.3 * cm),

    p(EXAMPLE_text := (
        "<b>How this compares to just buying the index.</b><br/>"
        "S&amp;P 500 over the same 2019-2024 window: ~+13% per year "
        "annualised, max drawdown -34% (COVID + 2022). Our strategy: "
        "+4.86%/year, max drawdown -14%. We make less money but with "
        "less than half the maximum loss. A risk-adjusted comparison "
        "(Sharpe) is the fair one — our 0.53 beats the index's ~0.5 "
        "for the same window, while being market-neutral (uncorrelated "
        "with what the market did)."
    ), EXAMPLE),
]

# =================================================================
# Q&A
# =================================================================

story += [
    p("Common questions, answered briefly", H1),

    p("<b>Q. We use 8 features, but the spec says ML can use 94. "
      "Why so few?</b>", BODY),
    p("Two reasons. (1) Data: 86 of GKX's 94 features need fundamentals "
      "we'd have to buy access to. (2) Simplicity: GKX's own analysis "
      "shows the top 10 most-important features are nearly all "
      "price-based (momentum, volatility, size) — adding 80 more "
      "features barely changes Sharpe. The 8 we chose cover the four "
      "categories the paper identifies as load-bearing: trend, "
      "liquidity, volatility, value.", BODY),

    p("<b>Q. Why long-short and not just long the top 20%?</b>", BODY),
    p("Two answers. (1) Market neutrality: shorting hedges out the "
      "\"will the market crash this month?\" risk. (2) The model's "
      "predictions are <i>relative</i> — \"stock A will beat the "
      "median\" is a different claim from \"stock A will earn 5%\". "
      "Going long-short captures the relative-ranking signal cleanly.", BODY),

    p("<b>Q. What happens if a stock in our long basket goes bankrupt?</b>",
      BODY),
    p("It realises the actual return — typically -99% or so — and "
      "we eat the loss. Bankruptcies in our universe (Lehman 2008, "
      "SVB 2023, etc.) are not back-filled, smoothed, or removed. "
      "This is the survivorship rule we enforce in Step 1.", BODY),

    p("<b>Q. Is this strategy actually tradeable in practice?</b>",
      BODY),
    p("In principle yes — S&amp;P 500 stocks are very liquid and 10 bps "
      "is a realistic cost assumption. The 1.8x monthly turnover is "
      "high but not extreme. The bigger practical concern is that "
      "factor strategies decay over time as more investors discover "
      "them, so historical Sharpe is not a guarantee.", BODY),

    p("<b>Q. The model thinks about each stock independently. Doesn't "
      "that miss interactions?</b>", BODY),
    p("Yes — that's why XGBoost is preferred over Lasso. Decision "
      "trees naturally pick up interactions like \"high momentum AND "
      "low volatility predicts higher returns than either alone\". "
      "Lasso can't, because it adds features linearly. This is exactly "
      "what we observed in our numbers: XGBoost +0.53 Sharpe vs Lasso "
      "+0.04 Sharpe on the same features.", BODY),

    p("<b>Q. What's NOT in the model?</b>", BODY),
    bullets([
        "Earnings announcement timing (we don't know when the next "
        "earnings call is).",
        "Analyst forecasts (we'd need a separate paid feed).",
        "Macro indicators (those go into Person C's separate regime "
        "model that scales our position sizes; not an input to the "
        "stock picks themselves).",
        "Anything about the market direction or VIX level. The model "
        "is time-blind and sector-blind by construction.",
    ]),
]


# =================================================================
# Build
# =================================================================

doc = SimpleDocTemplate(
    str(OUT_PATH),
    pagesize=A4,
    topMargin=1.6 * cm,
    bottomMargin=1.6 * cm,
    leftMargin=2.0 * cm,
    rightMargin=2.0 * cm,
    title="How Our Model Works",
    author="Person B (Nicolas)",
)


def on_page(canvas, doc_):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.drawString(2.0 * cm, 0.9 * cm,
                      "ml-factor-investing-2026 — Strategy mechanics — 2026-05-22")
    canvas.drawRightString(A4[0] - 2.0 * cm, 0.9 * cm,
                           f"Page {doc_.page}")
    canvas.restoreState()


doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
size_kb = OUT_PATH.stat().st_size / 1024
print(f"Wrote {OUT_PATH.name} ({size_kb:.1f} KB)")
