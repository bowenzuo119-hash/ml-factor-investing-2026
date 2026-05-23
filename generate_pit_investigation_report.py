"""Generate PIT_INVESTIGATION_REPORT.pdf — deep investigation of the
point-in-time universe filter and its impact on Phase 15.

The story: an audit flagged a survivorship leak in our walk-forward backtest.
Fixing it caused a catastrophic Sharpe collapse (+1.5 -> -0.3). We then ran
a deep investigation to verify whether (a) the fix was correct, (b) the
collapse is a real finding ("most of the alpha was look-ahead bias"), or
(c) a hyperparameter artifact of the smaller training set. This report
walks through every step.

Run with:
    .venv/bin/python generate_pit_investigation_report.py
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


OUT_PATH = Path(__file__).parent / "PIT_INVESTIGATION_REPORT.pdf"

styles = getSampleStyleSheet()
TITLE = ParagraphStyle("Title", parent=styles["Title"], fontSize=22, leading=26,
                       textColor=colors.HexColor("#1F3864"), spaceAfter=2)
SUBTITLE = ParagraphStyle("Sub", parent=styles["Normal"], fontSize=11, leading=14,
                          textColor=colors.HexColor("#666666"), spaceAfter=14)
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16, leading=21,
                    textColor=colors.HexColor("#1F3864"),
                    spaceBefore=18, spaceAfter=8)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, leading=17,
                    textColor=colors.HexColor("#2E5496"),
                    spaceBefore=12, spaceAfter=5)
H3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=11.5, leading=15,
                    textColor=colors.HexColor("#374151"),
                    spaceBefore=8, spaceAfter=3)
BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=10.5, leading=15,
                      spaceAfter=7, alignment=TA_LEFT)
LEAD = ParagraphStyle("Lead", parent=BODY, fontSize=11, leading=16,
                      textColor=colors.HexColor("#222222"), spaceAfter=9)
BULLET = ParagraphStyle("Bul", parent=BODY, fontSize=10.5, leading=15, leftIndent=4,
                        spaceAfter=3)
WARN = ParagraphStyle("Warn", parent=BODY, fontSize=10.5, leading=14,
                      leftIndent=10, rightIndent=10,
                      backColor=colors.HexColor("#FEE2E2"),
                      borderColor=colors.HexColor("#DC2626"),
                      borderWidth=0.6, borderPadding=8,
                      spaceAfter=10)
NOTE = ParagraphStyle("Note", parent=BODY, fontSize=10.5, leading=14,
                      leftIndent=10, rightIndent=10,
                      backColor=colors.HexColor("#FFF8E1"),
                      borderColor=colors.HexColor("#F2C744"),
                      borderWidth=0.6, borderPadding=8,
                      spaceAfter=10)
GOOD = ParagraphStyle("Good", parent=BODY, fontSize=10.5, leading=14,
                      leftIndent=10, rightIndent=10,
                      backColor=colors.HexColor("#D1FAE5"),
                      borderColor=colors.HexColor("#10B981"),
                      borderWidth=0.6, borderPadding=8,
                      spaceAfter=10)
CODE = ParagraphStyle("Code", parent=BODY, fontName="Courier", fontSize=9.5,
                      leading=12, leftIndent=10, rightIndent=10,
                      backColor=colors.HexColor("#F3F4F6"),
                      borderColor=colors.HexColor("#9CA3AF"),
                      borderWidth=0.5, borderPadding=6, spaceAfter=8)


def p(text, style=BODY):
    return Paragraph(text, style)


def bullets(items, style=BULLET):
    return ListFlowable(
        [ListItem(p(t, style), leftIndent=14,
                  bulletColor=colors.HexColor("#2E5496"))
         for t in items],
        bulletType="bullet", bulletFontSize=10, leftIndent=14,
        spaceBefore=2, spaceAfter=8,
    )


def make_table(rows, col_widths=None, header=True, body_align="LEFT"):
    t = Table(rows, colWidths=col_widths, hAlign="LEFT")
    style = [
        ("FONT", (0, 0), (-1, -1), "Helvetica", 9.5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#888888")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CCCCCC")),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3864")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9.5),
        ]
    t.setStyle(TableStyle(style))
    return t


def main() -> int:
    doc = SimpleDocTemplate(
        str(OUT_PATH), pagesize=A4,
        leftMargin=1.8 * cm, rightMargin=1.8 * cm,
        topMargin=1.6 * cm, bottomMargin=1.6 * cm,
        title="Phase 15 PIT Investigation Report",
        author="Nicolas Couto Mota (Person B)",
    )
    story = []
    P = lambda t, s=BODY: story.append(p(t, s))
    SP = lambda h=0.4: story.append(Spacer(1, h * cm))
    PB = lambda: story.append(PageBreak())

    # ============== COVER ==============
    P("Phase 15 PIT Investigation", TITLE)
    P("How fixing one survivorship leak collapsed our headline Sharpe — and what we did to figure out why.<br/>"
      "<font color='#666666'>Person B (Alpha Model) · 2026-05-23 · Branch <font name='Courier'>personb-models</font></font>", SUBTITLE)

    P("Summary", H1)
    P("An external correctness audit identified a survivorship leak in our walk-forward backtest engine: "
      "the engine treated every ticker in the panel as eligible to trade at every rebalance, regardless of "
      "whether that ticker was actually in the S&amp;P 500 on that date. Bowen shipped engine v0.4.0 with a "
      "point-in-time (PIT) universe filter to close the leak. Wired into Phase 15, the filter caused the "
      "XGBoost canonical Sharpe to collapse from <b>+1.50 (long-OOS) / +1.01 (test 2019-2024)</b> down to "
      "<b>-0.31 / -0.29</b> — a striking and uncomfortable result.", LEAD)

    story.append(Paragraph(
        "<b>What we know after the investigation:</b> the engine fix is correct, the PIT membership "
        "function is correct, no off-by-one or ticker-mismatch bugs. A smoking-gun check showed our "
        "panel contained 125 pre-S&amp;P-join return observations for TSLA, 104 for ENPH, 133 for "
        "GNRC — all of which the old (no-PIT) model could see and trade as if they were already index "
        "members. So the survivorship leak was real and material. <b>But</b> we cannot yet conclude "
        "that all of the previous alpha was bogus, because three confounds remain open: "
        "(i) XGBoost was using hyperparameters tuned on the larger 941-ticker panel, almost certainly "
        "overfit on the smaller 56K-row PIT-filtered training set; (ii) LassoCV is silently picking "
        "max regularisation under PIT and predicting near-zero for everything; "
        "(iii) we have not been able to separate the effect of training-data restriction from "
        "trading-universe restriction (Bowen's API applies both together).",
        WARN))

    P("This report walks through every step and ends with the concrete next-action plan.", LEAD)

    P("Headline numbers (XGBoost canonical)", H2)
    story.append(make_table([
        ["Configuration", "Test Sharpe", "Long-OOS Sharpe", "XGBoost IC", "Training rows"],
        ["Phase 15 original (UNKNOWN-bucket bug + no PIT)", "+1.01", "+1.50", "+0.018", "85,527"],
        ["+ Fix 2 only (sector unification, still no PIT)", "+1.38", "+1.22", "+0.018", "85,527"],
        ["+ Fix 1 + Fix 2 (full PIT, engine v0.4.0)", "-0.29", "-0.31", "+0.004", "56,624"],
    ], col_widths=[7.5*cm, 2.4*cm, 2.6*cm, 2.0*cm, 2.5*cm]))

    PB()

    # ============== SECTION 1: background ==============
    P("1. Background — what was the canonical before this?", H1)
    P("Phase 15 (committed at <font name='Courier'>8b2e7a0</font>) was the final canonical of the alpha model. "
      "It built on Phase 14 by extending the training panel back to 2002-04 (the earliest defensible start "
      "given Sharadar SF1 fundamentals coverage), and reported the following:", LEAD)
    story.append(make_table([
        ["Window", "Sharpe (net)", "Ann return", "Max DD", "DSR"],
        ["Test 2019-2024", "+1.01", "+9.47%", "-7.9%", "0.887"],
        ["Long-OOS 2013-2024", "+1.50", "+12.32%", "-7.9%", "0.992"],
        ["FF5 alpha (long-OOS)", "+6.34%/yr", "t=2.44", "p=0.013", "SIG"],
    ], col_widths=[4.8*cm, 3.0*cm, 3.0*cm, 2.8*cm, 3.0*cm]))
    SP()
    P("These were the numbers in the report draft and the PR description merged into <font name='Courier'>main</font>. "
      "They are now superseded by the Phase 15 + Fix 2 + Fix 1 numbers above.", BODY)

    # ============== SECTION 2: the audit ==============
    P("2. The audit and the two fixes", H1)
    P("Before opening the integrated report, we ran a 3-agent correctness audit across feature construction, "
      "walk-forward engine, and metrics. The audit cleared the math (Sharpe annualisation, target alignment, "
      "FF5 regression alignment, HAC standard errors all correct). But it flagged two real issues:", LEAD)

    P("Fix 1 — Survivorship leak (engine side)", H2)
    P("The engine's eligibility check at each rebalance was just <font name='Courier'>pd.notna(returns.at[next_t, asset])</font>: "
      "any ticker with a non-NaN next-month return was tradable. There was no check that the ticker was actually "
      "an S&amp;P 500 member at the rebalance date. Bowen shipped engine v0.4.0 with an optional kwarg "
      "<font name='Courier'>eligible_universe_fn: date -&gt; set[str]</font> that filters both prediction-time eligibility "
      "and training labels to point-in-time index members.",
      BODY)
    P("Bowen's verification: a 2012-2019 RandomModel run traded 726 non-member positions without the filter, "
      "vs 0 with it. The filter does what it says.", BODY)

    P("Fix 2 — Sector-map divergence between Layer 2 and Layer 3 (driver side)", H2)
    P("The engine's Layer-3 (sector-neutral construction) bucketed any ticker missing from "
      "<font name='Courier'>load_sector_map()</font> into a synthetic <font name='Courier'>\"UNKNOWN\"</font> sector. "
      "Because <font name='Courier'>load_sector_map()</font> only contains the ~500 current S&amp;P 500 names, the other "
      "~440 tickers in our panel (delisted, renamed, former members) all collapsed into this UNKNOWN bucket. "
      "Layer 3 then picked 5 longs + 5 shorts from THAT bucket too — treating a random pile of unrelated "
      "delisted/distressed names as if they were a 12th GICS sector.", BODY)
    P("Fix: in the Phase 15 driver, build the sector_map from <font name='Courier'>features[\"sector\"]</font> "
      "(which already applies SIC-code fallback for delisted tickers via <font name='Courier'>factors.get_sector()</font>). "
      "This dissolves the UNKNOWN bucket and routes every ticker to its real industry.", BODY)
    story.append(Paragraph(
        "Effect of Fix 2 alone (no PIT yet): XGBoost test Sharpe went up (+1.01 → +1.38), long-OOS "
        "went down (+1.50 → +1.22). Predictions and IC are unchanged — only the portfolio composition "
        "shifted (10 of 110 positions were no longer pinned to the synthetic UNKNOWN sector).",
        NOTE))

    PB()

    # ============== SECTION 3: the catastrophic PIT run ==============
    P("3. Applying the PIT filter — the catastrophic result", H1)
    P("With Fix 2 in place and a verified-correct engine v0.4.0, we wired the PIT filter into the Phase 15 "
      "driver:", LEAD)
    story.append(Paragraph(
        "from src.data_loader import load_sp500_membership<br/>"
        "def universe_at(date): return set(load_sp500_membership(asof=pd.Timestamp(date)))<br/>"
        "...<br/>"
        "res = run_walk_forward_backtest(..., eligible_universe_fn=universe_at)",
        CODE))
    P("Re-running Phase 15 on the 2002-04 panel produced numbers that were not just lower, but actively "
      "negative:", BODY)
    story.append(make_table([
        ["Model", "Window", "Old Sharpe (no PIT)", "PIT Sharpe", "Δ"],
        ["XGBoost", "Test 2019-2024", "+1.383", "-0.295", "-1.68"],
        ["XGBoost", "Long-OOS", "+1.225", "-0.309", "-1.53"],
        ["NN", "Test 2019-2024", "+0.461", "-0.152", "-0.61"],
        ["Lasso", "Test 2019-2024", "-0.175", "+0.137", "+0.31"],
    ], col_widths=[2.5*cm, 4.5*cm, 3.5*cm, 3.0*cm, 2.0*cm]))
    SP()
    story.append(Paragraph(
        "A Sharpe collapse this large from a single correctness fix is suspicious. Either (a) the fix "
        "exposes that our previous alpha was almost entirely look-ahead bias — the most uncomfortable "
        "possible finding — or (b) something downstream broke when the training data shrank by 33%. "
        "We have not yet definitively distinguished (a) from (b).",
        WARN))

    PB()

    # ============== SECTION 4: deep verification ==============
    P("4. Deep verification — is the PIT setup actually correct?", H1)
    P("Before accepting the collapsed numbers as a real finding, we ran a battery of sanity checks on "
      "<font name='Courier'>load_sp500_membership</font> and our <font name='Courier'>universe_at</font> wrapper. "
      "All five checks passed.", LEAD)

    P("Check 1 — Universe size at each date", H3)
    P("Sampled 52 quarterly dates from 2012 to 2024. Membership size ranged from 497 to 507, exactly "
      "what the S&amp;P 500 should be (~500 names, occasional 501-503 during index reshuffles).", BODY)

    P("Check 2 — Blue-chip stocks always present", H3)
    P("AAPL, MSFT, JNJ, PG, JPM — all five present in all 52 sampled dates. No silent gaps.", BODY)

    P("Check 3 — Late joiners correctly excluded before they joined", H3)
    P("TSLA (joined 2020-12-21): not in membership at 2018-01-01 ✓, in membership at 2024-01-01 ✓. "
      "META/FB (joined 2013-12-23): not in at 2010-01-01 ✓, in at 2024-01-01 ✓. No look-ahead in the "
      "membership data.", BODY)

    P("Check 4 — Ticker overlap with our panel", H3)
    P("At 2019-12-31: membership has 505 names, our panel has 941; 496 names are in both. The 9 "
      "membership-not-in-panel mismatches are formatting differences (BRK.B vs BRK-B, BF.B vs BF-B, ANTM, "
      "UTX, etc.) — not material. The 445 panel-not-in-current-membership names are the legitimate "
      "extras (delisted, renamed, future joiners that hadn't joined yet).", BODY)

    P("Check 5 — Monthly churn", H3)
    P("Sampled monthly 2019-2024. Mean adds/drops = 1.8 per month, max = 8 per month. Matches realistic "
      "S&amp;P 500 reconstitution. No discontinuities.", BODY)
    story.append(Paragraph(
        "<b>Conclusion of Section 4:</b> the PIT setup is verifiably correct. Membership is right, "
        "ticker overlap is fine, no off-by-ones, no silent omissions. The Sharpe collapse is not an "
        "artifact of a buggy PIT filter.",
        GOOD))

    PB()

    # ============== SECTION 5: the smoking gun ==============
    P("5. Smoking gun — what was the old model actually trading?", H1)
    P("If the PIT setup is correct and the Sharpe collapsed, then by elimination the old model was "
      "trading something it shouldn't have been. The most damaging case would be trading future S&amp;P "
      "joiners (like TSLA in 2012) before they joined the index. We checked the panel directly:", LEAD)

    story.append(make_table([
        ["Ticker", "In panel from", "Joined S&P", "Pre-join return obs in panel"],
        ["TSLA", "2010-07-30", "2020-12-21", "125"],
        ["META / FB", "2002-04-30", "2013-12-23", "11–18"],
        ["ENPH", "2012-04-30", "2020-12-22", "104"],
        ["GNRC", "2010-03-31", "2021-04-08", "133"],
        ["NOW", "2002-04-30", "2019-11-21", "101"],
        ["CRM", "2002-04-30", "2008-09-15", "63"],
    ], col_widths=[3.0*cm, 3.0*cm, 3.0*cm, 5.0*cm]))
    SP()
    P("<b>This is the look-ahead bias.</b> All six of these names had monthly return data in our panel "
      "long before they joined the S&amp;P 500. Under the old no-PIT setup the model could see, train on, "
      "and trade these names. Some — like TSLA — had spectacular pre-join returns (>2000% from 2012 to "
      "2020) that the model could have been picking up via momentum / dvol / size features.", BODY)
    P("The PIT-corrected model correctly excludes these pre-join return rows. Under PIT, TSLA appears "
      "in the predictions only from December 2020 onward (48 predictions, 0 of them dated before "
      "the join date). Confirmed empirically.", BODY)

    P("Panel composition of the 445 \"extras\"", H2)
    P("Of the 941 tickers in our panel:", BODY)
    story.append(make_table([
        ["Group", "Count", "Status under PIT"],
        ["In S&P at end of sample (2024-12-31)", "501", "Tradable when they are members"],
        ["Former S&P members (left before now)", "425", "Tradable only while they were members"],
        ["Ever S&P (now OR former)", "926", "subtotal of above two"],
        ["Never an S&P member (in panel anyway)", "15", "NEVER tradable under PIT"],
    ], col_widths=[7.5*cm, 2.5*cm, 7.0*cm]))
    SP()
    P("Almost all our extras (926 of 941) WERE S&amp;P members at some point. Only 15 tickers were never "
      "in the index — most of those are tickers that got renamed/merged into stocks we already have "
      "(Comcast K-shares CMCSK → CMCSA, Compaq CPQ → HPQ, Palm PALM → HPQ, etc.). So the data isn't "
      "garbage — it's data we have for stocks that are legitimately part of the historical S&amp;P 500 "
      "universe at SOME dates, but not all.", BODY)

    PB()

    # ============== SECTION 6: what's broken inside the models ==============
    P("6. What happened inside the models under PIT", H1)
    P("We inspected the actual predictions produced by each of the three models under the PIT-filtered "
      "Phase 15 run. The findings are revealing:", LEAD)

    P("Lasso — predicting near-constant", H2)
    P("Per-stock prediction standard deviation: 0.00024 (essentially flat). "
      "<b>Per-date prediction spread (max − min across the cross-section): 0.0000 median.</b> "
      "On more than half the rebalance dates, Lasso emits the same prediction for every stock. This means "
      "LassoCV is selecting the maximum regularisation parameter, shrinking every coefficient to zero, "
      "and the model returns only its constant intercept. CV's fold sizes are miscalibrated for the smaller "
      "56K-row training set.", BODY)

    P("XGBoost — still learning, but weaker", H2)
    P("Per-stock prediction std: 0.00264 (much smaller than no-PIT). "
      "Per-date prediction spread: 0.020 median (2% spread within a date — reasonable). IC mean: +0.004 "
      "(positive but tiny; was +0.018 under no-PIT). XGBoost is producing a real ranking signal but it's "
      "much weaker than under no-PIT. The hypothesis: the canonical hyperparameters "
      "(<font name='Courier'>depth=3, n_estimators=200, learning_rate=0.0115</font>) were tuned via Optuna "
      "on a 941-ticker / 85K-row panel in Phase 3; with 33% less training data the same hyperparameters "
      "are over-complex and likely overfitting.", BODY)

    P("NN — same weak-signal pattern as XGBoost", H2)
    P("Per-stock std 0.0026, per-date spread 0.012 median, IC mean +0.007 — slightly weaker than XGBoost "
      "but in the same regime. The NN already had near-zero IC pre-PIT (+0.007); under PIT it remains "
      "near-zero. The previous high Sharpe of NN under no-PIT was almost certainly from factor exposure "
      "(implicit small/value tilt amplified by the UNKNOWN bucket), not from real predictive skill.", BODY)

    story.append(Paragraph(
        "<b>What this tells us:</b> the models ARE learning under PIT, they're just learning weaker "
        "patterns on the smaller training set. This is not consistent with \"the engine is broken.\" "
        "It is consistent with \"the model is undertuned for the new data size.\" We cannot yet tell "
        "whether retuning would restore most of the alpha or only a fraction of it.",
        NOTE))

    PB()

    # ============== SECTION 7: the diagnostic gap ==============
    P("7. What we can NOT yet test, and why", H1)
    P("The single most informative diagnostic is: run with the training panel UNRESTRICTED (full 941 "
      "tickers) but the trading universe RESTRICTED to PIT members. If that recovers most of the alpha, "
      "the loss is from training-data restriction (we can fix by retuning or by relaxing the training "
      "filter). If it doesn't, the loss is from trading-universe restriction (the alpha really was in "
      "the now-excluded names).", LEAD)
    P("Bowen's engine v0.4.0 API uses a single kwarg <font name='Courier'>eligible_universe_fn</font> that "
      "applies to BOTH training labels AND prediction-time eligibility. We cannot run the split "
      "diagnostic without a small engine extension — either a second kwarg "
      "<font name='Courier'>eligible_universe_train_fn</font>, or a flag "
      "<font name='Courier'>apply_pit_to_training: bool = True</font>. Asked Bowen for this; "
      "ETA short. Once it lands, this is the first thing we run.", BODY)

    P("8. What we CAN still test (and will, before locking in numbers)", H1)
    P("Tests that don't require engine changes:", BODY)
    story.append(bullets([
        "<b>Quick manual retune:</b> set XGBoost to a much simpler configuration (depth=2, "
        "n_estimators=100) and re-run. If Sharpe partially recovers, hyperparameter overfit is real. "
        "5 min.",
        "<b>Full Optuna retune on PIT panel:</b> 50 trials over depth × n_estimators × learning_rate × "
        "subsample × colsample. Tells us the proper hyperparameters for the new training size. 30 min.",
        "<b>Relaxed-PIT mid-ground (Option B):</b> at each training date d, use \"ever in S&P 500 up to "
        "d\" instead of \"in S&P 500 at d\". This expands the training universe to include legitimate "
        "former members (no look-ahead) without re-introducing the look-ahead leak of TSLA-in-2012. "
        "Likely gives somewhat more training data and may recover some alpha — but it's still strict "
        "PIT at trade time. Requires the same engine kwarg split as test (1).",
        "<b>IC attribution:</b> stratify the OLD predictions by ticker tenure (always-member vs late "
        "joiner vs former member) and compute IC per stratum. Locates where the original IC was "
        "concentrated. Needs the old predictions, which were overwritten — requires a no-PIT re-run "
        "first.",
    ]))

    PB()

    # ============== SECTION 9: candidate explanations ==============
    P("9. Three candidate explanations", H1)
    P("These are not mutually exclusive — the truth is likely a combination.", LEAD)

    P("Hypothesis A — Almost all the previous alpha was look-ahead bias", H2)
    P("Most of the +1.5 long-OOS Sharpe came from trading future S&amp;P joiners (TSLA, ENPH, GNRC, NOW) "
      "before they joined the index. The model was implicitly identifying \"large-cap stocks about to "
      "join the index\" from feature signatures and front-running the index addition. PIT closes this "
      "leak completely and the alpha vanishes because there's no real cross-sectional skill underneath.",
      BODY)
    P("<b>Evidence for:</b> XGBoost IC dropped 75% (0.018 → 0.004) under PIT. Late joiners had 100+ "
      "pre-join return observations in our panel.", BODY)
    P("<b>Evidence against:</b> not all the alpha gap can plausibly be from 6 specific late joiners "
      "out of ~941 tickers. The fix-2-only run (no PIT) showed +1.38 test Sharpe with the same model — "
      "if the alpha were ALL from look-ahead names, removing them shouldn't only halve the Sharpe "
      "from +1.38 to negative, it should produce roughly zero.", BODY)

    P("Hypothesis B — Hyperparameter overfit on the smaller training set", H2)
    P("XGBoost (depth=3, 200 trees, lr=0.0115) was Optuna-tuned on the 85K-row panel. With 56K rows, "
      "this configuration overfits training quickly and generalises poorly. The model's IC drops "
      "because it's memorising training-set noise, not because it has no signal. Retuning would "
      "restore much of the alpha.", BODY)
    P("<b>Evidence for:</b> Lasso is silently broken (predicting near-constant — classic CV-fold-size "
      "issue). Models that retune robustly across data sizes (NN with early stopping) suffered LESS "
      "than the gradient-boosted model.", BODY)
    P("<b>Evidence against:</b> a properly-tuned XGBoost with less data should still find some signal "
      "given the IC dropped 75%, not just 10-20%.", BODY)

    P("Hypothesis C — The training data narrowing changes the learnable signal", H2)
    P("Training on \"only ever-S&P-member-at-date\" excludes stocks during their pre-listing and "
      "post-delisting phases. Cross-sectional relationships in the model's feature space may be "
      "different in the broader (~941-ticker) population than in the strict S&amp;P 500 (~500 names). The "
      "model trained on a homogeneous large-cap universe has less to learn from.", BODY)
    P("<b>Evidence for:</b> the no-PIT panel had stronger cross-sectional dispersion in many features.",
      BODY)
    P("<b>Evidence against:</b> if anything, the relationships SHOULD be cleaner on the homogeneous "
      "S&amp;P 500 universe — that's the practical universe institutional investors actually trade. "
      "Restricting to a sensible universe shouldn't kill the signal entirely.", BODY)

    PB()

    # ============== SECTION 10: next steps ==============
    P("10. Concrete next-step plan", H1)
    P("Before we publish ANY canonical numbers in the report, we will:", LEAD)
    story.append(bullets([
        "<b>(1) Wait for Bowen's engine v0.5.0:</b> separate train-side and predict-side PIT filters. "
        "ETA short.",
        "<b>(2) Quick retune of XGBoost</b> (manual depth=2/n_estimators=100). Decides whether to "
        "commit to full Optuna.",
        "<b>(3) Full Optuna retune of XGBoost on PIT panel.</b> 50 trials, ~30 min.",
        "<b>(4) Run \"train on full panel, trade on PIT only\".</b> Definitive diagnostic for "
        "hypothesis A vs B.",
        "<b>(5) Try relaxed-PIT mid-ground</b> (\"ever in S&P 500 up to date t\"). Gives more training "
        "data without re-introducing look-ahead.",
        "<b>(6) Re-tune Lasso's CV regularisation</b> on the new fold size.",
        "<b>(7) Decide which configuration is the honest canonical for the final report.</b> The "
        "headline numbers in REPORT.md §5 stay at their current values (clearly marked TODO) until "
        "all of the above runs.",
    ]))

    P("11. Implications for the team", H1)
    story.append(bullets([
        "<b>Bowen:</b> engine v0.5.0 with separate train/predict PIT kwargs. The current single-kwarg "
        "design is correct for production but doesn't let us decompose effects for the audit.",
        "<b>Andrea:</b> the regime overlay results in §4 of REPORT.md are unaffected — the overlay "
        "operates on whatever portfolio the engine produces. But the §5 ablation table "
        "(with-overlay vs no-overlay) needs to be re-run on the new canonical once it's locked in.",
        "<b>All:</b> the previous +1.5 long-OOS Sharpe in the report is no longer valid. The honest "
        "number is somewhere between -0.3 and +1.5, and we won't know until the diagnostics complete. "
        "Don't quote the old number externally.",
    ]))

    P("12. Lessons learned", H1)
    story.append(bullets([
        "<b>Always verify the membership-vs-panel overlap before reporting any walk-forward Sharpe.</b> "
        "A 22% overlap gap (panel-but-not-in-current-membership) is enough to silently inflate the "
        "headline by orders of magnitude.",
        "<b>Single-kwarg \"apply to everything\" APIs are fast to ship but hard to debug.</b> Two "
        "separate flags for training-filter and prediction-filter would have let us decompose this "
        "in 5 minutes instead of half a day.",
        "<b>Cross-validation regularisation choices are not invariant to data size.</b> LassoCV's "
        "fold-size assumptions need to be recalibrated when the training panel shrinks materially.",
        "<b>Hyperparameter tuning must be re-run when the training distribution changes.</b> "
        "Otherwise the Optuna search outcomes inherit assumptions of the old panel.",
    ]))

    doc.build(story)
    print(f"Wrote {OUT_PATH}")
    print(f"Size: {OUT_PATH.stat().st_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
