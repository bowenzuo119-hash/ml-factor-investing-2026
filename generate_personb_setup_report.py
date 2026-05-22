"""Generate PERSONB_SETUP_REPORT.pdf - plain-English summary of the setup session.

Run with:
    .venv/bin/python generate_personb_setup_report.py
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


OUT_PATH = Path(__file__).parent / "PERSONB_SETUP_REPORT.pdf"


# ---------------------------------------------------------------------------
# Styles - intentionally generous on line height + spacing so the document
# reads like a one-page-at-a-time briefing, not a wall of text.
# ---------------------------------------------------------------------------

styles = getSampleStyleSheet()

TITLE = ParagraphStyle(
    "TitleBig", parent=styles["Title"], fontSize=22, leading=26,
    textColor=colors.HexColor("#1F3864"), spaceAfter=2,
)
SUBTITLE = ParagraphStyle(
    "Subtitle", parent=styles["Normal"], fontSize=11, leading=14,
    textColor=colors.HexColor("#666666"), spaceAfter=14,
)
H1 = ParagraphStyle(
    "H1Big", parent=styles["Heading1"], fontSize=18, leading=22,
    textColor=colors.HexColor("#1F3864"),
    spaceBefore=18, spaceAfter=8,
)
H2 = ParagraphStyle(
    "H2Big", parent=styles["Heading2"], fontSize=13, leading=17,
    textColor=colors.HexColor("#2E5496"),
    spaceBefore=10, spaceAfter=4,
)
BODY = ParagraphStyle(
    "BodyBig", parent=styles["BodyText"], fontSize=11, leading=16,
    spaceAfter=8, alignment=TA_LEFT,
)
LEAD = ParagraphStyle(
    "Lead", parent=BODY, fontSize=11.5, leading=17,
    textColor=colors.HexColor("#222222"),
    spaceAfter=10,
)
BULLET = ParagraphStyle(
    "Bullet", parent=BODY, fontSize=11, leading=16, leftIndent=4, spaceAfter=4,
)
BOX = ParagraphStyle(
    "Box", parent=BODY, fontSize=10.5, leading=15,
    leftIndent=10, rightIndent=10,
    backColor=colors.HexColor("#FFF8E1"),
    borderColor=colors.HexColor("#F2C744"),
    borderWidth=0.6, borderPadding=8,
    spaceAfter=10, spaceBefore=4,
)
GLOSSARY = ParagraphStyle(
    "Glossary", parent=BODY, fontSize=10.5, leading=14,
    leftIndent=8, rightIndent=8, spaceAfter=3,
)


def p(text: str, style=BODY):
    return Paragraph(text, style)


def bullets(items: list[str], style=BULLET):
    return ListFlowable(
        [ListItem(p(t, style), leftIndent=14, bulletColor=colors.HexColor("#2E5496"))
         for t in items],
        bulletType="bullet",
        bulletFontSize=11,
        leftIndent=14,
        spaceBefore=2,
        spaceAfter=8,
    )


def box(text: str):
    return p(text, BOX)


story = []

# ===========================================================================
# COVER
# ===========================================================================

story += [
    p("What I set up for you today", TITLE),
    p("A plain-English account of every step.<br/>"
      "Project: ml-factor-investing-2026 &nbsp;|&nbsp; "
      "Date: 21 May 2026 &nbsp;|&nbsp; Branch: personb-models",
      SUBTITLE),

    p("The 30-second version", H2),
    p("Your laptop now has everything it needs to run the project. "
      "I installed the tools, copied the stock-price data into the project "
      "folder, and wrote two new code files for you to build on. "
      "All of it has been committed to your branch.", LEAD),
    p("There are a few real gaps that you and Bowen need to talk about "
      "(end of this report). Nothing is broken; some pieces of the original "
      "plan need a small redesign because the data we have is monthly, "
      "but the original plan assumed daily.", LEAD),

    p("Words that show up below", H2),
    p("<b>Feature.</b> A number computed for each stock each month that the "
      "model uses as input. E.g. \"last 12 months\" return\" is a feature.",
      GLOSSARY),
    p("<b>Model.</b> A machine-learning recipe that takes features in and "
      "predicts a stock's next-month return.", GLOSSARY),
    p("<b>Backtest.</b> A simulation: pretend it is the past, let the model "
      "trade, see if it would have made money.", GLOSSARY),
    p("<b>Sector-relative.</b> Compare a stock only against others in the "
      "same industry (tech vs tech, banks vs banks), not against the whole "
      "market.", GLOSSARY),
    p("<b>Branch.</b> A version of the code separate from the team's main "
      "version. Yours is called <b>personb-models</b>. You commit there, "
      "then later open a pull request to merge into main.", GLOSSARY),
]

story += [PageBreak()]

# ===========================================================================
# PART 1: ENVIRONMENT
# ===========================================================================

story += [
    p("Part 1 - Setting up your laptop", H1),
    p("Before today, your laptop could not run the project. I made it so it "
      "can.", LEAD),

    p("1.1 The Python problem", H2),
    p("Your default Python is version 3.13. PyTorch (the deep-learning "
      "library this project uses for the neural-network model) does not yet "
      "have a Mac version that works with Python 3.13. So nothing would "
      "install.", BODY),
    p("Fix: I built a separate, project-only Python 3.12 environment "
      "inside a folder called <b>.venv</b> in the project. From now on, "
      "always run Python through this environment using:", BODY),
    box("$ .venv/bin/python &lt;your-script.py&gt;<br/><br/>"
        "(Not just \"python\", or you will use the system 3.13 which is "
        "missing all the packages.)"),

    p("1.2 Three small library conflicts I fixed", H2),
    p("When I ran <i>pip install -r requirements.txt</i> for the first time, "
      "three things broke. They are all known macOS issues, none are your "
      "fault. I fixed them quietly inside the project so you do not have to "
      "think about them:", BODY),
    bullets([
        "<b>numpy 2.x is too new for the PyTorch we have</b> - I pinned an "
        "older numpy (1.26).",
        "<b>The SHAP library expected the newer numpy</b> - I pinned an "
        "older SHAP (0.46) to match.",
        "<b>Two copies of a low-level library called \"libomp\"</b> get "
        "loaded into memory at the same time (one from Anaconda, one from "
        "the PyTorch package). They fight each other and the neural network "
        "hangs forever. I added a single line at the top of <b>models.py</b> "
        "that tells the system \"it is OK for both to be loaded\".",
    ]),
    p("Net result: every required library imports cleanly. You can verify "
      "by running:", BODY),
    box("$ .venv/bin/python -c \"import torch, xgboost, sklearn, "
        "pandas, hmmlearn, yfinance; print('ok')\""),
]

# ===========================================================================
# PART 2: DATA
# ===========================================================================

story += [
    p("Part 2 - Getting the data in place", H1),
    p("The project needs three kinds of data: stock prices, S&amp;P 500 "
      "membership history, and macro indicators (VIX, interest rates). "
      "Bowen wrote the code that loads all three. I ran it.", LEAD),

    p("2.1 The big CRSP file", H2),
    p("You had four identical copies of a 494 MB stock-price file on your "
      "Desktop. I copied one of them into the project at "
      "<b>data/raw/CRSPData_1925_2022.csv</b>. The other three copies on "
      "your Desktop are safe to delete whenever you want.", BODY),

    p("2.2 The processed caches", H2),
    p("Reading and cleaning a 494 MB CSV every time you run the project "
      "would be painfully slow (about a minute). So Bowen's code reads it "
      "once and writes a much faster format called <b>parquet</b> into "
      "<b>data/processed/</b>. After today, these files exist on your laptop:",
      BODY),
    Table([
        ["File in data/processed/", "Size", "What it is"],
        ["crsp_monthly.parquet", "152 MB",
         "Monthly U.S. stock prices, 1925-2022 (the fast version of the "
         "494 MB CSV)"],
        ["yfinance_monthly.parquet", "0.7 MB",
         "Monthly prices for 2023-2024 (CRSP ends in 2022; Bowen patches "
         "the gap from Yahoo Finance)"],
        ["returns_spliced_2019_2024.parquet", "0.7 MB",
         "The two sources above stitched together into one tidy 72-month x "
         "615-ticker table, ready for your model to read."],
    ], colWidths=[5.5 * cm, 1.5 * cm, 9.5 * cm], hAlign="LEFT", style=TableStyle([
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
    ])),
    Spacer(1, 0.3 * cm),
    p("These cache files are <i>not</i> committed to git (they are too "
      "large, and they can be regenerated from the original CSV at any "
      "time). If you ever delete the data/processed/ folder, just re-run "
      "Bowen's loader script and it will rebuild itself.", BODY),
]

story += [PageBreak()]

# ===========================================================================
# PART 3: THE TWO CODE FILES
# ===========================================================================

story += [
    p("Part 3 - The two files I wrote for you", H1),
    p("These are the two files you own as Person B. They were empty stubs "
      "before today; now they contain working code.", LEAD),

    p("3.1 src/factors.py", H2),
    p("This file's job: turn raw monthly returns into <i>features</i> the "
      "model can learn from. It also tells each stock what industry "
      "(sector) it is in, so we can compare stocks fairly within their "
      "sector.", BODY),
    p("The main function you will call is:", BODY),
    box("build_feature_panel(start=\"2005-01-01\", end=\"2024-12-31\")<br/>"
        "    -&gt; a tidy table of features, one row per (stock, month)"),
    p("Inside, it computes these features:", BODY),
    bullets([
        "<b>mom</b> - 12-month momentum (the return over the past year, "
        "ignoring the most recent month).",
        "<b>rev</b> - short-term reversal (last month's return; stocks that "
        "dropped tend to bounce back).",
        "<b>mvol</b> - monthly volatility (how jumpy the stock has been "
        "over the past 6 months).",
        "<b>ivol</b> - idiosyncratic volatility (how jumpy it is AFTER you "
        "remove what the whole market did).",
        "<b>log_mktcap</b> - the (log of) market value, i.e. how big the "
        "company is. Smaller companies tend to behave differently from "
        "giants.",
    ]),
    p("After it computes each feature, it converts the raw number into a "
      "<b>sector-relative rank</b> between 0 and 1: a value of 0.9 means "
      "the stock is in the top 10% of its sector for that feature, that "
      "month. This is the \"Layer 1\" trick the framework document calls "
      "for.", BODY),

    p("3.2 src/models.py", H2),
    p("This file's job: three different machine-learning recipes that "
      "learn from the features and predict each stock's next-month return.",
      BODY),
    bullets([
        "<b>LassoModel</b> - a simple linear recipe. Your baseline.",
        "<b>XGBoostModel</b> - a more sophisticated recipe based on "
        "decision trees. This is the primary model the framework "
        "document recommends.",
        "<b>NNModel</b> - a small neural network. The secondary recipe; "
        "interesting to compare against XGBoost.",
    ]),
    p("All three speak the same simple language that Bowen's backtest "
      "engine expects:", BODY),
    box("model.fit(features, target)   # learn the pattern<br/>"
        "model.predict(new_features)   # apply what was learned"),
    p("This means you can swap any of the three models in and out of the "
      "backtest without changing anything else.", BODY),
]

story += [PageBreak()]

# ===========================================================================
# PART 4: PROOF IT WORKS
# ===========================================================================

story += [
    p("Part 4 - Proof that everything works together", H1),
    p("I did not just write code; I ran it. Here is what came back:", LEAD),

    p("4.1 factors.py produced a real feature table", H2),
    bullets([
        "Shape: 40,186 rows x 5 columns (one row per stock-month from "
        "2019 to 2024)",
        "615 unique stocks covered",
        "All features successfully ranked within sector",
        "Sector counts at the first date: Financials 92, IT 82, "
        "Industrials 80, etc. - looks like a normal S&amp;P 500 "
        "distribution.",
    ]),

    p("4.2 All three models trained and predicted", H2),
    p("On a small synthetic test where the right answer is a known linear "
      "formula, all three recipes recovered most of the signal:", BODY),
    bullets([
        "Lasso: 0.585 correlation with the truth",
        "XGBoost: 0.632 correlation with the truth",
        "Neural network: 0.572 correlation with the truth",
    ]),

    p("4.3 The full pipeline runs end-to-end", H2),
    p("Features &rarr; XGBoost &rarr; portfolio &rarr; backtest, all wired "
      "together, no errors, finishes in 13 seconds. The backtest's Sharpe "
      "ratio (a measure of how good the strategy is) came out at -0.44.", BODY),
    box("Why is the Sharpe negative? Because the training data we have "
        "today only covers 2019-2024, so the model only sees 24 months of "
        "history before it has to make predictions. That is not enough. "
        "Once we extend the data back to 2005 (one of your next tasks), "
        "the number will tell us something real. Today's number just "
        "confirms the plumbing is connected."),
]

# ===========================================================================
# PART 5: WHAT IS UNFINISHED
# ===========================================================================

story += [
    p("Part 5 - What is unfinished, and why", H1),
    p("Some pieces of the original plan cannot be done yet. None of these "
      "are blockers; they are decisions you and Bowen need to make.", LEAD),

    p("5.1 Three of the eight features need daily data", H2),
    p("The feature spec Bowen wrote assumed we would have <i>daily</i> "
      "stock prices. The data we actually have is <i>monthly</i>. So three "
      "of the eight features (dollar volume, daily volatility, daily "
      "idiosyncratic volatility) cannot be computed exactly as written.",
      BODY),
    p("What I did: I left them as <i>NotImplementedError</i> stubs so it "
      "is obvious they are missing, and I built monthly substitutes for "
      "the volatility ones. These substitutes are defensible in the final "
      "report.", BODY),

    p("5.2 Two features need data we do not have", H2),
    p("Features 7 (Book-to-Market) and 8 (Earnings/Price) need a different "
      "dataset called <i>Compustat</i> that the TA was supposed to share. "
      "The DECISIONS.md file says the deadline for the TA to reply was "
      "yesterday (20 May). Ask Bowen what came of that.", BODY),
    p("If Compustat does not arrive, the project just uses 6 features "
      "instead of 8. That is fine - the academic paper this project is "
      "based on actually says price-based features are the most important.",
      BODY),

    p("5.3 Sector labels are approximate", H2),
    p("There is no clean source of \"what sector was every stock in on "
      "every historical date\". I built a workable substitute: I use the "
      "current sector classification from the team's S&amp;P 500 list, and "
      "for stocks that no longer exist (delisted, merged) I fall back to "
      "their industry code. 11 of 615 stocks ended up labelled "
      "\"Unknown\" - small enough to ignore.", BODY),

    p("5.4 The training data only covers 2019-2024", H2),
    p("The framework wants us to train on 2005-2015, validate on 2016-2018, "
      "and test on 2019-2024. But the frozen returns table I used today "
      "only covers 2019-2024. To get the full history, you just need to "
      "re-run Bowen's freeze script with <b>start=\"2005-01-01\"</b> "
      "instead of <b>start=\"2019-01-01\"</b>. Easy change, just takes a "
      "few minutes of compute. I can do this with you in the next "
      "session.", BODY),
]

story += [PageBreak()]

# ===========================================================================
# PART 6: NEXT STEPS
# ===========================================================================

story += [
    p("Part 6 - What you should do next", H1),
    p("In the order I would tackle them:", LEAD),

    Table([
        ["#", "What", "Why"],
        ["1",
         "Push your branch to GitHub (click \"Publish Branch\" in VS Code) "
         "and open a pull request so Bowen and Person C can see your code.",
         "Gets the scaffold visible to the team."],
        ["2",
         "Re-run the freeze script to cover 2005-2024 (not just 2019-2024).",
         "Gives you enough training history for a real backtest."],
        ["3",
         "Talk to Bowen about the daily-vs-monthly question and the "
         "Compustat question (Part 5 and the next page).",
         "These decisions affect what features you can ship."],
        ["4",
         "Add evaluation metrics: out-of-sample R-squared, and the "
         "Diebold-Mariano test for comparing two models. Both are in the "
         "framework document (section 8).",
         "Required for the final report."],
        ["5",
         "Tune XGBoost hyper-parameters on the 2016-2018 validation "
         "window using Optuna (already installed).",
         "Untuned XGBoost is a baseline only."],
    ], colWidths=[0.7 * cm, 8 * cm, 7.3 * cm], hAlign="LEFT", style=TableStyle([
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
    ])),
]

# ===========================================================================
# PART 7: TALK TO BOWEN
# ===========================================================================

story += [
    p("Part 7 - What to say to Bowen", H1),
    p("Four questions, in plain language:", LEAD),

    p("Q1. About daily vs monthly data", H2),
    p("\"Three of the eight features in your spec need daily prices. We "
      "only have monthly. Are you willing to build a daily-price loader, "
      "or should we officially swap in monthly substitutes (which I have "
      "already implemented) and log it in DECISIONS.md?\"", BODY),

    p("Q2. About Compustat", H2),
    p("\"What happened with the TA's reply about Compustat? The deadline "
      "you set in DECISIONS.md was 20 May. If we did not get access, can "
      "we agree to drop B/M and E/P and ship with 6 features?\"", BODY),

    p("Q3. About the requirements.txt file", H2),
    p("\"On macOS with Anaconda Python 3.12, three things break the "
      "install. I worked around them inside my branch, but they will hit "
      "anyone setting up a fresh machine. Can we add these three lines to "
      "requirements.txt to spare future-us the pain?\"", BODY),
    box("numpy&lt;2<br/>"
        "shap&lt;0.47<br/>"
        "# README note: if NN hangs on Mac, set KMP_DUPLICATE_LIB_OK=TRUE"),

    p("Q4. About the backtest engine", H2),
    p("\"Your engine refits the model at every monthly rebalance, but the "
      "docstring describes a block-refit (refit only every "
      "<i>test_window</i> months). One of those two is what you intended; "
      "which one? It affects how slow the backtest runs.\"", BODY),
]

story += [PageBreak()]

# ===========================================================================
# PART 8: GIT STATE
# ===========================================================================

story += [
    p("Part 8 - Where everything was saved", H1),
    p("Everything I changed is committed to your branch "
      "<b>personb-models</b>, in three separate commits so the history is "
      "easy to read:", LEAD),

    Table([
        ["Commit", "What it contains"],
        ["personb: scaffold factors.py and models.py",
         "The two new code files. About 970 lines of new code."],
        ["docs: add reference materials",
         "The 11 PDFs and Word documents from the documents/ folder "
         "(framework, kickoff plan, Person A spec, GKX paper, etc.)."],
        ["personb: add setup session status report",
         "This PDF you are reading right now, plus the script that "
         "generates it."],
    ], colWidths=[7 * cm, 9 * cm], hAlign="LEFT", style=TableStyle([
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
    ])),
    Spacer(1, 0.3 * cm),
    p("Your <b>personb-models</b> branch is now 3 commits ahead of main. "
      "Click \"Publish Branch\" in VS Code to push it to GitHub, then open "
      "a pull request.", BODY),

    p("If something looks wrong later", H2),
    p("&bull; The two source files are at <b>src/factors.py</b> and "
      "<b>src/models.py</b>. Open them in VS Code; the comments inside "
      "explain each function.", BODY),
    p("&bull; If a model hangs on your Mac, the first thing to check is "
      "that the line <b>os.environ.setdefault(\"KMP_DUPLICATE_LIB_OK\", "
      "\"TRUE\")</b> at the top of <b>models.py</b> is still there.", BODY),
    p("&bull; If you ever need to rebuild the data caches from scratch, "
      "run <b>.venv/bin/python -m src.data_loader</b> from the project "
      "folder and it will regenerate everything.", BODY),
    p("&bull; If you want to read this report again later, it is at "
      "<b>PERSONB_SETUP_REPORT.pdf</b> in the project folder.", BODY),
]


# ---------------------------------------------------------------------------
# Build the PDF
# ---------------------------------------------------------------------------

doc = SimpleDocTemplate(
    str(OUT_PATH),
    pagesize=A4,
    topMargin=1.8 * cm,
    bottomMargin=1.8 * cm,
    leftMargin=2.0 * cm,
    rightMargin=2.0 * cm,
    title="Person B Setup Session - Status Report",
    author="Person B (Nicolas)",
)


def on_page(canvas, doc_):
    canvas.saveState()
    canvas.setFont("Helvetica", 8.5)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.drawString(2.0 * cm, 1.0 * cm,
                      "ml-factor-investing-2026  -  Person B setup  -  21 May 2026")
    canvas.drawRightString(A4[0] - 2.0 * cm, 1.0 * cm,
                           f"Page {doc_.page}")
    canvas.restoreState()


doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
size_kb = OUT_PATH.stat().st_size / 1024
print(f"Wrote {OUT_PATH.name} ({size_kb:.1f} KB)")
