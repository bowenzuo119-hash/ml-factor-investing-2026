"""Generate PERSONB_SETUP_REPORT.pdf summarising the 2026-05-21 setup session.

One-off. Run with:
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
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


OUT_PATH = Path(__file__).parent / "PERSONB_SETUP_REPORT.pdf"


styles = getSampleStyleSheet()
H1 = ParagraphStyle(
    "H1", parent=styles["Heading1"], fontSize=16, spaceAfter=6, spaceBefore=12,
    textColor=colors.HexColor("#1F3864"),
)
H2 = ParagraphStyle(
    "H2", parent=styles["Heading2"], fontSize=12, spaceAfter=4, spaceBefore=8,
    textColor=colors.HexColor("#2E5496"),
)
H3 = ParagraphStyle(
    "H3", parent=styles["Heading3"], fontSize=10.5, spaceAfter=3, spaceBefore=6,
    textColor=colors.HexColor("#4472C4"), fontName="Helvetica-Bold",
)
BODY = ParagraphStyle(
    "Body", parent=styles["BodyText"], fontSize=9.5, leading=13,
    spaceAfter=4, alignment=TA_LEFT,
)
CODE = ParagraphStyle(
    "Code", parent=styles["Code"], fontSize=8.5, leading=11,
    backColor=colors.HexColor("#F2F2F2"),
    borderColor=colors.HexColor("#CCCCCC"),
    borderWidth=0.5, borderPadding=4,
    leftIndent=6, rightIndent=6, spaceAfter=6,
)
NOTE = ParagraphStyle(
    "Note", parent=BODY, fontSize=9, leading=12,
    leftIndent=10, rightIndent=10,
    backColor=colors.HexColor("#FFF8E1"),
    borderColor=colors.HexColor("#F2C744"), borderWidth=0.5,
    borderPadding=5, spaceAfter=6,
)


def p(text: str, style=BODY):
    return Paragraph(text, style)


def code(text: str):
    # reportlab Paragraphs don't honour newlines literally; use <br/>
    return Paragraph(text.replace("\n", "<br/>"), CODE)


def table(rows, col_widths=None, header=True):
    t = Table(rows, colWidths=col_widths, hAlign="LEFT")
    style = [
        ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
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
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
        ]
    t.setStyle(TableStyle(style))
    return t


story = []

# --- Title --------------------------------------------------------------
story += [
    p("Person B Setup Session - Status Report",
      ParagraphStyle("Title", parent=styles["Title"], fontSize=18,
                     textColor=colors.HexColor("#1F3864"))),
    p("<b>Project:</b> ml-factor-investing-2026 &nbsp;&nbsp; "
      "<b>Date:</b> 2026-05-21 &nbsp;&nbsp; "
      "<b>Branch:</b> personb-models &nbsp;&nbsp; "
      "<b>Owner:</b> Person B (Alpha Model)", BODY),
    Spacer(1, 0.3 * cm),
    p("This report documents exactly what was set up, what code was added, "
      "what data is in place, what is still blocked, and what to discuss "
      "with Bowen (Person A). Read sections 7 and 8 for the action items.",
      BODY),
    Spacer(1, 0.3 * cm),
]

# --- 1. Summary --------------------------------------------------------
story += [
    p("1. Summary of work completed", H1),
    table([
        ["#", "Item", "Status"],
        ["1", "Local Python 3.12 venv with all project dependencies", "DONE"],
        ["2", "Three macOS dependency footguns identified + fixed", "DONE"],
        ["3", "Git branch personb-models created off main", "DONE"],
        ["4", "CRSP CSV placed in data/raw/", "DONE"],
        ["5", "Person A's data pipeline run end-to-end; all caches built",
            "DONE"],
        ["6", "Wide returns panel frozen for 2019-2024",
            "DONE"],
        ["7", "src/factors.py scaffolded (feature panel builder)", "DONE"],
        ["8", "src/models.py scaffolded (Lasso, XGBoost, NN)", "DONE"],
        ["9", "End-to-end smoke test against the backtest engine", "DONE"],
        ["10", "Nothing yet committed to personb-models", "PENDING"],
    ], col_widths=[0.8 * cm, 11 * cm, 2 * cm]),
    Spacer(1, 0.2 * cm),
]

# --- 2. Environment ----------------------------------------------------
story += [
    p("2. Environment setup", H1),
    p("<b>Python interpreter:</b> Anaconda Python 3.12 at "
      "<font face='Courier'>/opt/anaconda3/bin/python3.12</font>.", BODY),
    p("<b>Why not 3.13:</b> The system default <font face='Courier'>"
      "python3</font> is 3.13, but PyTorch has no Python 3.13 wheel for "
      "macOS x86_64 yet. The README says \"Python 3.11+\" so 3.12 is "
      "compliant.", BODY),
    p("<b>venv creation:</b>", BODY),
    code("/opt/anaconda3/bin/python3.12 -m venv .venv<br/>"
         ".venv/bin/pip install --upgrade pip<br/>"
         ".venv/bin/pip install -r requirements.txt"),
    p("<b>Three dependency footguns fixed in this venv:</b>", H2),
    table([
        ["Issue", "Symptom", "Fix"],
        ["pip pulled numpy 2.3 by default",
         "Torch 2.2.2 wheel was built for numpy 1.x ABI; first call into "
         "torch.nn imports crashes with _ARRAY_API not found",
         ".venv/bin/pip install \"numpy<2\""],
        ["shap 0.51 requires numpy>=2",
         "After downgrading numpy, shap import breaks the same way",
         ".venv/bin/pip install \"shap<0.47\""],
        ["Two libomp libraries loaded simultaneously",
         "Anaconda ships libomp; the torch wheel ships its own libomp. "
         "Both load into one process and PyTorch training deadlocks. "
         "Lasso+XGBoost finished in 0.1s each; NN hung forever.",
         "Set environment variable KMP_DUPLICATE_LIB_OK=TRUE before any "
         "import of torch. Done inside src/models.py at module top via "
         "os.environ.setdefault."],
    ], col_widths=[3.5 * cm, 7 * cm, 6 * cm]),
    Spacer(1, 0.2 * cm),
    p("Verified imports after fixes: torch 2.2.2, xgboost 3.2.0, "
      "scikit-learn 1.8.0, pandas 2.3.3, numpy 1.26.4, shap 0.46.0, "
      "hmmlearn 0.3.3 - all clean.", BODY),
]

story += [PageBreak()]

# --- 3. Data ------------------------------------------------------------
story += [
    p("3. Data pipeline state", H1),
    p("Person A's loaders in <font face='Courier'>src/data_loader.py</font> "
      "were run end-to-end. All artefacts now sit under <font face='Courier'>"
      "data/</font> (which is gitignored).", BODY),
    table([
        ["File", "Origin", "Size", "Coverage"],
        ["data/raw/CRSPData_1925_2022.csv",
         "Copied from ~/Desktop/coqueret_3/", "494 MB",
         "1925-12 to 2022-12, all US listed"],
        ["data/raw/sp500_ticker_start_end.csv",
         "Downloaded from fja05680/sp500", "~25 KB",
         "S&P 500 membership spells"],
        ["data/raw/sp500.csv",
         "Downloaded from fja05680/sp500", "~85 KB",
         "Current 500 members + GICS sector"],
        ["data/processed/crsp_monthly.parquet",
         "Built by _load_crsp_monthly_raw", "152 MB",
         "Parsed + typed CRSP MSF"],
        ["data/processed/yfinance_monthly.parquet",
         "Built by _load_yfinance_monthly_raw", "~690 KB",
         "S&P 500 union, 2019-2024 month-ends"],
        ["data/processed/returns_spliced_2019_2024.parquet",
         "Built by notebooks/persona/freeze_returns_panel.py",
         "~670 KB",
         "72 months x 615 tickers wide returns matrix"],
    ], col_widths=[5.5 * cm, 5 * cm, 1.5 * cm, 4.5 * cm]),
    Spacer(1, 0.2 * cm),
    p("FRED macro cache was not built in this session (Person C's "
      "workstream; Person B's factors do not depend on macro features). "
      "Bowen's loader supports it via <font face='Courier'>load_macro()"
      "</font> on demand.", BODY),
]

# --- 4. factors.py -----------------------------------------------------
story += [
    p("4. src/factors.py - what was added", H1),
    p("Person A's feature spec lists 8 features. The current data pipeline "
      "is <b>monthly</b> only (CRSP MSF + month-end-resampled yfinance), "
      "so the daily-frequency features cannot be computed as specified. "
      "I implemented the feasible ones and added clearly-marked stubs for "
      "the rest.", BODY),
    p("4.1 Feature implementations", H2),
    table([
        ["#", "Spec name", "Implementation", "Status"],
        ["1", "Stock momentum (12-1)",
         "momentum(returns, lookback=11, skip=1) - cumulative monthly "
         "return from t-12 to t-2", "Implemented"],
        ["2", "Short-term reversal",
         "reversal(returns) - prior 1-month return", "Implemented"],
        ["3", "Market cap (size)",
         "log_market_cap_from_crsp - log(price x shrout) from CRSP. "
         "Available only for CRSP era (<=2022-12); NaN for 2023-2024.",
         "Implemented (partial)"],
        ["4", "Dollar volume",
         "daily_dollar_volume - raises NotImplementedError. Needs daily "
         "data.", "Stubbed"],
        ["5", "Return volatility",
         "monthly_volatility(returns, window=6) - 6-month rolling std of "
         "monthly returns. Substitute for the spec's 21-day daily vol.",
         "Implemented (substitute)"],
        ["6", "Idiosyncratic volatility",
         "idiosyncratic_volatility(returns, market, window=24) - 24-month "
         "rolling residual std from OLS on a market proxy (current proxy "
         "is the equal-weighted cross-sectional mean).",
         "Implemented (substitute)"],
        ["7", "Book-to-market (B/M)",
         "book_to_market - raises NotImplementedError. Needs Compustat.",
         "Stubbed"],
        ["8", "Earnings yield (E/P)",
         "earnings_to_price - raises NotImplementedError. Needs Compustat.",
         "Stubbed"],
    ], col_widths=[0.7 * cm, 3.5 * cm, 9 * cm, 3 * cm]),
    Spacer(1, 0.2 * cm),
    p("4.2 Sector handling", H2),
    p("Sector-relative ranking (Layer 1 of the Framework's three-layer "
      "stack, sec. 3.2) is the main contribution of factors.py. Two "
      "helpers:", BODY),
    code("load_sector_map() -> dict[ticker, GICS sector]<br/>"
         "    Reads sp500.csv for current GICS classification.<br/><br/>"
         "get_sector(ticker, sic_code, current_map) -> str<br/>"
         "    Lookup order: (1) current_map[ticker]; (2) 2-digit SIC<br/>"
         "    fallback via _SIC2_TO_SECTOR; (3) \"Unknown\"."),
    p("In the smoke test, 11 of 615 tickers ended up as \"Unknown\" "
      "(mostly long-delisted names CRSP carries but fja05680 does not). "
      "That is <2% of the panel and survives downstream filtering.", BODY),
    p("4.3 Top-level orchestrator", H2),
    code("build_feature_panel(start, end, include=(...), sector_rank=True)<br/>"
         "    -> pd.DataFrame indexed by (date, ticker)<br/>"
         "    -> columns: requested features + 'sector'"),
    p("Designed to be the single entry point your model code calls. "
      "Defaults span the Framework's full sample (2005-2024).", BODY),
    p("4.4 Smoke test result (real data)", H2),
    code("panel.shape         : (40186, 5)  # rows x cols<br/>"
         "date range          : 2019-01-31 -> 2024-12-31 (72 months)<br/>"
         "unique tickers      : 614<br/>"
         "NaN fraction by col:<br/>"
         "  mom                17%   (first 12 months are warmup)<br/>"
         "  rev                <1%<br/>"
         "  mvol               8%    (first 6 months are warmup)<br/>"
         "  log_mktcap         31%   (2023-24 has no CRSP coverage)<br/>"
         "  sector             0%"),
]

story += [PageBreak()]

# --- 5. models.py -------------------------------------------------------
story += [
    p("5. src/models.py - what was added", H1),
    p("Three model classes, each satisfying the <font face='Courier'>"
      "CrossSectionalModel</font> Protocol in <font face='Courier'>"
      "src/backtest.py</font> (i.e. <font face='Courier'>fit(X, y) -> self"
      "</font> and <font face='Courier'>predict(X) -> Series indexed like "
      "X</font>). This is the only interface the backtest engine relies on.",
      BODY),
    table([
        ["Class", "Role", "Defaults"],
        ["LassoModel",
         "L1-regularised linear baseline (sklearn LassoCV)",
         "alphas=100, cv=5, n_jobs=1, max_iter=20000, random_state=42"],
        ["XGBoostModel",
         "Gradient-boosted trees (primary per GKX 2020)",
         "n_estimators=300, max_depth=4, learning_rate=0.05, "
         "subsample=0.8, colsample_bytree=0.8, tree_method='hist', "
         "n_jobs=1, random_state=42"],
        ["NNModel",
         "Small feedforward net (secondary baseline)",
         "hidden_dim=32, n_layers=3, dropout=0.2, lr=1e-3, "
         "weight_decay=1e-4, batch=512, max_epochs=100, patience=10, "
         "random_state=42"],
    ], col_widths=[3 * cm, 5 * cm, 8 * cm]),
    Spacer(1, 0.2 * cm),
    p("5.1 Conventions inside each model", H2),
    p("&bull; <b>Non-predictive columns are dropped before fit.</b> The "
      "feature panel may carry a <font face='Courier'>sector</font> column "
      "for bookkeeping; <font face='Courier'>_split_X</font> strips it "
      "so the models cannot accidentally use it as a feature.", BODY),
    p("&bull; <b>NaN handling.</b> Lasso and NN median-impute training "
      "features and re-use those medians at predict time. XGBoost handles "
      "NaNs natively and is left to do so.", BODY),
    p("&bull; <b>Scaling.</b> Lasso and NN standardise inputs with "
      "sklearn's StandardScaler. XGBoost is scale-invariant by "
      "construction.", BODY),
    p("&bull; <b>Reproducibility.</b> Every model sets random_state=42 per "
      "the project's reproducibility rule (kickoff doc, principle 4).",
      BODY),
    p("5.2 NN training loop", H2),
    p("Architecture: <font face='Courier'>input -> (Linear + ReLU + Dropout) "
      "x n_layers -> Linear(1)</font>. Adam optimiser, MSE loss. Held-out "
      "20% of the training rows as a validation slice for early stopping "
      "(patience=10 by default). CPU device only; the panel is small "
      "enough that GPU is not needed.", BODY),
    p("5.3 Smoke test result (synthetic panel)", H2),
    code("Panel: 60 months x 50 assets, 4 features (mom, rev, mvol, "
         "log_mktcap)<br/>"
         "y = 0.3*mom - 0.2*mvol + N(0, 0.5) - a linear-ish target<br/><br/>"
         "Lasso  : in-sample corr(pred, y) = +0.585<br/>"
         "XGBoost: in-sample corr(pred, y) = +0.632<br/>"
         "NN     : in-sample corr(pred, y) = +0.572"),
    p("All three correctly recovered most of the linear signal. Real "
      "out-of-sample performance is a separate, harder question.", BODY),
    p("5.4 End-to-end against the backtest engine", H2),
    p("Wired factors.py + models.py + Person A's "
      "<font face='Courier'>run_walk_forward_backtest</font> against the "
      "frozen 2019-2024 panel. XGBoost with train_window=24, test_window=6, "
      "long/short quantiles 0.8/0.2, 10 bps transaction cost:", BODY),
    code("n_rebalances : 47<br/>"
         "net Sharpe    : -0.44<br/>"
         "ann return    : -3.1%<br/>"
         "max drawdown  : -20.9%<br/>"
         "avg turnover  : 2.20<br/>"
         "runtime       : 13.4 s on this laptop"),
    Paragraph(
        "<b>Interpretation:</b> Sharpe of -0.44 is not surprising and does "
        "not mean the scaffold is broken. The Framework's training "
        "window is 2005-2015 but the frozen panel only covers 2019-2024, "
        "so the model only ever sees 24 months of training data. The "
        "smoke test's job was to prove the pipeline runs end-to-end and "
        "the Protocol contract holds. Both are confirmed.",
        NOTE,
    ),
]

story += [PageBreak()]

# --- 6. Files changed ---------------------------------------------------
story += [
    p("6. Files changed in this session", H1),
    table([
        ["Path", "Change", "Lines"],
        ["src/factors.py",
         "Was a 2-line stub; now ~330 lines of working code",
         "+330"],
        ["src/models.py",
         "Was a 2-line stub; now ~300 lines (3 model classes + helpers)",
         "+300"],
        ["data/raw/CRSPData_1925_2022.csv",
         "Copied in from Desktop. Gitignored; not in the commit.",
         "(data, not commit)"],
        ["data/processed/*.parquet",
         "Three caches built by running Person A's loaders. Gitignored.",
         "(data, not commit)"],
        [".venv/",
         "Local Python 3.12 environment. Gitignored.",
         "(env, not commit)"],
        ["generate_personb_setup_report.py",
         "One-off script that produced this PDF. Safe to keep or delete.",
         "+250"],
    ], col_widths=[6 * cm, 8 * cm, 2 * cm]),
    Spacer(1, 0.2 * cm),
    p("Nothing has been committed yet. Recommended commit before next "
      "session:", BODY),
    code("git add src/factors.py src/models.py<br/>"
         "git commit -m \"personb: scaffold factors.py and models.py\""),
]

# --- 7. Open issues -----------------------------------------------------
story += [
    p("7. Open issues and gaps", H1),
    p("7.1 Daily vs monthly data (highest-impact gap)", H2),
    p("Person A's feature spec (Person_A_Feature_Spec.docx) assumes daily "
      "prices for features 4-6 (dollar volume, 21-day vol, 21-day "
      "idiosyncratic vol). The actual pipeline (CRSP MSF + yfinance "
      "resampled to month-end) is monthly. Three options:", BODY),
    table([
        ["Option", "Cost", "Effect on features"],
        ["A. Keep monthly, accept substitutes",
         "Zero engineering",
         "Features 5/6 use longer-window monthly proxies (already "
         "implemented). Feature 4 (dvol) cannot be implemented and is "
         "dropped from the 8-feature spec."],
        ["B. Add a daily loader to data_loader.py",
         "1-2 days of Person A work. Daily yfinance is feasible; CRSP DSF "
         "(daily) is much larger than MSF.",
         "All three features become implementable as written."],
        ["C. Stretch the substitutes",
         "Zero engineering",
         "Treat the spec as a guideline; monthly proxies are defensible "
         "in the report. This is the pragmatic course-project answer."],
    ], col_widths=[5 * cm, 4 * cm, 7 * cm]),
    Spacer(1, 0.2 * cm),
    p("My recommendation is C unless Bowen has bandwidth for B.", BODY),
    p("7.2 Compustat fundamentals (features 7, 8)", H2),
    p("DECISIONS.md 2026-05-13 \"Defer fundamentals\" set a 2026-05-20 "
      "fallback deadline for the TA reply on Compustat access. That date "
      "was yesterday. Either:", BODY),
    p("&bull; The TA replied and we have Compustat: implement B/M and E/P "
      "in factors.py (separate PR; needs the 45-day reporting lag).",
      BODY),
    p("&bull; The TA did not reply: formally adopt skip-fundamentals, "
      "log it in DECISIONS.md, and the project ships with 6 price-based "
      "features (the GKX paper supports this; its top-10 predictors are "
      "almost all price/liquidity/volatility).", BODY),
    p("7.3 GICS sector data", H2),
    p("The pipeline has no historical sector classification. I built a "
      "hybrid in factors.py: current GICS from sp500.csv plus a 2-digit "
      "SIC fallback for delisted tickers. 11 of 615 tickers in the "
      "2019-2024 panel end up as \"Unknown\". This is acceptable for the "
      "course but should be logged in DECISIONS.md as a known "
      "approximation.", BODY),
    p("7.4 Training window", H2),
    p("Frozen panel covers 2019-2024 only. To reach the Framework's "
      "2005-2015 training window, either re-run the freeze script with "
      "<font face='Courier'>start=\"2005-01-01\"</font> or call "
      "<font face='Courier'>load_prices_spliced</font> directly from a "
      "Person B notebook. CRSP cache already covers 1925-2022 so this is "
      "just a parameter change.", BODY),
]

story += [PageBreak()]

# --- 8. Next steps ------------------------------------------------------
story += [
    p("8. Next steps for Person B", H1),
    p("In priority order:", BODY),
    table([
        ["#", "Task", "Why it matters"],
        ["1",
         "Commit src/factors.py and src/models.py on personb-models and "
         "open a PR.",
         "Get the scaffold visible to Bowen and Person C so they can react."],
        ["2",
         "Extend the frozen returns panel to 2005-2024 (or call "
         "load_prices_spliced(start='2005-01-01') from a notebook).",
         "Without a long training window the walk-forward backtest cannot "
         "produce a meaningful Sharpe."],
        ["3",
         "Implement sector-relative target (Framework Layer 2): inside "
         "each model's fit(), subtract the per-(date, sector) mean from "
         "y before training. Sector column is already on the feature "
         "panel.",
         "Layer 2 is half the Framework's sector-neutrality story; Layer 1 "
         "is already done in factors.py."],
        ["4",
         "Add an OOS R-squared metric in src/metrics.py (vs zero and vs "
         "S&P 500 mean) per Framework section 8.2.",
         "Required Level-1 evaluation deliverable."],
        ["5",
         "Add a Diebold-Mariano test for pairwise model comparison per "
         "Framework section 8.4.",
         "Required for ranking Lasso vs XGBoost vs NN in the report."],
        ["6",
         "Tune XGBoost hyperparameters on the validation window 2016-2018 "
         "(Optuna is in requirements.txt).",
         "Untuned XGBoost is a baseline only."],
        ["7",
         "Iterate on features: lag features, dynamics features "
         "(Framework section 3.5) once the base 6 features pass the "
         "sanity bar.",
         "Improves model expressiveness once the base pipeline works."],
    ], col_widths=[0.7 * cm, 7.3 * cm, 7 * cm]),
]

# --- 9. Talk to Bowen ---------------------------------------------------
story += [
    p("9. What to tell Bowen (Person A)", H1),
    p("Short version: \"My scaffold for factors.py and models.py is on "
      "personb-models and runs end-to-end against your backtest engine. "
      "Four things I want your input on:\"", NOTE),
    p("9.1 Decide on the daily-vs-monthly question (section 7.1).", H3),
    p("Specifically: are you willing to add a daily-frequency loader, or "
      "should I officially substitute monthly proxies for features 4-6 "
      "and document that in DECISIONS.md?", BODY),
    p("9.2 Compustat status (section 7.2).", H3),
    p("Has the TA replied? If 2026-05-20 has passed with no answer, can "
      "we agree to formally drop B/M and E/P and add a DECISIONS.md entry?",
      BODY),
    p("9.3 Three requirements.txt pins worth adding.", H3),
    p("On macOS x86_64 with Anaconda Python 3.12 (which is the setup I "
      "ended up on), the fresh install of <font face='Courier'>"
      "requirements.txt</font> breaks in three ways: numpy 2.x is "
      "incompatible with the torch 2.2.2 wheel, shap 0.51 hard-requires "
      "numpy>=2, and the libomp duplicate causes deadlocks. Suggested "
      "pins:", BODY),
    code("numpy<2<br/>"
         "shap<0.47<br/>"
         "# README addition: set KMP_DUPLICATE_LIB_OK=TRUE if you hit a<br/>"
         "# hang during NN training on macOS."),
    p("These are low-risk and matter for any teammate setting up a fresh "
      "venv.", BODY),
    p("9.4 The engine refits every period, not every test_window.", H3),
    p("Reading <font face='Courier'>backtest.py</font> line 286 onwards, "
      "the walk-forward loop calls <font face='Courier'>model.fit()</font> "
      "on every rebalance date, not at refit boundaries every "
      "test_window periods. The docstring (lines 156-170) describes a "
      "block-refit scheme. Either the docstring is aspirational or the "
      "implementation got simplified; worth confirming which is intended "
      "before either of us optimises around it. With Lasso it makes the "
      "backtest noticeably slower than the docstring's mental model.",
      BODY),
    Spacer(1, 0.2 * cm),
    p("Anything else, ask me in the next session.", BODY),
]


doc = SimpleDocTemplate(
    str(OUT_PATH),
    pagesize=A4,
    topMargin=1.4 * cm,
    bottomMargin=1.5 * cm,
    leftMargin=1.6 * cm,
    rightMargin=1.6 * cm,
    title="Person B Setup Session Status Report",
    author="Person B (Nicolas)",
)


def on_page(canvas, doc_):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.drawString(1.6 * cm, 0.8 * cm,
                      "ml-factor-investing-2026 - Person B setup - 2026-05-21")
    canvas.drawRightString(A4[0] - 1.6 * cm, 0.8 * cm,
                           f"Page {doc_.page}")
    canvas.restoreState()


doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
size_kb = OUT_PATH.stat().st_size / 1024
print(f"Wrote {OUT_PATH.name} ({size_kb:.1f} KB)")
