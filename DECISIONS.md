# DECISIONS.md

A running log of every significant design, modeling, or data choice made in this project. The Week 5 report is basically a synthesis of this file, so keep it up to date.

## How to use this log

Every time you make a non-trivial "choose A over B" decision, add a new entry using the template below. Keep entries short and honest: what you picked, what you rejected, and why.

## Entry template

### YYYY-MM-DD — Short title of the decision

**Context:** What problem or question prompted this choice?

**Options considered:** Option A with its pros and cons; Option B with its pros and cons; Option C with its pros and cons.

**Decision:** The option actually chosen.

**Reasoning:** Why this one? What trade-offs are you accepting?

**Revisit if:** Conditions under which you would reopen this decision (e.g., "if OOS Sharpe < 0.3", "if training time > 2h").

## Log

### 2026-04-23 — Repo structure and tooling baseline

**Context:** Project kickoff. Need a reproducible layout before any code or data lands.

**Options considered:** Follow the kickoff doc's recommended layout (data/, notebooks/, report/, results/, src/) — standardized and matches the grading rubric, but slightly heavier than needed for a solo project. Flat layout with everything at repo root — simpler, but breaks down fast once notebooks and scripts multiply.

**Decision:** Use the kickoff doc layout verbatim.

**Reasoning:** Standard structure makes the repo reviewable and lets tooling (pytest, nbstripout, etc.) work out of the box. Cost of following the convention is near zero.

**Revisit if:** The structure actively gets in the way (unlikely).

### 2026-04-23 — Lock the data, not the dataset

**Context:** Review feedback flagged that data/ was not gitignored and no DECISIONS.md existed.

**Options considered:** Commit raw parquet/CSV files directly — anyone cloning gets data immediately, but this hits GitHub's 100MB file limit, bloats the repo, and couples data version to code version. Gitignore everything under data/ — clean, but loses any folder-level docs like data/README.md. Gitignore only data file extensions under data/ (parquet, csv, feather, h5, pkl) and keep data/README.md and .gitkeep — preserves folder structure and docs while excluding heavy binaries.

**Decision:** Option 3. Pattern-based ignore in data/.

**Reasoning:** Version-control the code that produces data, not the data itself. Anyone who clones the repo should be able to regenerate the dataset via the download scripts in src/.

**Revisit if:** We start needing to share a small canonical processed sample for tests — in which case commit it under a separate path like tests/fixtures/.

### 2026-04-23 — Start a decisions log

**Context:** Kickoff doc recommends a DECISIONS.md so the Week 5 report writes itself.

**Decision:** Create this file now and log every non-trivial choice going forward.

**Reasoning:** Cheap insurance against "why did I do this again?" five weeks from now.

**Revisit if:** Never.

### 2026-04-23 — Pin pandas to <3.0

**Context:** Fresh `pip install -r requirements.txt` on macOS arm64 resolved pandas to 3.0.2. `import pandas_datareader` then failed with `TypeError: deprecate_kwarg() missing 1 required positional argument: 'new_arg_name'` — pandas_datareader 0.10.0 has not been updated for pandas 3.0's API changes.

**Options considered:** Leave pandas unpinned and wait for pandas_datareader to catch up — simplest, but blocks clean installs today with no upstream fix ETA. Drop pandas_datareader and hit FRED directly via requests — removes the dependency, but adds a custom code path to maintain. Pin `pandas<3` in requirements.txt — one-line fix, keeps the existing data-loading path working, and pandas 2.x is the line the rest of the stack (scikit-learn, xgboost, shap) is built against.

**Decision:** Pin `pandas<3` in requirements.txt.

**Reasoning:** The incompatibility lives entirely inside pandas_datareader; forcing ourselves off it is bigger than the problem. Nothing we use depends on a pandas-3.0-only feature.

**Revisit if:** pandas_datareader ships a pandas-3.x-compatible release, or we replace it with a different data source.

## 2026-05-11 - Equity universe & primary data source: S&P 500 via yfinance + FRED for macro

**Context:** Day 2 - need to lock in *what* data we pull and *from where* before Person B can finalize feature definitions and Person C can finalize regime features.

**Options considered:** (a) S&P 500 via yfinance + FRED via pandas_datareader - free, no key, fast iteration, well-trodden path matching every reference implementation we've read. (b) CRSP / Compustat through the university WRDS account - the academic gold standard, has point-in-time membership and proper survivorship handling, but onboarding takes days and rate limits make rapid iteration painful in a 5-week project. (c) Alpha Vantage / Tiingo paid APIs - cleaner adjustments than yfinance but require keys, billing, and per-request quotas that break notebooks.

**Decision:** Start with **yfinance for daily adjusted-close prices** and **FRED (via pandas_datareader) for macro/regime features** (VIX, term spread, credit spread, etc.). Universe = S&P 500. Survivorship is handled separately via a point-in-time membership table (`load_sp500_membership`, source TBD - Wikipedia history is the leading candidate).

**Reasoning:** Free, public, reproducible by any reviewer, and matches the data setup in the Gu-Kelly-Xiu replications we are using as a reference. yfinance's adjustments are good enough for a relative-ranking strategy where every stock gets the same treatment.

**Revisit if:** We find a systematic adjustment bias in yfinance that biases ranks (unlikely), or we want to extend the universe beyond the S&P 500 (Russell 1000 has less coverage on yfinance).


## 2026-05-11 - Backtest interface contract: `run_walk_forward_backtest` is the only seam between workstreams

**Context:** With three people working in parallel, we need to pin down the *interface* between the alpha model (B), the regime overlay (C), and the backtest engine (A) before any of us go too deep. Otherwise we will rewrite glue code three times.

**Options considered:** (a) One monolithic `Strategy` class that owns features, model, regime, and backtest - tight coupling, but easier to refactor for one person. (b) Functional pipeline with a single `run_walk_forward_backtest` entry point that takes a model object (duck-typed with fit/predict) and a `leverage_fn: Callable[[date], float]` - explicit seam, each person can iterate behind their own interface. (c) Event-driven backtest a la zipline / vectorbt - overkill for monthly rebalancing and adds a heavy dependency.

**Decision:** Option (b). `src/backtest.py` defines a `CrossSectionalModel` Protocol (just `fit(X, y)` and `predict(X)`) plus a `LeverageFn` type alias, and `run_walk_forward_backtest` is the single entry point. Output is a frozen `BacktestResult` dataclass.

**Reasoning:** Smallest possible contract that still produces a useful tear sheet. B can swap Lasso for XGBoost for an NN without touching A's code. C can swap GMM for HMM without touching A's code. The interface is versioned via `INTERFACE_VERSION` in backtest.py and changes require shouting at standup.

**Revisit if:** We need to support intraday rebalancing (we don't), or the leverage overlay needs to see more than a date (e.g. portfolio state) - in which case widen `LeverageFn` rather than redesigning.


## 2026-05-13 - Point-in-time S&P 500 membership source: fja05680/sp500

**Context:** The 2026-05-11 entry decided "Universe = S&P 500 with point-in-time membership" but left the actual data source open ("source TBD - Wikipedia history is the leading candidate"). Need to lock this down before `load_prices` can know which tickers to pull.

**Options considered:** (a) **github.com/fja05680/sp500** - a maintained CSV that records every (ticker, start_date, end_date) membership spell back to 1996. Includes delisted tickers (LEHMQ, AABA, TWTR, etc.) with post-bankruptcy SEC suffixes. 27 KB, plain CSV, no auth, no rate limit. (b) Wikipedia "List of S&P 500 companies" - scrape the "Selected changes" table and reconstruct membership by walking revisions. Authoritative but requires custom parsing and the revisions only go back ~10 years cleanly. (c) CRSP via WRDS - the academic gold standard, but onboarding takes days (already rejected on 2026-05-11 for the same reason).

**Decision:** Option (a). `download_sp500_universe()` fetches `sp500_ticker_start_end.csv` and `sp500.csv` from fja05680/sp500 into `data/raw/`. `load_sp500_membership(asof)` reads the membership CSV and returns sorted tickers whose spell straddles `asof`.

**Reasoning:** Smallest possible thing that works. Verified against 5 famous events (Lehman in Sept 2008, Tesla joining Dec 2020, Twitter going private Oct 2022, etc.) — all consistent with this dataset. Wikipedia would have been a multi-day side quest.

**Gotcha to remember:** fja05680/sp500 uses **post-bankruptcy SEC tickers** (e.g. `LEHMQ` not `LEH`, `WAMUQ` not `WM`). yfinance generally indexes by the *trading* ticker, so the price loader will need a small mapping/fallback layer for the dozen-or-so bankruptcy cases. Track as TODO when implementing `load_prices`.

**Revisit if:** fja05680/sp500 stops being maintained (last commit was recent as of 2026-05), or we find a ticker that's clearly wrong on a date we care about.


## 2026-05-13 — Adopt Project Framework Complete (May 2026) as the master design doc

**Context:** Several reference documents now exist (kickoff plan, Person A data pipeline checklist, Fama-French takeaways, Gu-Kelly-Xiu paper, stock-prediction RF guide, sample-splitting report) plus a new comprehensive "Project Framework Complete" PDF. With multiple sources of truth, it is unclear which takes precedence when they conflict.

**Options considered:** (a) Treat each document as advisory and resolve conflicts case by case — flexible but creates ambiguity about what "the plan" actually is. (b) Designate the Project Framework Complete as the master spec, with all other documents as supporting/explanatory material — single source of truth for design questions; the price is that deviating from the framework now requires a DECISIONS.md entry. (c) Re-author a single new master doc by merging all references — high effort, duplicates the framework which is already comprehensive.

**Decision:** Option (b). Project Framework Complete (May 2026) is the master spec. The other PDFs are reference / pedagogy / methodological background. Any deviation from the framework gets logged here.

**Reasoning:** The framework is the most concrete and recent document, and it cites/incorporates the others. Aligns with the user's explicit instruction "我们以后主要按照那个 pdf 说的来吧". Keeps the rule simple: when in doubt, check the framework.

**Revisit if:** A new comprehensive design document supersedes it, or the framework develops gaps we cannot resolve by extension.


## 2026-05-13 — Widen regime overlay interface: `LeverageFn` → `RegimeFn` returning `RegimeParams`

**Context:** The 2026-05-11 entry defined `LeverageFn = Callable[[pd.Timestamp], float]` as the contract between Person C's regime overlay and Person A's backtest engine. The newly-ratified Project Framework (see entry above) §5.3 specifies that the regime overlay must communicate **three** risk knobs to the backtest: gross leverage, breadth (number of stocks per sector leg), and entry threshold (quantile cutoff). A scalar return type cannot encode the latter two.

**Options considered:** (a) Keep `Callable[..., float]` and silently drop breadth/threshold from the framework — fastest path, but the framework's risk-management story (half the project's contribution) becomes inexpressible in code. (b) Add two more parallel callables (`LeverageFn`, `BreadthFn`, `ThresholdFn`) — keeps each contract scalar-simple but explodes the `run_walk_forward_backtest` signature and forces Person C to expose three separate entry points. (c) Widen the contract to `Callable[..., RegimeParams]` where `RegimeParams` is a `TypedDict(total=False)` with optional keys `{leverage, long_quantile, short_quantile, k_per_sector}`. The regime overlay populates only the keys it actively controls; the backtest fills missing keys from its static defaults or neutral fallbacks.

**Decision:** Option (c). [src/backtest.py](src/backtest.py) now defines `RegimeParams` (TypedDict) and `RegimeFn = Callable[[pd.Timestamp], RegimeParams]`. The `run_walk_forward_backtest` signature replaces `leverage_fn: LeverageFn | None` with `regime_fn: RegimeFn | None`. `INTERFACE_VERSION` bumped from `0.1.0` to `0.2.0` with a changelog block at the bottom of the file.

**Reasoning:** Smallest schema change that accommodates the framework's three-knob model. `TypedDict(total=False)` gives type-checker support without forcing the regime to specify keys it does not care about — a leverage-only regime just returns `{"leverage": 0.7}`, identical-ish ergonomics to the old `LeverageFn`. No production code depended on the v0.1.0 contract yet (only signature stubs), so this is a clean break with zero migration cost.

**Revisit if:** The framework adds a fourth knob (extend `RegimeParams`), or we discover the regime needs to see portfolio state to make its call (in which case widen to `Callable[[pd.Timestamp, PortfolioState], RegimeParams]`).


## 2026-05-13 — Switch primary price source: yfinance → CRSP MSF

**Context:** The 2026-05-11 entry chose yfinance for daily adjusted-close prices on cost / accessibility grounds. A course TA then shared a CRSP Monthly Stock File (MSF) covering 1925-12 → 2022-12 (471 MB CSV, ~37k unique PERMNOs, all US listed common stocks). CRSP is the academic gold standard, and the Project Framework (now the master spec) implicitly assumes CRSP-level data quality. The yfinance plan would have left us with several known footguns (LEHMQ-vs-LEH ticker mismatches, missing delisting returns, currently-listed-only universe, rate limits).

**Options considered:** (a) Stick with yfinance — easiest, no new code, but inherits all the survivorship and quality issues we just paid to identify. (b) CRSP MSF as primary, no fallback — best data quality, but CRSP file ends 2022-12 so the Project Framework's 2019-2024 test window can only be evaluated 2019-2022. (c) CRSP MSF primary + yfinance splice for 2023-2024 — full test window preserved, but yfinance for 2023-2024 silently re-introduces survivorship bias for any S&P 500 stock delisted in that window (e.g., ATVI, SPLK, SIVB, FRC, SBNY).

**Decision:** Option (c). Primary source is now CRSP MSF, loaded via `_load_crsp_monthly_raw` and exposed through the rewritten `load_prices`. The yfinance splice for 2023-2024 is planned but not yet implemented; see the next entry.

**Reasoning:** CRSP solves the bias problems we worked hardest to identify (point-in-time universe, proper delisting handling for LEH and AABA, stable PERMNO identifier, accurate adjustment for splits/dividends). The fja05680 membership table still stays in the pipeline as the S&P 500 universe filter — CRSP has all US stocks, not an index-membership flag. The yfinance "rescue" of 2023-2024 is the smallest acceptable departure from the Framework's 2019-2024 test window.

**Implementation note:** `load_prices` now returns a `(date, permno)` MultiIndex frame (PERMNO is the stable CRSP identifier; TICKER changes when companies restructure). Bid-ask midpoint convention (negative PRC), alpha-coded RET values (`B`, `C`), and SHROUT-in-thousands are all handled inside `_load_crsp_monthly_raw`. Parsed result cached to `data/processed/crsp_monthly.parquet` (~152 MB). Adds `pyarrow` to requirements.txt for parquet I/O.

**Revisit if:** The TA shares an updated CRSP extending past 2022-12 (kills the need for the yfinance splice), or we get access to CRSP-Compustat Merged (would let us add fundamentals too — see next-next entry).


## 2026-05-13 — Yfinance splice for 2023-2024 (planned, not yet implemented)

**Context:** CRSP MSF ends 2022-12. The Project Framework's test window is 2019-2024. To preserve the full window we need to splice yfinance data for the 2023-2024 tail.

**Options considered:** (a) Shrink the test window to 2019-2022 — clean data, but only 3 years of OOS evaluation and misses the 2023 banking-crisis regime (a perfect natural experiment for the regime overlay). (b) Splice CRSP (2005-2022) + yfinance (2023-2024) with explicit safeguards: overlap-month price sanity check (Dec 2022), manual patch list for stocks delisted from S&P 500 in 2023-2024 (ATVI, SPLK, SIVB, FRC, SBNY at minimum), and dual reporting of `Sharpe(2019-2022)` vs `Sharpe(2019-2024)` so any "improvement" from the spliced years that looks like a survivorship boost is visible. (c) Wait for the TA to send an updated CRSP — unknown timeline, blocks Person B and C.

**Decision:** Option (b) at the design level, but implementation is **deferred to a follow-up PR**. The current PR ships only the CRSP loader; calls with `end > 2022-12-30` will simply return what CRSP has and the caller has to know not to read past that. The splice + safeguards will land alongside Person B's first model run, when the test-window dates actually start mattering.

**Reasoning:** Don't pre-build the splice machinery before we know the model side is ready to consume it. The CRSP-only loader is the dependency-of-everything; ship that first and unblock Person B / C.

**Revisit if:** Person B starts the first end-to-end backtest and we hit "CRSP ends 2022-12-30" as a real blocker (then implement the splice immediately).


## 2026-05-13 — Defer fundamentals (B/M, E/P, D/P) pending Compustat access check

**Context:** Project Framework §4.4 and the Fama-French Takeaways both list firm-level fundamentals (book equity, earnings, dividends) in Person A's responsibilities, with the rationale that B/M, E/P, D/P are the canonical Fama-French value factors. CRSP MSF has none of these — that's Compustat. User (Person A) has emailed the course TA to ask whether the same channel that produced the CRSP file can also produce Compustat.

**Options considered:** (a) Skip fundamentals permanently and rely on price-based factors only — supported by GKX (2020) Figures 4-5 which place price/liquidity/volatility features in the top ~10 predictors and fundamentals at rank 50+. Saves ~3-5 days of Person A work and avoids the 45-day reporting lag enforcement, ticker→GVKEY mapping, and quarterly-to-monthly ffill machinery. (b) Wait for the TA reply and pursue Compustat if available — best factor coverage (B/M, E/P, D/P plus the broader 94-feature set GKX uses), defensible in the report ("we replicated the full Fama-French feature stack"), but blocks on a response that may take days and may be negative. (c) Use yfinance fundamentals as a fallback — bad quality (only last 4 quarters, no delisted firms, occasional wrong numbers) and re-introduces survivorship bias.

**Decision:** **Defer** the binary skip/include decision. Person A proceeds with the CRSP-only price loader (this PR) and the price-based feature pipeline; Person B is asked to chase the Compustat data request rather than blocking Person A's work on it. If Compustat lands, we add a separate `load_fundamentals` later in a follow-up PR. If the TA says no, we formally adopt option (a) and explicitly note in the final report that we used only price-based features.

**Reasoning:** Person B wants fundamentals (his words: "even 20% importance is still important"), which is a defensible position — the Fama-French value factors do add marginal signal even if they're not the top predictors. But the data-acquisition risk shouldn't block Person A's downstream work; pipelining the request alongside the price loader keeps the project's critical path unblocked.

**Revisit if:** TA replies with Compustat access (then build `load_fundamentals` in a follow-up PR with 45-day reporting lag), or by 2026-05-20 with no reply (then formally adopt skip-fundamentals and update this entry).


## Upcoming decisions to log

Placeholders to fill in as they happen:

data source(s) and universe definition (which stocks, regions, date range); feature set and factor definitions; train / validation / test split scheme (walk-forward or expanding window); target definition (returns horizon, winsorization, normalization); model family choice (linear baseline, tree ensemble, NN) and reason for baseline-first; hyperparameter search strategy and budget; evaluation metrics (IC, Sharpe, turnover, drawdown) and which one is primary; transaction cost assumptions.
