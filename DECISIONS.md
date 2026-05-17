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


## Upcoming decisions to log

Placeholders to fill in as they happen:

data source(s) and universe definition (which stocks, regions, date range); feature set and factor definitions; train / validation / test split scheme (walk-forward or expanding window); target definition (returns horizon, winsorization, normalization); model family choice (linear baseline, tree ensemble, NN) and reason for baseline-first; hyperparameter search strategy and budget; evaluation metrics (IC, Sharpe, turnover, drawdown) and which one is primary; transaction cost assumptions.
