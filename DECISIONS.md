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

## Upcoming decisions to log

Placeholders to fill in as they happen:

data source(s) and universe definition (which stocks, regions, date range); feature set and factor definitions; train / validation / test split scheme (walk-forward or expanding window); target definition (returns horizon, winsorization, normalization); model family choice (linear baseline, tree ensemble, NN) and reason for baseline-first; hyperparameter search strategy and budget; evaluation metrics (IC, Sharpe, turnover, drawdown) and which one is primary; transaction cost assumptions.
