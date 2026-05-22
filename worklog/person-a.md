# Person A — Bowen

Role: data + backtest infrastructure (@bowenzuo119-hash)

---

## 2026-05-11

**Done:**
- Re-invited Person B (@nicolascoutomota-boop) as collaborator after the original invite expired.
- Created branch `persona-data-pipeline` and opened PR #1 (merged into main).
- Wrote `src/data_loader.py` skeleton: `load_prices`, `load_macro`, `load_sp500_membership` signatures + numpy-style docstrings (bodies raise NotImplementedError). Flagged survivorship-bias risk in comments.
- Wrote `src/backtest.py` skeleton: `run_walk_forward_backtest` signature, `CrossSectionalModel` Protocol (the fit/predict interface B will implement), `LeverageFn` type alias (the function C will return), `BacktestResult` frozen dataclass, and `INTERFACE_VERSION = "0.1.0"`.
- Expanded `README.md` from a stub to real project documentation (goal, repo layout, team roles, setup, reproducibility rules).
- Added two `DECISIONS.md` entries (data sources = yfinance + FRED; backtest interface contract).
- Created this worklog folder with templates for all three of us.

**In progress:**
- Step 5 from the kickoff plan: run `yfinance.download("AAPL", start="2010-01-01")` in `notebooks/01_data_exploration.ipynb` locally and push it. Will open a follow-up PR.

**Blocked on / need from teammates:**
- @nicolascoutomota-boop needs to accept the GitHub invite (still pending) so he can push to `personc-regime` and review PRs.
- Need @agfontana / @nicolascoutomota-boop to actually Approve the next PR before I merge — last time I self-merged PR #1, which was against our review rule. Won't happen again.

**Decisions / notes:**
- Locked the backtest interface at version 0.1.0. If B or C want to change the function signature, bump the version and call it out in DECISIONS.md.
- Accidentally hit "Revert" on the GitHub UI after merging PR #1. It only created a branch, no revert PR was opened, so main was unaffected. Cleaned up the orphan branch. Lesson logged.
