# 1-PAGE SUBMISSION — MARKDOWN DRAFT

> **Status:** Draft for review. Convert to PDF via Pandoc / Typst / a Google Doc / a LaTeX template once the team signs off on the wording.
> **Format constraint:** must fit on a single A4 / letter page per the professor's instructions. The wording below is sized for that — if it overflows in the chosen layout, trim the Brief Report first (the Highlights and Keywords are load-bearing).
>
> **Layout target:** 1 title block + 1 author block (3 columns since we are a team of 3) + Highlights bullet list + Brief Report paragraph + 1-line Appendix description. The template PDF (`Project_Template.pdf`) is the reference for visual layout.

---

# Machine-Learning Factor Investing on a Survivorship-Free US Equity Universe

**Team 6** *(check / replace with assigned team number)*

---

## Authors (3 columns; replace photos when laying out the PDF)

| **Bowen Zuo** | **Nicolas Couto Mota** | **Andrea Fontana** |
|---|---|---|
| *Person A — data & infrastructure* | *Person B — alpha model* | *Person C — regime overlay* |
| • Walk-forward backtest engine (PIT, sector-neutral, block-gated refit) | • XGBoost / Lasso / Neural Net cross-sectional regressors | • Hidden Markov Model regime detection (Gaussian HMM) |
| • Survivorship-free data lane (Sharadar SF1/SEP/DAILY/TICKERS) | • Feature engineering — 14 firm features (GKX 2020 stack + chmom) | • Gaussian Mixture Model regime detection (model comparison) |
| • Fama-French 5-factor + Carhart + block-bootstrap + DSR robustness | • Optuna hyperparameter tuning + Diebold-Mariano model comparison | • Walk-forward expanding-window OOS evaluation (look-ahead-free) |

> *Keywords above indicate scope/contribution lanes so questions can be distributed across speakers. All three authors contributed equally in terms of effort across the 5-week project. **Note for the examiner:** Andrea Fontana is unable to attend the live presentation; the regime-overlay workstream will be summarised briefly by Bowen Zuo (engine integration owner), and the full methodology + audit script is in the appendix. Regime-internal Q&A can be taken at the high level by the present authors and a deeper follow-up arranged with Andrea after the session if needed.*

---

## Highlights

- **Full-OOS Sharpe +1.15 / FF5 alpha +18.7%/yr at t=+6.85 (p < 0.001)** over 2012–2024 on a broad survivorship-free US equity universe (~4,400 names/month median); confirming long-OOS (Sh +0.97, α +19.1%/yr, t=+6.00) and test-OOS (Sh +1.00, α +21.2%/yr, t=+5.00) numbers.
- **Alpha survives every rigor check:** Carhart 6-factor (UMD-controlled α +20.1%/yr at t=+7.4, momentum-averse), block bootstrap (P(SR≤0)=0.0002), Deflated Sharpe Ratio (0.85–0.88 at N=25 trials, Bailey & López de Prado 2014), cost-grid (significant up to ~50 bps/side), **feature-shuffle placebo** (+1.15 → −0.94 when feature→ticker mapping is randomly permuted within each rebalance — rules out engine / target / cost leakage).
- **Audit-driven methodology:** caught and corrected a *survivorship leak in our own engine* mid-project (apparent Sharpe +1.49 → −0.31 on S&P-500-only once point-in-time eligibility is enforced). The honest +1.15 is what remains after the rebuild on the broad survivorship-free universe.
- **Honest down-cap finding** (Gu-Kelly-Xiu 2020 style): on the strict rolling top-2,000 large-cap sub-universe alone, FF5 α collapses to **+1.8%/yr at t=0.96 (not significant)** — the headline alpha lives in the small/mid-cap tail where capacity and realistic trading costs are the binding constraints at deployable AUM.
- **Honest regime-overlay finding:** HMM-based leverage overlay is net-zero on the broad book (COVID-2020 monthly-frequency timing lag), but adds value on the strict-S&P canonical (max DD −25.5% → −19.9%). Universe-dependent result, transparently discussed in §6 of the long-form report.

## Brief Report

**Scope.** We test whether **machine-learning cross-sectional alpha** on US equities survives survivorship-bias correction, point-in-time eligibility filters, realistic transaction costs, and multiple-testing penalties on a single ~13-year out-of-sample path (2012–2024). The pipeline is monthly-rebalanced, dollar-neutral long-short with sector-neutral construction (k=20 long/short picks per GICS sector ≈ 440 positions), trained via **walk-forward expanding-window cross-validation** with a 120-month training window and block-gated refit at each 12-month test boundary.

**Method.** Three models with the same interface — a regularised linear baseline (**Lasso**), the canonical **gradient-boosted tree ensemble (XGBoost)** tuned by **Optuna** TPE on the 2017–18 validation window, and a small **PyTorch feed-forward neural net** — share Person A's `run_walk_forward_backtest` engine and our 14-feature panel (12-month momentum, short-term reversal, return / idiosyncratic volatility, log market cap, log dollar volume, book/market, earnings/price, ROE, ROA, debt/equity, asset growth, accruals, **chmom = change-in-6-month momentum**, GKX top-5 feature). The target is **sector-relative monthly return**, and selection picks top-k/bottom-k by sector-normalised XGBoost score each month.

**Evaluation.** Headline metrics are **annualised Sharpe ratio (with Newey-West-corrected significance test), Fama-French 5-factor alpha (HAC SE), and information coefficient (Spearman)**. Model comparison uses the **Diebold-Mariano test** (XGBoost wins decisively on every window and every reduction). Robustness includes a 3/3 **sanity gate** (Random / Oracle / Uniform predictor on a synthetic panel), block bootstrap (P(SR≤0)), DSR at N=25 trials, FF5 → Carhart 6-factor with UMD, a single-name leave-one-out study, a **dense k-sweep** (k ∈ [1, 100] with bootstrap CIs on the [10,20] plateau showing k=20 is statistically indistinguishable from every neighbouring value), and the feature-shuffle placebo cited above. A regime overlay (Person C's HMM on 6 macro-financial features) scales leverage in detected crises and is reported in ablation.

**What was left out of the presentation:** the long-form **PIT-leak audit narrative** (preserved in `PIT_INVESTIGATION_REPORT.pdf` and `DECISIONS.md`); the **per-window FF3 regressions** (only FF5 + Carhart shown in slides); the bug-corrections to the canonical pkl (Q-filter `endswith('Q') AND isdelisted=='Y'`; INCLUDE_FEATURES subset enforcement); the full **Phase 23a/24a Optuna search-space and trial-by-trial logs**; the regime model's in-sample-vs-OOS crisis-detection-rate breakdown (91.7% IS vs 51.1% OOS); the four pre-correction lineage phases (Phases 1–15) and their decision rationale.

**Main references** *(beyond course material on Lasso / gradient boosting / NN / cross-validation / hyperparameter tuning):* Gu, S., Kelly, B. & Xiu, D. (2020) *Empirical Asset Pricing via Machine Learning,* RFS 33(5) — the methodological template and source of the feature set; Bailey, D. & López de Prado, M. (2014) *The Deflated Sharpe Ratio,* JPM 40(5) — multiple-testing penalty; Fama, E. F. & French, K. R. (2015) *A Five-Factor Asset Pricing Model,* JFE 116(1); Carhart, M. M. (1997) — momentum factor control; Newey, W. K. & West, K. D. (1987) — HAC standard errors; Sharadar / Nasdaq Data Link — data source under licence. Full citation list in the appendix.

## Description of Appendices

**Optional appendix material** (single combined PDF, in the order it appears): full long-form report with all tables and figures (`report/REPORT.md`, ~30 pages when typeset); the project decision log with chronological provenance for every methodological choice (`DECISIONS.md`, 1,233 lines); the survivorship-leak audit and methodology corrections (`PIT_INVESTIGATION_REPORT.pdf`); robustness-check artefacts (Phase 25 Deflated Sharpe + bootstrap; Phase 26 leave-one-out; Phase 27/27b dense k-sweep with CI plateau); regime-overlay ablation and overlay-failure diagnostic; all source code under `src/` and `notebooks/{persona,personb,personc}/`. **Total submitted PDF size:** *(to be measured after compilation; target < 30 pages, single PDF)*.

---

*Project repository:* https://github.com/bowenzuo119-hash/ml-factor-investing-2026
