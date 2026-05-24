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

- **Full-OOS Sharpe +1.15 / FF5 α +18.7%/yr at t=+6.85 (p<0.001)** over 2012–2024 on a broad survivorship-free US equity universe (~4,400 names/month); confirming long-OOS (Sh +0.97, α +19.1%, t=+6.00) and test-OOS (Sh +1.00, α +21.2%, t=+5.00).
- **Alpha survives every rigor check:** Carhart 6F (α rises to +20.1%/yr, UMD β=−0.43, momentum-averse), block-bootstrap (P(SR≤0)=0.0002), Deflated Sharpe Ratio (0.85–0.88 at N=25 trials), cost-grid stress (significant to ~50 bps/side), **feature-shuffle placebo** (+1.15 → −0.94 when feature→ticker mapping is permuted — rules out engine/target/cost leakage).
- **Audit-driven methodology:** caught and corrected a *survivorship leak in our own engine* (Sharpe +1.49 → −0.31 on S&P-500-only once PIT eligibility enforced). The honest +1.15 is what remains after rebuilding on the broad survivorship-free universe.
- **Honest down-cap finding (GKX 2020 §IV.D):** on the strict rolling top-2,000 large-cap subset, FF5 α collapses to +1.8%/yr at t=0.96 (n.s.) — alpha lives in the small/mid-cap tail where capacity is the binding constraint at deployable AUM. Regime overlay is universe-dependent (works on strict-S&P, net-zero on broad due to COVID monthly-lag).

## Brief Report

**Scope.** We test whether **machine-learning cross-sectional alpha** on US equities survives survivorship-bias correction, PIT eligibility filters, realistic costs, and multiple-testing penalties on a single ~13-year OOS path (2012–2024). Pipeline: monthly-rebalanced dollar-neutral long-short with **sector-neutral construction** (k=20 per GICS sector ≈ 440 positions), trained via **walk-forward expanding-window cross-validation** (120-month training window, block-gated refit every 12 months).

**Method.** Three models with a shared `fit`/`predict` interface — **Lasso** (regularised linear baseline), **XGBoost** (canonical gradient-boosted ensemble, **Optuna**-tuned via TPE on a 2017–18 validation window), and a small **PyTorch NN** — on a **14-feature GKX-style panel** (price-trend, volatility, liquidity, value, quality, plus `chmom`). Target = **sector-relative monthly return**. Evaluation: annualised Sharpe (Newey-West-corrected), **FF5 alpha with HAC SE**, information coefficient (Spearman). Model comparison via **Diebold-Mariano**; XGBoost wins on every window. Robustness battery: 3/3 **sanity gate** (Random/Oracle/Uniform on synthetic panel), block-bootstrap (6-mo blocks), DSR at N=25, **Carhart 6F momentum control**, dense **k-sweep** (k∈[1,100] + bootstrap-CI plateau zoom), single-name leave-one-out, and the **feature-shuffle placebo** (the cleanest leakage test). Regime overlay: walk-forward Gaussian **HMM** on 6 macro-financial features, leverage-only rule.

**What was left out:** long-form audit narrative (in `PIT_INVESTIGATION_REPORT.pdf`); per-window FF3 regressions; Optuna trial-by-trial logs; the regime HMM's IS-vs-OOS crisis-detection-rate breakdown (91.7% / 51.1%); pre-correction lineage Phases 1–15.

**Main references** *(beyond course material on Lasso / gradient boosting / NN / cross-validation / hyperparameter tuning):* Gu, Kelly & Xiu (2020) *Empirical Asset Pricing via Machine Learning,* RFS 33(5) — methodological template + feature set; Bailey & López de Prado (2014) — Deflated Sharpe Ratio; Fama & French (2015) — FF5 model; Carhart (1997) — momentum control; Newey & West (1987) — HAC standard errors; **Sharadar/Nasdaq Data Link** — primary data source under licence. Full citation list in the appendix.

## Description of Appendices

**Optional appendix material** (single combined PDF, in the order it appears): full long-form report with all tables and figures (`report/REPORT.md`, ~30 pages when typeset); the project decision log with chronological provenance for every methodological choice (`DECISIONS.md`, 1,233 lines); the survivorship-leak audit and methodology corrections (`PIT_INVESTIGATION_REPORT.pdf`); robustness-check artefacts (Phase 25 Deflated Sharpe + bootstrap; Phase 26 leave-one-out; Phase 27/27b dense k-sweep with CI plateau); regime-overlay ablation and overlay-failure diagnostic; all source code under `src/` and `notebooks/{persona,personb,personc}/`. **Total submitted PDF size:** *(to be measured after compilation; target < 30 pages, single PDF)*.

---

*Project repository:* https://github.com/bowenzuo119-hash/ml-factor-investing-2026
