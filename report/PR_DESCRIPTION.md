# PR — Final canonical + submission deliverables (Phase 24-RT)

> Use this as the PR body if opening a "wrap-up" PR on GitHub, OR as a reference summary for teammates and the examiner. Mirror of the current `main` state as of `13bc674`.

---

## Summary

Final merge bringing `main` to submission-ready state for the ML factor-investing 5-week course project. End-to-end pipeline: data lane (Sharadar, survivorship-free) → alpha model (XGBoost on 14 GKX-style features) → regime overlay (HMM, Andrea's workstream). Includes full audit trail, robustness battery, and the three compiled submission PDFs.

**Headline numbers** (Phase 24-RT, broad ~4,400-name survivorship-free US equity universe, walk-forward 2012–2024, 10 bps/side costs, both filter bugs corrected):

| Window | Sharpe | FF5 α/yr | t-stat |
|---|---|---|---|
| **Full-OOS 2012–2024** | **+1.15** | **+18.73%** | **+6.85** |
| Long-OOS 2015–2024 | +0.97 | +19.10% | +6.00 |
| Test-OOS 2019–2024 | +1.00 | +21.17% | +5.00 |

**vs S&P 500 over the same window:** we beat passive on Sharpe (+1.14 vs +0.99), on calendar-year win rate (84.6% — 11 of 13 years), and on return (+34.7%/yr vs +14.3%/yr). The Sharpe edge holds up to **~25 bps/side** transaction costs (4× our 10 bps headline; well above the 6–15 bps realistic range per Frazzini-Israel-Moskowitz 2018).

## What's in this PR

### Code

- `src/data_loader.py` — corrected `is_bankruptcy_ticker()` gated on SHARADAR `isdelisted == 'Y'` (the original `endswith('Q')`-only rule wrongly dropped NDAQ and IONQ).
- `src/regime.py` — slimmed to leverage-only overlay; optional k/quantile cols parsed if present (backwards-compat).
- `src/backtest.py` — engine v0.5.0 (PIT filter on both training + trading via `eligible_universe_fn`, `apply_pit_to_training` flag).
- `notebooks/personb/24_canonical_with_chmom.py` — final canonical driver, INCLUDE_FEATURES subset enforced (catches the silent 16-feature bug Bowen found).
- `notebooks/personb/{24a,24b}_*.py` — Optuna retunes (14-feature 24a wins; 16-feature 24b rejected).
- `notebooks/personb/25_statistical_robustness_broad.py` — Phase 25 DSR + bootstrap on the corrected pkl, `N_TRIALS=25`.
- `notebooks/personb/{26_name_concentration, 27_k_sweep_dense, 27b_k_sweep_plateau, qa_figures, vs_sp500_figures}.py` — robustness + presentation-defense figure generators.
- `notebooks/persona/*` — Bowen's data lane + audit scripts (canonical_qfix_validate, decompose_qfix, canonical_true_top2000, check_momentum_factor, placebo_shuffle_features, regime_crisis_detection_rate, etc.).
- `notebooks/personc/{week1,week2,week3}_*.py` — Andrea's regime overlay (HMM + GMM, walk-forward, `.bfill()` look-ahead bug fixed).

### Result artefacts

- `results/24_canonical_with_chmom/per_model_results.pkl` — **the authoritative canonical pkl** (corrected Q-filter + INCLUDE_FEATURES subset).
- `results/25_statistical_robustness_broad/summary.json` + bootstrap distribution figure.
- `results/26_name_concentration/{leave_one_out, top_contributors}.csv`.
- `results/27_k_sweep_dense/{sweep_metrics.csv, k_sweep_dense.png}` — 37 k values 1..100.
- `results/27b_k_sweep_plateau/{plateau_metrics.csv, k_sweep_plateau_zoom.png}` — k=10..20 with bootstrap CIs.
- `results/qa_figures/{placebo_vs_real, model_comparison, where_alpha_lives, momentum_control}.png` — 4 Q&A defence figures.
- `results/vs_sp500/{cumulative_growth, drawdown_comparison, rolling_sharpe, cost_sweep_vs_sp, annual_returns, risk_return_scatter, rolling_correlation, return_distribution}.png` — 8 vs-S&P comparison figures + summary JSON + cost-sweep table.
- `results/long_short_decomp/`, `results/final_canonical_plots/`, `results/persona_figures/` — supporting visuals (long/short P&L decomposition, equity curves, phase progression, etc.).
- `results/regime_overlay_rules.csv` (slimmed leverage-only), `results/regime_walkforward_labels.csv` (Andrea's audit input).

### Submission deliverables — compiled PDFs

- `report/build/onepager.pdf` (1 page) — the mandatory single-page submission per the professor's template.
- `report/build/slides.pdf` (18 slides) — Beamer presentation deck for the ~10-min talk; speakers Bowen + Nicolas (Andrea absent).
- `report/build/appendix.pdf` (15 pages) — optional combined appendix (full REPORT.md + DECISIONS.md audit-era extract).

Rebuild any of these with `bash scripts/build_pdfs.sh` (requires pandoc + xelatex; both already on the machine).

### Documentation

- `report/REPORT.md` (~900 lines) — full long-form report with Abstract, §1 Introduction, §2 Data & Infrastructure, §3 Alpha Model, §4 Regime Overlay, §5 Integrated Results (incl. headline, cost-grid, S&P comparison, momentum control, placebo, k-sweep plateau), §6 Limitations (DSR, IC vs Sharpe, annualisation convention, Q-filter sensitivity, costs & capacity, name-fragility LO), §7 Conclusion, §8 Reproducibility, §9 References (11 entries).
- `report/SUBMISSION_ONEPAGER.md` — markdown source for the 1-page PDF.
- `report/SLIDES.md` — markdown source for the 18-slide Beamer deck.
- `report/PRESENTATION_OUTLINE.md` — slide-by-slide speaking outline with timings.
- `report/QA_PREP.md` — 25 anticipated examiner questions across 4 tiers with full answers and assigned speakers.
- `report/QA_FIGURE_MAP.md` — figure-to-question lookup table for the live Q&A.
- `report/PRE_PR_CHECKLIST.md` — pre-submission audit checklist (all green).
- `report/PROFESSOR_INSTRUCTIONS.md` — verbatim grading + deliverables spec + template structure.
- `report/PR_DESCRIPTION.md` — this document.
- `DECISIONS.md` (1,233 lines) — chronological decision log; every methodological choice has a dated entry with rationale and revisit conditions.

## Methodology highlights

- **Walk-forward expanding-window cross-validation** with 120-month training window, block-gated refit every 12 months.
- **Three models behind a single Protocol interface** (`fit`/`predict`): Lasso baseline, XGBoost canonical, PyTorch NN secondary. Comparison via **Diebold-Mariano test** with Newey-West HAC SE.
- **Hyperparameter tuning via Optuna TPE** (60 trials per model, objective = OOS R² vs zero on 2017-18 validation window).
- **Survivorship-free universe** via SHARADAR/TICKERS with PIT eligibility (`firstpricedate ≤ asof ≤ lastpricedate`).
- **Sector-neutral construction** (top-k / bottom-k per GICS sector, k=20 ≈ 440 positions, dollar-neutral).
- **Robustness battery** (in the report, all reproducible from committed scripts):
  - 3/3 **sanity gate** (Random / Oracle / Uniform synthetic predictors)
  - **Block bootstrap** Sharpe CI (6-mo blocks, 10k iterations, P(SR ≤ 0))
  - **Deflated Sharpe Ratio** at N=25 trials (Bailey-López de Prado 2014)
  - **Carhart 6F momentum control** (FF5 + UMD) — alpha rises to +20.1%/yr at t=+7.4
  - **Feature-shuffle placebo** — Sharpe collapses +1.15 → −0.94 when feature→ticker mapping is permuted (rules out engine/target/cost leakage)
  - **Dense k-sweep** (37 values 1..100) + plateau-zoom with bootstrap CIs (k=10..20 statistically indistinguishable)
  - **Single-name leave-one-out** (n=11, underpowered but suggestive of single-name fragility)
  - **Cost grid** stress test (significant up to ~50 bps/side gross)
- **Honest counterweights documented in §6** — not market-neutral (Mkt-β = +1.5, ML-output-emergent), down-cap concentration (strict top-2,000 α n.s.), capacity-binding at very large AUM, regime overlay net-zero on broad book due to COVID monthly-frequency timing lag.

## Audit trail (the project's methodological contribution)

Three corrections caught mid-project, all transparently reported:

| Phase | Sharpe (long-OOS) | What happened |
|---|---|---|
| Phase 14 (pre-audit) | **+1.49** | Apparent canonical; survivorship leak in engine (filtering by panel membership, not PIT S&P 500 membership) |
| Phase 15 (PIT applied) | **−0.31** | After enforcing `eligible_universe_fn` — the entire +1.49 was 100% survivorship leak |
| Phase 22 (S&P-only honest) | +0.31 (n.s.) | After PIT correction; no significant FF5 alpha at S&P-500 scale |
| Phase 23g (broad rebuild) | +0.95 (t=+5.3) | First significant FF5 alpha after rebuilding on broad Sharadar universe |
| **Phase 24-RT (FINAL)** | **+0.97 long / +1.15 full** (t=+6.85) | + GKX `chmom` feature; Q-filter bug + INCLUDE_FEATURES bug both fixed |

The path from +1.49 → −0.31 → +1.15 is the **methodological contribution**; the alpha number is the empirical contribution.

## Test plan / verification (everything reproducible from `main`)

- [x] `KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python -m src.sanity` returns **3/3 PASS** (random −0.51, oracle +99, uniform flat)
- [x] All 28 phase scripts import cleanly (`import smoke-test` in pre-PR audit)
- [x] All 9 figure paths in REPORT.md resolve to existing files
- [x] All 25+ key result artefacts present under `results/`
- [x] No stale TODO / placeholder markers in REPORT.md
- [x] References cross-check: all 11 entries in §9 cover all in-text citations
- [x] No stale `is_q_suffix_bankruptcy` references remain; 15 import sites use `src.data_loader.is_bankruptcy_ticker`
- [x] Headline numbers consistent across Abstract / §3 / §5 / §6 / §7 (Sharpe +1.15, α +18.73%/yr, t=+6.85)
- [x] DECISIONS.md tail has 8 dated 2026-05-24 entries covering the full audit + correction sequence
- [x] All 3 submission PDFs build cleanly from `bash scripts/build_pdfs.sh` (onepager 1 page, slides 18 pages, appendix 15 pages)

## Workstream sign-offs

- **Person A (Bowen — data lane + audit + integration):** §2 + DATA_AND_ENGINE_SECTION signed; figures regenerated against corrected pkl; sanity 3/3 pass; Carhart momentum control + out-of-time test + feature-shuffle placebo independently verified; q-fix decomposition (NDAQ legit, IONQ fragile) reproducible.
- **Person B (Nicolas — alpha model + robustness + report):** §3 + §5 + §6 + §7 + §1 Intro + §8 Reproducibility + §9 References written; Phase 24-RT canonical locked; robustness battery (Phase 25/26/27/27b) ran on corrected pkl; vs-S&P figure suite + Q&A prep doc + presentation outline + slide deck source built.
- **Person C (Andrea — regime overlay):** §4 written + signed off; IS-vs-OOS crisis-detection-rate audit committed (`notebooks/persona/regime_crisis_detection_rate.py`); regime walk-forward labels CSVs committed; `.bfill()` look-ahead bug in week2/week3 fixed (caught by Person B's regime audit). Andrea is not present at the live presentation; her workstream is summarised briefly by Bowen on slide 8.

## Notes for the examiner

- **Per the professor's brief**, the grade is based on the **presentation + 10+ min Q&A**, not the PDFs directly. The PDFs serve as reference material to formulate questions.
- The repo is **fully reproducible from a clean clone** with a Sharadar/Nasdaq Data Link API key + the `mlfactor` venv (Python 3.12, numpy<2 per `setup_macos_gotchas`).
- The complete chronological decision log lives in `DECISIONS.md` — every methodological choice has a dated entry with "Context / Decision / Reasoning / Revisit if" structure.
- **The project's contribution is methodological as much as empirical:** the audit-driven journey from a leaky +1.49 Sharpe to an honest +1.15 demonstrates that ML pipelines need to be audited as rigorously as the models themselves.

🤖 Reference: see `report/QA_PREP.md` for 25 anticipated examiner questions with full answers across ML methodology, evaluation, audit, costs, regime overlay, and adversarial framings.
