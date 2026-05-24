# Q&A Prep — 10+ min discussion after the presentation

> **Context:** the professor's brief explicitly says the **10+ min Q&A is part of the grade** ("how well you convey your understanding of the machine learning aspects that relate to your project"). Examiner will dig into ML methodology, evaluation, reproducibility, and honest limitations. This document anticipates likely questions and pre-loads answers.
>
> **Speakers:** Nicolas (N) + Bowen (B). Andrea is absent — regime-specific deep questions route per the fallback policy in `PRESENTATION_OUTLINE.md`.
>
> **Style of answers below:** short, direct, ML-flavoured. **Do not bluff** — for anything we genuinely don't know, the honest answer is the right one ("we didn't test that — it would be the obvious next step"). Pre-emptive honesty earns more credit than confident guessing.
>
> **Rehearsal note:** print this doc, mark the questions each speaker is responsible for in colour. Goal: in the actual session, the *named* speaker takes the first crack; the other can supplement.

---

## Tier 1 — Almost certain (the examiner will ask at least 3 of these)

### Q1 — "How do you know the +1.15 Sharpe isn't a backtest artefact?"

**Answer (N):** Three independent leakage tests. First, a **3/3 sanity gate** on a synthetic panel: a Random predictor gives Sharpe ≈ 0, an Oracle predictor gives Sharpe → ∞, a Uniform predictor gives flat returns — so the engine + cost + target machinery don't manufacture signal on their own. Second, a **feature-shuffle placebo** — run the exact 14-feature canonical recipe but randomly permute the feature-vector → ticker mapping within each rebalance date. Real features give +1.15 Sharpe; shuffled give **−0.94** (mean of 2 seeds). The 2.1-Sharpe swing goes *negative* because turnover-cost drag wins when there's no signal. Third, **independent reproduction** — Bowen wrote a separate `canonical_qfix_validate.py` two-arm script that loads the same data and model spec independently of the main driver, and reproduced the +1.15 within recipe noise. **Source:** `notebooks/persona/placebo_shuffle_features.py`, `src/sanity.py`, `notebooks/persona/canonical_qfix_validate.py`.

**Trap to avoid:** don't lead with "it's significant" — examiner has heard p < 0.001 a hundred times. Lead with the placebo and the sanity gate, because they directly answer the *engineering* question of leakage.

---

### Q2 — "Why XGBoost over the neural net or the linear baseline?"

**Answer (N):** We ran **all three with the same interface** (`fit`/`predict` Protocol) on the same panel and compared via the **Diebold-Mariano test** (adapted Gu-Kelly-Xiu variant with Newey-West-HAC standard errors on per-month squared-error differences). XGBoost beats Lasso at p < 0.01 and beats NN at p < 0.05 on both MSE and Sharpe reductions across every reporting window. The intuition: the cross-sectional task has ~250 monthly observations × ~5,000 names, with substantial non-stationarity across the 2008/2020 regime changes — that effective sample size doesn't favour a deep model without much heavier regularisation, and trees handle the noisy / fat-tailed feature distributions better than a linear baseline. Per-window XGBoost vs NN vs Lasso headline: Sharpe +1.15 / +0.62 / +0.71 (full-OOS). **Source:** REPORT.md §3 + §5 model-comparison table; `src/metrics.py::diebold_mariano`.

**Trap to avoid:** don't say "deep learning is overrated." Say "the data regime favours trees" — specific, defensible.

---

### Q3 — "What's your hyperparameter tuning strategy and how do you avoid overfitting to validation?"

**Answer (N):** **Optuna with TPE sampler**, 60 trials per model, **objective = OOS R² vs zero on a held-out validation window (2017–18)**. Tuning is done on a Q-filtered training panel that *excludes* the test-OOS window (2019+). The walk-forward backtest then evaluates on the *unseen* test window — so validation is used for hyperparameter selection, test is used for final scoring, and those windows don't overlap. Optuna's TPE is a Bayesian-style tree-structured Parzen estimator that converges faster than grid search and handles mixed continuous/integer search spaces (`n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_alpha`, `reg_lambda`). Seeded `random_state=42` for reproducibility. **Source:** `notebooks/personb/24a_retune_xgb_with_chmom.py`; best-params file `results/24a_retune_xgb_with_chmom/best_params.json`.

**Trap to avoid:** don't claim "no overfitting." Honestly: we ran ~25 trials across the project lineage (Phases 23a-24c), which is why we have the **Deflated Sharpe Ratio at N=25** in §6 — to deflate for that multiple-testing exposure.

---

### Q4 — "How do you handle look-ahead bias?"

**Answer (B):** Four mechanisms. **(1) Point-in-time universe filter** (`eligible_universe_fn`) on both training labels and trading-time eligibility — a stock is in the panel for a given month iff its `firstpricedate ≤ asof ≤ lastpricedate` in SHARADAR's TICKERS table. A delisted name drops out only *after* its last price, never before. **(2) Walk-forward expanding-window** training: the training set at month t contains only data with `date < t`. **(3) Sector map** is derived from the *features* index per date (so a stock that gets reclassified later doesn't backfill the earlier classification). **(4) Fundamentals are AS-REPORTED** (SF1 ARQ/ART), not restated — so the model sees what was actually published at the time. We also caught a survivorship leak mid-project (the engine was originally filtering by panel membership rather than PIT eligibility, leading to a +1.49 → −0.31 Sharpe collapse when fixed) — disclosed in §6 of the report and in `PIT_INVESTIGATION_REPORT.pdf`. **Source:** `src/backtest.py`, `src/data_loader.py::load_universe_at`.

**Trap to avoid:** don't gloss over the audit. Mentioning that *we caught our own leak* signals intellectual honesty.

---

### Q5 — "What baseline are you comparing against?"

**Answer (N):** Three layers of baseline. **(1) Random / Oracle / Uniform** synthetic predictors (the 3/3 sanity gate) — these confirm the *engine* doesn't manufacture signal. **(2) Lasso linear regression on the same 14 features** — full-OOS Sharpe +0.71, FF5 α +10.4%/yr at t=+2.40. XGBoost beats this by +0.44 Sharpe / +8 pp α / +4.5 t-stat units. **(3) Phase-progression comparison** — Phase 14 (leaky pre-audit, S&P-500, k=5) gave apparent +1.49 Sharpe; Phase 22 (S&P honest, PIT-applied) gave +0.31 with no significant α; Phase 24-RT (broad survivorship-free, k=20, 14 features) gives +1.15 with α at t=+6.85. So we're showing the result against (i) sanity, (ii) a linear baseline, and (iii) the un-audited self-baseline. **Source:** §5 phase-progression table; §3 model-comparison.

**Trap to avoid:** don't just say "we compare to Lasso." Show the *three layers* — sanity, model baseline, and self-baseline (audit history).

---

### Q6 — "Why a 14-feature panel? How did you choose features?"

**Answer (N):** The feature set is the **Gu-Kelly-Xiu (2020) §IV.B top-feature stack** plus our additions. 13 features come from their published importance ranking (12-1 momentum, short-term reversal, return vol, idiosyncratic vol vs CAPM, log market cap, log dollar volume, book/market, earnings/price, ROE, ROA, D/E, asset growth, accruals). We added **`chmom` (change in 6-month momentum)** — GKX's rank #4 feature — verified to be orthogonal to existing momentum (|corr| < 0.06). We tested adding `maxret` and `mom36m` in Phase 24b (the next two GKX features) but the 16-feature variant *underperformed* on validation R² (+0.0046 vs +0.0055) and on walk-forward Sharpe (+0.97 vs +1.15). Bias-variance: at our effective sample size, the extra two features cost more in tuning instability than they earned in signal. **Source:** REPORT.md §3 feature-list; DECISIONS.md 2026-05-24 Phase 24-RT entry for the 14-vs-16 A/B.

**Trap to avoid:** don't claim feature engineering credit — the heavy lifting is GKX. Our contribution is the *honest evaluation* on a survivorship-free universe, not the feature set.

---

### Q7 — "What's the Deflated Sharpe Ratio and why N=25 trials?"

**Answer (N):** Bailey & López de Prado (2014). The intuition: if you try N strategies on the same OOS window, the *best* one will have an apparent Sharpe inflated by the maximum-of-N-draws statistic — roughly √(2 ln N) standard errors above the population mean under the null. DSR deflates the observed Sharpe by this expected-max-under-the-null and turns it into a posterior probability that the *true* Sharpe exceeds that expected maximum. We bumped N from 10 to **N=25** to honestly count every configuration we evaluated on the same OOS window during model development (Phases 23a–24c lineage + Optuna retunes + k-sweeps + cost-grid). Result: DSR = 0.85–0.88 across windows — even after the harder penalty, ~85–88% posterior probability that the true Sharpe exceeds the expected-max-under-null. **Source:** `notebooks/personb/25_statistical_robustness_broad.py`; `results/25_statistical_robustness_broad/summary.json`.

**Trap to avoid:** don't claim "DSR proves the alpha is real" — DSR is a *penalty*, not a proof. The honest framing is "even after penalising for 25 trials, the result clears the conventional 0.5 cut-off comfortably."

---

### Q8 — "Why is the alpha so high (~+18%/yr) — isn't that suspicious?"

**Answer (N):** Two pieces of decomposition. **(1) Where the alpha lives:** on the strict rolling top-2,000 by market cap (large/mid-cap end of our universe), the FF5 α collapses to **+1.8%/yr at t=0.96 — not significant**. SMB loading also collapses (+1.26 → +0.15). The headline +18.7% is a *down-cap effect* concentrated in names below the current top-2,000. This is **exactly Gu-Kelly-Xiu's prediction** — and the down-cap tail is where small-cap costs and capacity become the binding constraints at deployable AUM. **(2) Decomposition of the realised return:** Mkt-β ≈ +1.3, so ~55% of the headline annual return comes from leveraged market exposure; the remaining ~45% is genuine FF5 alpha. So the "+18%" is real cross-sectional skill, but the strategy is *not* market-neutral, and the alpha is *not* harvestable at scale without paying real small-cap impact costs. **Source:** REPORT.md §5 "Where the alpha lives" + §6 "Costs and capacity"; `notebooks/persona/canonical_true_top2000.py`.

**Trap to avoid:** don't get defensive. Lead with the GKX-consistency framing — "yes, it's a known finding that ML alpha lives down-cap, and we confirm it independently."

---

### Q9 — "How do you know the +18% alpha isn't just momentum?"

**Answer (N):** The natural test is a **Carhart 4-factor regression** (FF5 + UMD). We ran it: FF5-only α is +17.7%/yr at t=+6.11; **FF5 + UMD α RISES to +20.1%/yr at t=+7.40**, with UMD loading **−0.43 at t=−4.61**. The portfolio is **momentum-AVERSE** (short loading on the UMD factor) — and the alpha actually *increases* when momentum is added as a control because the FF5-only spec was modestly under-stating alpha by missing the short-momentum tilt. So the headline α is not repackaged momentum premium; it survives any reasonable momentum control. **Source:** `notebooks/persona/check_momentum_factor.py`; REPORT.md §5 "Momentum control" table.

**Trap to avoid:** don't say "we use Carhart because it's standard." Explain the UMD coefficient — it being negative is the *content* of the result.

---

### Q10 — "Walk through the audit story — what went wrong and how did you catch it?"

**Answer (B):** Three corrections, two of them caught by us mid-project. **(1) Survivorship leak (week 4):** our backtest engine was originally filtering eligibility by *panel membership* (any ticker with a non-NaN return for that month), not by *point-in-time* S&P 500 membership. We had pre-join history for TSLA, ENPH, GNRC, NOW in our panel — and the engine was happily trading them as if they'd been S&P 500 members before joining. A 2012–2019 RandomModel run traded 726 non-member positions. We added `eligible_universe_fn` to the engine, and the S&P-500 canonical's Sharpe collapsed from **+1.49 to −0.31** — the entire +1.49 was 100% survivorship leak. **(2) Q-filter bug (week 5):** the bankruptcy filter `endswith('Q') AND len ≥ 4` wrongly dropped NDAQ (Nasdaq Inc.) and IONQ (IonQ Inc.) — both alive common stock. We re-gated on SHARADAR's `isdelisted == 'Y'` field; un-dropping these two names moved the headline by +0.12 Sharpe. **(3) INCLUDE_FEATURES bug (week 5):** the canonical driver wasn't subsetting features to its declared 14-feature list, so when we added `maxret` and `mom36m` to the same parquet for the Phase 24b test, the committed pkl silently became the 16-feature variant (which we'd separately rejected as worse). Both bugs partially cancelled in the previously-committed pkl. We fixed both, re-froze the pkl, and the corrected canonical is the +1.15 Sharpe / +18.7% α reported in §5. **Source:** `PIT_INVESTIGATION_REPORT.pdf` + DECISIONS.md tail entries 2026-05-23 / 2026-05-24.

**Trap to avoid:** don't try to make this a "we caught it because we're so smart" story. The honest framing is *we built the audit into the methodology, and the audit did its job*.

---

## Tier 2 — Likely (be ready, ~50/50)

### Q11 — "Why sector-neutral construction (k=20 per GICS sector) and not global top/bottom quantile?"

**Answer (N):** Sector-neutral is the third layer of GKX's construction stack — global top/bottom quantile would concentrate the book in whatever sector the model is most bullish on at the moment (sector-tilt risk). Sector-neutral forces ~440 positions split evenly across the 11 GICS sectors, so the book is **diversified across sectors by construction** and the residual sector tilt is small. k=20 specifically: we ran a **dense k-sweep** (k ∈ [1, 100], 37 values) on the corrected pkl and found a **flat plateau between k=10 and k=20** with the Sharpe curve falling sharply below k=5 (concentration + turnover drag) and decaying smoothly above k=25 (over-diversification). A follow-up **plateau-zoom with bootstrap CIs** (k=10..20 with 2,000 bootstrap iterations each) confirms k=20's Sharpe falls inside the 90% CI of every other k in that range (11/11 = 100%) — so k=20 is statistically indistinguishable from the peak. We chose 20 ex-ante for round-number defensibility; the dense sweep validates that empirically. **Source:** `notebooks/personb/27_k_sweep_dense.py`, `notebooks/personb/27b_k_sweep_plateau.py`.

---

### Q12 — "What about transaction costs? Is 10 bps/side realistic?"

**Answer (B):** 10 bps/side is the optimistic end for the strategy's actual cap profile, and we explicitly flag this as the binding limit. The **cost grid** sweeps 10 / 30 / 50 / 75 bps/side: at 10 bps the alpha is +18.7%/yr (t=+6.85); at **30 bps/side** (the recommended conservative basis) the alpha drops to ~+12.1%/yr at t=+4.39 — **still significant**; at 50 bps it's still at t=+2.8; dies around 75 bps. So the alpha survives reasonable cost stress, but the down-cap tail of our universe (where the alpha actually lives) has bid-ask spreads + market impact that can exceed 30 bps/side on routine trades and rise sharply with order size. We did **not** run an Almgren-Chriss-style size-impact-aware cost model — that's the obvious extension and we flag it in the conclusion. The 175% monthly turnover at ~440 positions also implies ~770% NAV throughput per month, which is the capacity constraint. **Source:** `notebooks/persona/cost_sensitivity_phase23.py`; §6 "Costs and capacity".

---

### Q13 — "How do you do statistical inference on a single OOS path?"

**Answer (N):** Two complementary tools. **(1) Block bootstrap** on the realised monthly returns: resample contiguous 6-month blocks (with replacement) to a series of original length, recompute Sharpe, repeat 10,000 times — gives a 5–95% CI on the Sharpe and a p-value for `P(SR ≤ 0)`. Block resampling preserves any short-horizon autocorrelation in returns. Long-OOS P(SR ≤ 0) = 0.0002. **(2) Newey-West HAC standard errors** on the FF5 regression (6-month lag): corrects the alpha t-stat for heteroskedasticity and autocorrelation in the residuals, which simple OLS would understate. The FF5 alpha t-stat we report (+6.85) is the Newey-West-corrected one, not the naive OLS one. We use the Politis-Romano stationary bootstrap as a more sophisticated alternative — not done here, but would be the next-level rigor check. **Source:** `src/metrics.py::block_bootstrap_sharpe`, `notebooks/persona/verify_phase23_headline.py::nw_ols`.

---

### Q14 — "What's the regime overlay's mechanism and why doesn't it help on the broad book?"

**Answer (B):** *(Andrea owns the detail; brief answer with appendix pointer.)* Andrea fit a **Gaussian Hidden Markov Model** with 2 states (`calm` / `crisis`) on 6 macro-financial features (realised vol, VIX, term spread, credit spread, 3-mo S&P return), walk-forward expanding window with 60-month burn-in and per-step `StandardScaler` refit. Overlay rule: calm → 1.00× leverage, crisis → 0.40×. **Result is universe-dependent.** On the strict-S&P canonical, DD improves −25.5% → −19.9% with a small Sharpe gain. On the broad canonical, DD is **unchanged at −33.8%** because the deepest drawdown is the Feb-Mar 2020 COVID crash — the HMM correctly flags March as crisis, but the overlay sets leverage from the *prior* month-end label, so January and February were both 'calm' and we entered the crash at full leverage. **This is a monthly-frequency timing limit, not a model flaw.** The reproducible audit script is `notebooks/persona/regime_crisis_detection_rate.py`. For deeper questions on HMM specifics (state-count choice, transition matrix, IS vs OOS detection rate breakdown), happy to arrange a follow-up with Andrea after the session. **Source:** `notebooks/personc/week3_regime_finalise.py`; §4 + appendix.

---

### Q15 — "Why a long-short construction and not long-only?"

**Answer (N):** Three reasons. **(1) Cross-sectional ML predictions have value in both tails** — if the model can rank stocks, the top-decile alpha and the bottom-decile alpha (with sign flipped) are both informative. Long-only throws away the bottom-half information. **(2) Dollar-neutral construction reduces overall market exposure** in principle (though we show Mkt-β ≈ +1.3 in practice because of the asymmetric beta selection — see §5). **(3) Sharpe ratio is the primary metric** and long-short with the same signal beats long-only on Sharpe in our setup. But: §5 also shows the **long-leg decomposition** — the long leg alone makes +37.9%/yr (Sharpe +1.16) while the short leg makes −2%/yr (Sharpe −0.47, near-zero P&L). So the long leg is doing essentially all the work; the short leg is a market-neutralising hedge that contributes very little direct P&L. The "long-short" framing is honest at the L1-turnover level but the realised P&L is **long-leg dominated**. **Source:** `notebooks/personb/long_short_decomp.py`; §5 "Honest characterisation" subsection.

---

### Q16 — "What's the dataset like? How many examples, how many features, how much memory?"

**Answer (B):** Monthly panel, 2002-04 to 2024-12. **~5,500 unique tickers** across the broad survivorship-free universe (median ~4,400 active per month). **735,827 (date, ticker) rows** in the predictions matrix; **14 features + 1 sector tag** in the features parquet (1.24M rows × 17 cols). Total raw data on disk: ~50 MB (returns parquet 11 MB, features parquet 30 MB, predictions 12 MB, the canonical pkl 20 MB). Walk-forward training set at each step: 120 months × ~3,000 names average = ~360,000 (X, y) rows per fit. XGBoost fits in ~5–15 seconds per step on Bowen's machine (where the Sharadar raw data lives); Lasso in <1 sec; NN in ~30 sec (CPU PyTorch). Walk-forward has 155 prediction steps over 2012–2024 → ~30 min total for the canonical end-to-end. **Source:** REPORT.md §2 + §3; data files in `data/processed/`.

---

### Q17 — "What's the loss function?"

**Answer (N):** For XGBoost, **squared loss** on the (date, ticker) target = next-month realised return demeaned by per-(date, sector) mean (the sector-relative target). For Lasso, same squared loss. For the NN, MSE loss with Adam optimiser. The target demeaning is the GKX Layer-2 refinement — by subtracting the per-(date, sector) mean from each y value before training, the model learns to predict *relative-to-sector* returns rather than absolute, which removes the (large, common) sector-level signal that the model would otherwise spend capacity fitting. Selection-time scoring then ranks by the model's raw prediction, picks top-k / bottom-k per sector, dollar-neutral. **Source:** `src/models.py::XGBoostModel::fit` (the `target_kind="sector_relative"` branch).

---

### Q18 — "Did you check for feature leakage in specific features?"

**Answer (N):** Yes, and we caught one. We ran the **placebo shuffle** (Q1 answer) which destroys the feature → ticker mapping — the strategy's Sharpe collapses to −0.94, confirming the alpha needs *genuine feature content*. We also checked **orthogonality** of newly-added features (`chmom`) against the existing 13 — |corr| < 0.06 across the panel, so chmom is genuinely new signal not a re-encoding of momentum. We did *not* run per-feature shuffles (drop one feature, retune, see how much α drops) — that would be the next granularity of leakage check; we flag it as an extension. The features themselves are lagged appropriately: momentum is 12-1 (skip last month to avoid the short-term reversal effect), volatility is realised over the trailing 12 months ending at month t (using prices observable by month-end), all fundamentals are AS-REPORTED (SF1 ARQ/ART), not restated. **Source:** `notebooks/persona/placebo_shuffle_features.py`; orthogonality check in `notebooks/personb/compute_chmom_maxret_features.py`.

---

## Tier 3 — Possible (have a one-liner ready)

### Q19 — "How would this generalise to non-US equities?"

**Answer (N):** We didn't test. The methodology transfers (PIT universe, walk-forward, sector-relative target, XGBoost on GKX features), but: (i) Sharadar is US-only — would need an equivalent survivorship-free panel for international markets; (ii) the GKX feature stack was selected on US CRSP data — different markets may favour different features (e.g., earnings yield matters more in Europe); (iii) costs / capacity profile is market-dependent. Likely directional finding would hold (ML alpha lives in the small/mid-cap tail) but magnitudes would differ.

---

### Q20 — "If you had another month, what would you do?"

**Answer (N):** Top 3, in priority order. **(1) Size-impact-aware costs** — implement Almgren-Chriss or Kissell on top of the cap-bucket distribution of the actual positions to get realistic capacity estimates. **(2) Sub-monthly regime detection** — weekly or daily HMM on rolling z-scores to catch the COVID-2020-style fast crashes that monthly frequency misses; would directly address Andrea's overlay limitation. **(3) Cliff-style leave-one-out robustness** on the full universe (every name with a top-decile lifetime contribution, with bootstrap CIs on each Δ Sharpe) — the n=11 LO study we did is suggestive but underpowered; the full study would give defensible single-name fragility estimates. Lower priority: per-feature shuffle (Q18), walk-forward Optuna retune cadence (currently one-shot 2017–18), Almgren-Chriss cost modelling instead of flat bps grid.

---

### Q21 — "Why monthly rebalancing — why not weekly or daily?"

**Answer (B):** Three reasons. **(1) Data resolution** — Sharadar SF1 fundamentals update at the 10-Q/10-K cadence (~quarterly), so the marginal information from sub-monthly rebalancing on fundamentals-heavy features (B/M, E/P, ROE, accruals) is small. **(2) Cost amplification** — at 10 bps/side and 175% monthly turnover we already pay ~210 bps/year in costs; weekly rebalancing would push that ~3× higher and would erode the alpha below the breakeven we showed in the cost grid. **(3) GKX 2020 benchmark uses monthly** — we wanted apples-to-apples comparison. The downside is the regime overlay's monthly-frequency lag (Q14) which a sub-monthly setup would address. The honest answer: monthly is a defensible default for cross-sectional factor strategies on fundamentals features, but sub-monthly would be the obvious extension if we were targeting market-microstructure features. **Source:** §6 limitations.

---

### Q22 — "How do you reproduce your results?"

**Answer (B):** All seeds pinned to 42. Data caches are gitignored but rebuild from a clean clone with the data-pull scripts in `notebooks/persona/run_all_data.py` (requires a Sharadar / Nasdaq Data Link key). The canonical regenerates via:

```bash
KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python -m notebooks.personb.24_canonical_with_chmom
```

Robustness suite:

```bash
.venv/bin/python -m notebooks.personb.25_statistical_robustness_broad
.venv/bin/python -m notebooks.personb.26_name_concentration_ablation
.venv/bin/python -m notebooks.personb.27_k_sweep_dense
.venv/bin/python -m notebooks.personb.27b_k_sweep_plateau
.venv/bin/python -m notebooks.persona.placebo_shuffle_features
.venv/bin/python -m notebooks.persona.check_momentum_factor
```

Sanity gate:
```bash
.venv/bin/python -m src.sanity   # must return 3/3 PASS
```

Every methodological choice is logged in `DECISIONS.md` with a date + rationale + revisit-if condition. **Source:** REPORT.md §8 Reproducibility.

---

## Tier 4 — Hostile / adversarial (have a calm answer ready)

### Q23 — "Couldn't this just be a lucky path? You ran a lot of phases."

**Answer (N):** Three responses. **(1) Yes, we tested ~25 configurations, which is why we honestly count N=25 in the Deflated Sharpe Ratio** — and after that penalty the result still clears 0.85 (≫ 0.5 cut-off). **(2) The single OOS path is the one in the data; there isn't a second one we could test on.** We mitigate via the block bootstrap (P(SR ≤ 0) = 0.0002) and by showing consistency across three reporting windows (full-OOS, long-OOS 2015+, test-OOS 2019+) that exclude different startup periods. **(3) The Carhart momentum control + placebo + sanity gates are all *complementary* — they don't share the multiple-testing concern.** A lucky path wouldn't survive all of those. The honest framing is in §6: "This study reports a single ~13-year OOS path on monthly data; we did not test sub-monthly rebalancing, intraday execution, or post-2024 data." We are not claiming "ML factor strategies work" in general; we are claiming the narrower "this specific recipe produces statistically significant alpha on this specific window after these specific controls."

---

### Q24 — "Your strategy has Mkt-β ≈ +1.3 — isn't this just leveraged market exposure?"

**Answer (N):** Partially yes, but the FF5 framework decomposes that exactly. The realised +34.7%/yr annual return splits into roughly +18.7%/yr **pure FF5 alpha** (the residual after the FF5 factor returns are stripped out) and ~+16%/yr **systematic factor exposure** (Mkt-β × Mkt-RF premium + SMB exposure + minor HML/RMW/CMA loadings). So ~55% of the headline return is leveraged market + size exposure, and ~45% is genuine cross-sectional skill. The +18.7%/yr is what the FF5 regression *cannot* explain — that's the residual our model produces. We do not claim market-neutrality; we explicitly characterise the strategy as a **high-beta directional long-short book with significant cross-sectional alpha on top** (§5). The β-hedged pure-alpha curve is plotted in the headline equity-curve figure to make this decomposition visible. **Source:** §5 "Honest characterisation"; `results/final_canonical_plots/equity_curve_phase24_honest.png`.

---

### Q25 — "What's the conflict of interest with your own audit? You graded yourselves."

**Answer (B):** Two pieces. **(1) The audit findings were quantitative, not qualitative.** The +1.49 → −0.31 collapse, the +0.116 Sharpe Q-filter swing, the +0.10 Sharpe INCLUDE_FEATURES swing — all reproducible from the committed scripts on the committed data. They aren't subjective judgments we're handing ourselves a passing grade on. **(2) The audit process is in the git history** — every commit is a separate, time-stamped, reviewable artefact. DECISIONS.md has the chronological log of what we changed and why. Anyone can clone the repo at commit `<pre-audit>` and at commit `<post-audit>` and see the actual difference in numbers. If we'd hidden the audit, the +1.49 would still be in the report; we deliberately kept it as the "before" number to be transparent about the journey. The intellectual-honesty discount earned by self-auditing is bigger than the marginal credit from polishing it under the rug — and we knew that going in.

---

## Hand-off / closing

If the examiner runs out of questions early or asks "anything else you want to add" — close with:

> "The honest takeaway: ML cross-sectional alpha is *real* on a survivorship-free US equity universe under realistic costs, but it lives in the down-cap tail where capacity becomes binding at deployable AUM. The methodology journey — from the leaky +1.49 to the audit-driven +1.15 — is the contribution we most want to be remembered for, because it generalises to any ML-pipeline-on-real-data project: build the audit into the pipeline, run it relentlessly, and publish the journey honestly."

---

## Last-minute rehearsal checklist

- [ ] Each speaker has marked their assigned Q1–Q25 questions
- [ ] Both speakers can answer Q1 / Q4 / Q10 (the integrity questions — anyone might be asked)
- [ ] We have the regime-fallback policy memorised (Q14 — "Andrea owns that detail")
- [ ] We have the closing one-liner memorised
- [ ] One full mock Q&A with a third party (a friend in the program?) playing examiner
