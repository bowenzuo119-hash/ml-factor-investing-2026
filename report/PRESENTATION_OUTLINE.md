# Presentation Outline — ~10 min talk + 10+ min Q&A

> **Target:** ~10 minutes of speaking time + 10+ min Q&A (per professor's brief). Pace: ~1 min per slide on average; **strict budget — over-running the talk eats Q&A time, which is half the grade.**
>
> **Audience:** senior teaching coordinator with strong ML background, possibly limited quant-finance background. Pitch ML concepts at the right level (no need to define Sharpe ratio in slide 1 but do explain it briefly when first used). Q&A will dig deep on ML methods + evaluation + reproducibility.
>
> **Style:** every slide should have a **headline** (one-sentence takeaway in the title) so the examiner can read the deck in 30 seconds. Bullets are for elaboration only.
>
> **Speakers:** **Bowen (B) and Nicolas (N) only.** Andrea (A) is not present at the presentation — her regime-overlay section is compressed to a single brief slide delivered by Bowen (the data-lane / integration owner). The full regime methodology lives in the optional appendix PDF for the examiner to reference if they ask about it in Q&A. If a Q&A question is deeply regime-specific, fall back to "Andrea owns that workstream; we can give you the high-level answer and point you to the relevant script/section in the appendix."

---

## Slide budget — 11 slides × ~55s each = 10 min

> Time freed by compressing Andrea's slot (was 75s, now 45s = save 30s) is reallocated to slide 7 (robustness battery — now 90s, which is where the strongest ML-methodology content lives and where the most examiner questions will land).

| # | Slide title (headline) | Speaker | Time | Key figure |
|---|---|---|---|---|
| 1 | Title + team + 1-line pitch | N | 30s | none (cover) |
| 2 | Question: does ML cross-sectional alpha survive when the backtest is honest? | N | 45s | none / hook quote |
| 3 | Data lane — broad survivorship-free Sharadar universe (~4,400 names/month) | B | 75s | universe_coverage_broad.png |
| 4 | Engine — walk-forward, PIT-correct, sanity-gated | B | 60s | walkforward_scheme.png OR sanity 3/3 chip |
| 5 | Alpha model — 14 features, 3 models, XGBoost canonical | N | 75s | feature_correlation.png OR feature list table |
| 6 | Walk-forward result — XGBoost wins on every window | N | 75s | equity_curve_phase24_honest.png |
| 7 | Robustness — Carhart, DSR, bootstrap, **placebo, k-sweep** (extended) | N | **90s** | placebo + k-sweep small-multiples |
| 8 | Regime overlay (Andrea's workstream — brief mention) | B | **45s** | overlay_failure_regime.png |
| 9 | The audit story — Sharpe +1.49 → −0.31 → +1.15 | B | 60s | phase_progression_phase24.png |
| 10 | Honest counterweights — not market-neutral; down-cap concentration | N | 60s | ff5_decomposition_phase24.png |
| 11 | Conclusion + what we'd do differently | N | 30s | none / takeaway bullets |

**Total:** ~10 min. Built-in 30s buffer for hand-offs. **Speaker split:** Nicolas 6 slides (~6 min), Bowen 5 slides (~4 min).

---

## Slide-by-slide content

### 1. Title (30s — N)

```
Machine-Learning Factor Investing on a Survivorship-Free US Equity Universe

Bowen Zuo — data & infrastructure
Nicolas Couto Mota — alpha model
Andrea Fontana — regime overlay

Question: does ML cross-sectional alpha survive on US equities when the
backtest is honest about survivorship, look-ahead, and costs?
```

**Speaker notes:** straight in. One sentence on the question, hand off to slide 2.

---

### 2. Question + motivation (45s — N)

**Headline:** *"GKX 2020 reported Sharpe > 1.5 — we wanted to know if it survives the audit a sceptical reader would do."*

- Gu-Kelly-Xiu (2020): Sharpe ~1.5 on CRSP 3000–6000 with neural-net cross-sectional regression. Replicated widely, but…
- Three classic failure modes: **survivorship bias, look-ahead leakage, multiple-testing on a single OOS path**.
- Our project: build the pipeline from scratch under strict audit controls, then test whether the headline survives.
- Why this matters for the course: every ML concept lives in this pipeline — walk-forward CV, hyperparameter tuning, model comparison via statistical tests, baseline regression, neural net, gradient boosting, bias/variance trade-off in feature choice.

**Speaker notes:** the audience cares about ML methodology, not the dollars. Lead with the methodological question, dollars come later.

---

### 3. Data lane (75s — B)

**Headline:** *"Broad survivorship-free US universe, point-in-time eligibility — ~4,400 names/month median, not the rolling top-2,000."*

- **Source:** Sharadar (Nasdaq Data Link) — SF1 fundamentals, SEP prices, DAILY market caps, TICKERS security master, SP500 history, ACTIONS for corporate actions. **Single source, no splice.**
- **Universe:** alive set of historical top-2,000 by market cap, filtered by `firstpricedate ≤ asof ≤ lastpricedate`. ~4,400 names/month median; 5,897 unique tickers union over 2002–2024.
- **Critical for honesty:** SEP carries delisted tickers under their *historical* symbols (LEHMQ, ENRNQ, SIVBQ). The panel is survivorship-free by *construction* — not retroactively backfilled.
- **Bankrupt-ticker filter:** drop `endswith('Q') AND isdelisted=='Y'` (corrected: the symbol-only rule wrongly dropped NDAQ / IONQ).

**Figure:** `results/persona_figures/universe_coverage_broad.png` — universe counts over time, S&P 500 vs strict top-2,000 vs ~4,400 canonical vs total.

**Speaker notes:** make the "survivorship-free by construction" point explicit — this is the data-side answer to GKX's classic objection.

---

### 4. Engine (60s — B)

**Headline:** *"Walk-forward, point-in-time, sector-neutral, sanity-gated — single seam between three workstreams."*

- **`run_walk_forward_backtest`** — 120-month sliding training window, block-gated refit every 12 months (test_window), 10 bps/side L1-turnover cost.
- **Point-in-time enforced** via `eligible_universe_fn` on both training and trading; `apply_pit_to_training=True`.
- **Sector-neutral construction:** top-k / bottom-k by sector-relative score, k=20 per GICS sector → ~440 positions, dollar-neutral.
- **Sanity gate (Project Framework §4.6):** 3 synthetic-panel checks must pass every commit — RandomModel Sharpe ≈ 0, OracleModel Sharpe → ∞, UniformModel mean ≈ 0. **3/3 PASS** on the canonical.
- **One seam, three workstreams:** every callsite imports the same engine signature; Person A owns the engine, Person B / C consume it.

**Figure:** sanity 3/3 result chip OR `results/persona_figures/walkforward_scheme.png`.

**Speaker notes:** emphasise the *engineering discipline* — single interface, contract-tested, repeatable. The audience will recognise this as good ML systems hygiene.

---

### 5. Alpha model (75s — N)

**Headline:** *"14 firm features, 3 models share an interface, XGBoost wins decisively on Diebold-Mariano."*

- **Three models, same interface:**
  - **Lasso** — regularised linear baseline (sklearn).
  - **XGBoost** — gradient-boosted trees, canonical. Hyperparameters tuned via **Optuna TPE** (60 trials, validation-window OOS R² vs zero on 2017–18).
  - **PyTorch feed-forward NN** — small (2-layer, dropout), secondary.
- **14 features** spanning price-trend (5: mom 12-1, reversal, vol, ivol, **chmom**), liquidity (2: log mcap, log dvol), value (2: B/M, E/P), quality (4: ROE, ROA, D/E, asset growth, accruals). `chmom` (change-in-6-month momentum) is the GKX top-5 feature; orthogonal to existing momentum (|corr| < 0.06).
- **Target:** sector-relative monthly return (Layer 2 in the GKX framework). The model learns to predict relative-to-sector, not absolute, returns.
- **Model selection:** Diebold-Mariano test, adapted GKX variant with Newey-West HAC SE. XGBoost beats Lasso (p < 0.01) and NN (p < 0.05) on both MSE and Sharpe across every reporting window.

**Figure:** small table comparing the three models' Sharpe + α t-stat per window (from REPORT.md §3 summary).

**Speaker notes:** name-check the ML methods explicitly. Optuna + TPE + Diebold-Mariano + Newey-West are the kind of jargon the examiner will ask about — pre-empt by mentioning briefly.

---

### 6. Walk-forward result (75s — N)

**Headline:** *"Full-OOS Sharpe +1.15 / FF5 α +18.7%/yr at t=+6.85 (p<0.001) over 2012–2024."*

- Three reporting windows, all consistent (confirming, not competing):
  | Window | Sharpe | FF5 α/yr | t-stat |
  |---|---|---|---|
  | Full-OOS 2012–24 | **+1.15** | **+18.73%** | **+6.85** |
  | Long-OOS 2015–24 | +0.97 | +19.10% | +6.00 |
  | Test-OOS 2019–24 | +1.00 | +21.17% | +5.00 |
- **All α t-stats > 5**, all p-values < 0.001.
- **All three Sharpes > 0.95.**
- ~440 positions per rebalance, 175% monthly turnover, 10 bps/side cost.

**Figure:** `results/final_canonical_plots/equity_curve_phase24_honest.png` — cumulative growth of $1 on log scale, XGBoost vs SPY benchmark vs β-hedged pure-alpha curve.

**Speaker notes:** the headline. Be precise about the windows. Mention β-hedged decomposition shown in the figure — that's the honest visual that addresses "isn't this just market beta?"

---

### 7. Robustness battery (90s — N)

**Headline:** *"Alpha survives Carhart, DSR-25, bootstrap, cost grid, and a feature-shuffle placebo (+1.15 → −0.94 when features are scrambled)."*

| Check | Result | Interpretation |
|---|---|---|
| Carhart 6F (FF5+UMD) | α rises to +20.1%/yr at t=+7.4; UMD β = −0.43 (t=−4.6) | momentum-AVERSE — alpha is NOT repackaged momentum |
| Block bootstrap (6-mo blocks, 10k iter) | P(SR ≤ 0) = 0.0002 long-OOS | excludes zero at >99% conf |
| Deflated Sharpe (Bailey-LdP, N=25) | DSR = 0.85–0.88 | survives multiple-testing penalty for 25 configs |
| Cost grid | α significant up to ~50 bps/side; dies ~75 bps | survives conservative cost assumptions |
| **Feature-shuffle placebo** | Sharpe **+1.15 → −0.94** when feature→ticker mapping is randomly permuted within each date | **rules out engine / target / cost leakage** — the edge requires genuine feature content |
| Dense k-sweep (k=1..100) + plateau-zoom (k=10..20 with bootstrap CIs) | k=20 is statistically indistinguishable from every k ∈ [10,20] (CIs overlap 11/11) | empirically defensible hyperparameter choice |

**Figure:** 2-panel small-multiples — left: placebo bar (real +1.15 vs shuffled −0.94); right: k-sweep plateau-zoom with bootstrap error bars.

**Speaker notes (extended for the 90s slot):** placebo is the *strongest* leakage test we have — emphasise it. Examiner will ask: "How do you know your +1.15 isn't a backtest artefact?" Answer: the placebo Sharpe goes negative when features are scrambled — the strategy *needs* the genuine feature→ticker mapping to make money. Combined with the 3/3 sanity gates (Random/Oracle/Uniform on a synthetic panel), engine + target + cost machinery are clean; the alpha is feature-driven. Mention briefly: the **bootstrap** uses **6-month moving blocks** to preserve autocorrelation in monthly returns (standard for financial-time-series bootstraps — Politis-Romano stationary bootstrap would be the textbook alternative). The **DSR** penalty grows with √(2 ln N) per BLdP — so even at N=25 trials the deflation only brings us from a raw Sharpe ~+1.0 to a DSR ~0.85; the result has comfortable headroom against the conventional 0.5 cut-off.

---

### 8. Regime overlay — Andrea's workstream (brief mention) (45s — B)

> **Speaker note up front:** Andrea is not presenting today; this is a brief summary of her workstream. Full methodology and walk-forward audit script live in the appendix (`notebooks/persona/regime_crisis_detection_rate.py`); we can take Q&A questions at the high level.

**Headline:** *"HMM-based leverage overlay — works on the narrow-cap canonical (DD −25.5% → −19.9%), net-zero on the broad book (universe-dependent monthly-frequency lag)."*

- **Model:** Gaussian **Hidden Markov Model** (2 states: `calm` / `crisis`) on 6 macro-financial features (realised vol, VIX, term spread, credit spread, 3-mo S&P return). Walk-forward expanding-window training with 60-mo burn-in; `StandardScaler` refit each step (no look-ahead).
- **Overlay rule:** calm → 1.00× leverage, crisis → 0.40× (leverage-only; breadth lever tested + rejected as an ablation).
- **Result:** **strict-S&P canonical** — DD improves −25.5% → −19.9% with small Sharpe gain. **Broad canonical** — DD unchanged at −33.8% because the deepest drawdown is the Feb-Mar 2020 COVID crash; the HMM correctly flags March but the overlay de-levers off the *prior* month-end label, so we enter the crash at full leverage. **Honest universe-dependent limit, transparently reported.**

**Figure:** `results/persona_figures/overlay_failure_regime.png` — monthly returns coloured by regime label with the COVID drawdown window shaded; shows the timing-lag visually in one chart.

**Speaker notes:** keep this brief — 45 seconds, no deep dive. The point is to acknowledge the workstream exists, summarise the finding honestly, and signal that the appendix has the full methodology. If the examiner asks deeper questions during Q&A, route to "high-level we can answer, scripted detail is in the appendix."

---

### 9. The audit story (60s — B)

**Headline:** *"Mid-project, we caught our own survivorship leak: Sharpe +1.49 → −0.31 → +1.15."*

- **Phase 14 (pre-audit):** S&P 500 union, k=5, Sharpe **+1.49**. Looked great.
- **PIT audit (week 4):** discovered the engine wasn't filtering by point-in-time S&P 500 membership — it traded any ticker in the panel (pre-join history for TSLA, ENPH, GNRC, NOW etc.). 726 non-member trades in a 2012–2019 RandomModel run.
- **Phase 15 (PIT applied to S&P canonical):** Sharpe **−0.31**, α not significant on any window. The +1.49 was 100% survivorship leak.
- **Rebuild on broad survivorship-free Sharadar universe:** Phase 23 → 24-RT, final canonical Sharpe **+1.15**, α significant at t > 5.
- **Two further bugs caught in week 5:** Q-filter dropping NDAQ/IONQ (alive); INCLUDE_FEATURES subset not applied (canonical was silently 16-feature). Both fixed; numbers re-baselined.

**Figure:** `results/final_canonical_plots/phase_progression_phase24.png` — Sharpe history phase by phase.

**Speaker notes:** this is the *integrity story*. Make it clear we *caught and reported* our own bugs rather than letting them slide. Examiners weight intellectual honesty highly.

---

### 10. Honest counterweights (60s — N)

**Headline:** *"Not market-neutral; alpha concentrates in the down-cap tail; single-name fragility exists."*

- **Not market-neutral:** realised Mkt-β ≈ +1.3 (longs are higher-beta than shorts in the model's natural selection). Long leg makes +37.9%/yr alone; short leg is a near-zero-P&L hedge (−2%/yr). **~55% of the headline return is leveraged market exposure; ~45% is genuine cross-sectional skill** that survives every FF adjustment.
- **Down-cap concentration:** on strict rolling top-2,000 (large-cap end of our panel), α collapses to **+1.8%/yr at t=0.96 (n.s.)**. SMB loading collapses too (+1.26 → +0.15). **Confirms GKX 2020 §IV.D:** ML alpha lives in the small/mid-cap tail.
- **Capacity is the binding limit at deployable AUM:** 175% turnover × ~440 positions = ~770% NAV/mo throughput across small-cap names. Realistic small-cap costs (size-impact-aware) likely above 30 bps/side; the cost-grid result (significant to 50 bps) is the upper envelope, not the operating point.
- **Single-name fragility** *(suggestive, n=11)*: leave-one-out study shows the top contributor (LSCG) moves Sharpe by **+0.089**; IONQ (a 2021 high-vol SPAC) moves it by +0.04. The strategy has measurable name-fragility — institutional deployment should cap per-name realised vol.

**Figure:** `results/final_canonical_plots/ff5_decomposition_phase24.png` — annualised return broken into pure α + factor contributions.

**Speaker notes:** this slide is *defence* against the obvious questions. Pre-empt: "isn't this just beta?" → no, ~45% is genuine α. "Could you actually trade this?" → not at scale, capacity-constrained.

---

### 11. Conclusion + what we'd do differently (30s — N)

**Headline:** *"ML cross-sectional alpha is real on a survivorship-free US universe, but capacity-constrained — and we caught our own bugs along the way."*

- **Defensible claim:** under realistic survivorship controls + PIT eligibility + 10 bps/side costs, the Phase 24-RT canonical produces statistically significant cross-sectional alpha on the post-2015 OOS sample.
- **What we'd do differently:**
  - **Cap per-name realised vol** in the long book (drop high-vol tail names from the trade list) — the LSCG / IONQ fragility evidence points to this.
  - **Sub-monthly regime detection** (weekly or daily HMM on rolling z-scores) would have caught the COVID Feb-Mar 2020 crash that monthly-frequency missed.
  - **Almgren-Chriss-style cost modelling** instead of flat 10/30/50 bps/side — would give honest capacity estimates.
  - **Walk-forward Optuna retune cadence** rather than one-shot tuning on 2017–18 — captures regime-change in hyperparameter optimum.
- **What we learned:** the audit journey (+1.49 → −0.31 → +1.15) is the methodological contribution; the alpha number is the empirical contribution.

**Speaker notes:** end with the *humility + methodology* framing. Not "we built a great strategy"; rather "we built an honest pipeline that produces a defensible result, and we know its limits."

---

## Q&A prep — short list of likely questions

To be expanded into a dedicated `report/QA_PREP.md` document next. Top 5 by likelihood, with **assigned answerer** (Andrea is absent, so all answers route to N or B):

1. **"How do you know the +1.15 isn't a backtest artefact?"** → **N** (slide 7 placebo) + B (sanity gates, independent audit via `canonical_qfix_validate.py`).
2. **"Why XGBoost over the neural net?"** → **N** — Diebold-Mariano on MSE + Sharpe; NN underperforms on the cross-sectional task; small effective sample size (~250 monthly observations × ~5,000 names with non-stationarity) doesn't favour deep models without much heavier regularisation.
3. **"How did you choose k=20?"** → **N** — dense k-sweep (slide 7); plateau k=10..20 statistically indistinguishable on bootstrap CIs (Phase 27b); k=20 chosen ex-ante for round-number defensibility.
4. **"What's the cost-grid sensitivity?"** → **B** (cost grid is his ablation) + N (small-cap realistic costs are the binding limit per §6).
5. **"Why does the regime overlay not help on the broad book?"** → **B** (high-level: monthly frequency can't catch the Feb-Mar 2020 COVID crash; universe-dependent finding; full methodology in appendix). If the examiner pushes for HMM-internals depth, say "Andrea owns that detail; the appendix has her reproducible audit script `regime_crisis_detection_rate.py` if you want to see the IS-vs-OOS breakdown."

### Regime-specific Q&A — fallback policy with Andrea absent

If the examiner asks any of: *HMM hyperparameter choice, GMM vs HMM model selection, regime feature engineering, walk-forward burn-in length, in-sample vs OOS crisis-detection rate, transition-matrix interpretation, why 2 states vs 3, look-ahead in `StandardScaler`*:

- **Give the one-sentence answer from the methodology** (see slide 8 + `notebooks/personc/week3_regime_finalise.py` for reference if rehearsing) and **say**: *"Andrea owns the regime workstream; the reproducible methodology and detailed write-up are in the appendix — happy to take a follow-up after the session if you'd like deeper detail."*
- Do **not** try to bluff regime-internal details. The honesty earns more credit than a guess.

---

## Production checklist

Before the talk:

- [ ] Convert this outline into actual slides (Keynote / Google Slides / Beamer)
- [ ] All figures exported at presentation resolution (≥150 dpi)
- [ ] Per-slide speaker notes added to the actual deck
- [ ] One full team rehearsal with a stopwatch (target 10:00 ± 0:30)
- [ ] Slide 1 has correct team number once assigned
- [ ] One backup laptop with the deck + a USB stick with PDF backup
- [ ] PDF of slides exported and bundled into the optional appendix submission
- [ ] Q&A prep doc circulated to all three speakers (each speaker knows their assigned questions)
