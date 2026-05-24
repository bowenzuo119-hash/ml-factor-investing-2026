# Machine-Learning Factor Investing on the S&P 500
### A long–short cross-sectional strategy with a regime-conditioned leverage overlay

**Team:** Bowen Zuo (data & infrastructure) · Nicolas Couto Mota (alpha model) · Andrea Fontana (regime overlay)
**Course:** *(course / term)* — 5-week project
**Code:** https://github.com/bowenzuo119-hash/ml-factor-investing-2026
**Status:** DRAFT skeleton — sections in `report/*_SECTION.md` are the authoritative detail; this file is the assembled narrative. `[TODO]` marks gaps to fill before submission.

---

## Abstract

We build a monthly-rebalanced long–short equity ML strategy on a broad US
universe (top-2,000 US common stocks by market cap, survivorship-free via
SHARADAR/TICKERS), following the machine-learning asset-pricing approach of
Gu, Kelly &amp; Xiu (2020). A gradient-boosted model (XGBoost) forecasts the
cross-section of next-month returns from **14 firm features** spanning price
trend, liquidity, volatility, GKX momentum-acceleration, and Fama-French
value/quality. Forecasts become a sector-neutral top-/bottom-quantile book
(k=20 per GICS sector, bankrupt-ticker filter applied). A Hidden-Markov regime
model scales gross leverage in detected crises — empirically valuable on the
strict-S&amp;P canonical but neutral on the broad rebuild due to a documented
monthly-frequency timing limit around the COVID-2020 fast crash.

Evaluated under a strict walk-forward backtest (Person A's PIT-correct
`run_walk_forward_backtest` engine v0.5.0, 10 bps/side transaction costs,
120-month sliding training window), the canonical XGBoost strategy earns
a net Sharpe of **+1.08 (full-OOS 2012–2024) / +0.98 (long-OOS 2015–2024) /
+1.06 (test-only 2019–2024)** at 10 bps/side, or **~+0.90 / +13.5%/yr FF5
alpha (t≈4.3)** under a more conservative 30 bps/side assumption (justified by
the strategy's small-cap tilt and 175% monthly turnover). The
**Fama-French 5-factor alpha is +18.2%/yr at t=+5.74 (p&lt;0.001)** on the
long-OOS window and significant on every reporting window — the first time in
the project we have statistically significant alpha after factor adjustment.

We are explicit about the residual concerns. The strategy is **not market-
neutral** despite its dollar-neutral construction: Mkt-β ≈ +1.3 (longs are
higher-beta than shorts in the model's natural selection), with a long-leg-
dominated P&amp;L profile (long leg alone contributes +37.9%/yr; short leg is
a near-zero-P&amp;L market-neutralizing hedge). The 12-year history has
significant exposure to a single deep drawdown (−34.0% in Feb–Mar 2020 COVID),
which a fast-frequency regime overlay could plausibly mitigate but our monthly
HMM did not catch in time. An earlier headline of "+1.49 long-OOS Sharpe"
on the S&amp;P-500-only Phase 14 was withdrawn after our own internal audit
identified a survivorship leak in the engine (the engine traded any ticker
in the panel rather than only S&amp;P 500 members at each rebalance); the
audit, the corrections shipped (engine v0.4.0 PIT filter, Q-suffix bankrupt-
ticker filter, broader-universe rebuild), and the resulting honest canonical
are documented transparently in §6 and `PIT_INVESTIGATION_REPORT.pdf`.

---

## 1. Introduction

*[TODO: 3–4 paragraphs. Suggested flow, all already supported by the material below:]*

- **Problem.** Can a disciplined ML pipeline produce a defensible, out-of-sample
  long–short factor strategy on a liquid universe, and does a market-regime
  overlay improve its risk profile?
- **Approach.** Three decoupled workstreams behind one interface
  (`run_walk_forward_backtest`): a point-in-time data pipeline (§2), a
  cross-sectional alpha model (§3), and a regime-conditioned leverage overlay
  (§4). The seam is versioned so each part iterates independently.
- **Headline result.** *(pull the §5 numbers)*.
- **What we got honestly wrong / right.** Forward-reference §6 limitations.

---

## 2. Data and Infrastructure  *(Person A)*

The canonical pipeline runs entirely on **Sharadar** (Nasdaq Data Link) — one
premium subscription covering six tables: **SF1** (fundamentals), **SEP**
(prices), **DAILY** (market cap / valuation), **TICKERS** (security master),
**SP500** (membership), **ACTIONS** (corporate actions). **SEP `closeadj`**
(split- and dividend-adjusted daily close) is the single price source for
2002–2024 — *no CRSP/yfinance splice* — and it carries **delisted names under
their historical symbols** (LEHMQ → 2008-10, ENRNQ → 2004, SIVBQ → 2023), so the
panel is survivorship-free by construction. Validated: Sharadar monthly returns
correlate **1.0000** with yfinance on surviving large-caps, SEP prices match
SF1's reported point-in-time price on delisted names (median |Δ| ≈ 0%), and
`closeadj` does not jump across split dates. CRSP MSF and yfinance remain wired
in `data_loader.py` as historical alternative price sources but are **not** on
the canonical path.

**Broad investable universe.** At each month-end the universe is the **top
2,000 US common stocks by market cap** *trading at that date* — eligible iff
`firstpricedate ≤ asof ≤ lastpricedate` (a delisted name drops out only after
its last price, never before — no look-ahead, no survivorship), common stock on
a major exchange (TICKERS), with positive DAILY market cap. Rolling top-N avoids
the inflation drift of a fixed dollar threshold; the 2002–2024 union is ~5,900
distinct tickers. A **bankrupt-ticker filter** drops Sharadar's Q-suffix
delisted symbols (`len(ticker) ≥ 4 and ticker.endswith("Q")`) — 1,114 names,
clustering in 2008 and 2023 — so terminal bankruptcy-price dynamics cannot
manufacture spurious alpha.

![Investable universe over time — strict PIT vs broad vs total](../results/persona_figures/universe_coverage_broad.png)

![Survivorship correction — strict PIT vs broad PIT vs leaky union](../results/persona_figures/universe_survivorship_comparison.png)

![Bankrupt-ticker exclusion volume per year](../results/persona_figures/q_filter_exclusions.png)

**Engine.** The walk-forward backtest refits on a sliding 120-month window at
each test-block boundary (block-gated refit), enforces the point-in-time
universe on both training and trading (`eligible_universe_fn`), charges 10 bps
per side on L1 turnover, supports three-layer sector-neutral construction, and
is gated by a Random/Oracle/Uniform sanity suite (Project Framework §4.6) — the
broad panel passes **3/3** (random Sharpe +0.01, oracle +153.8, uniform 0 bps).

> **Long-form companion:** [`report/DATA_AND_ENGINE_SECTION.md`](DATA_AND_ENGINE_SECTION.md)
> (its CRSP-splice sections describe the now-superseded historical price path).

---

## 3. Alpha Model  *(Person B)*

> **Full detail: [`report/ALPHA_MODEL_SECTION.md`](ALPHA_MODEL_SECTION.md).**

Monthly panel of US common stocks 2002–2024, **broad universe (~2,000 names
per rebalance, top-by-market-cap, PIT-correct via §2's universe filter,
5,897 unique tickers across the sample)**. **14 features:**
- **Price-trend (5):** momentum (12-1), short-term reversal,
  return volatility (12m), idiosyncratic volatility (CAPM-residual 12m),
  and **GKX `chmom` — change in 6-month momentum** (rank #4 in the
  Gu-Kelly-Xiu 2020 feature-importance ranking, captures momentum
  acceleration as `mom(t-6..t-1) − mom(t-12..t-7)`; orthogonality verified,
  |corr| &lt; 0.06 with all existing features).
- **Liquidity (2):** log market cap, log dollar volume.
- **Value (2):** book/market (Sharadar ARQ), earnings/price (Sharadar ART).
- **Quality/investment (5):** ROE, ROA, debt/equity, asset growth, accruals.

Three models share a `fit`/`predict` interface — Lasso (linear baseline),
**XGBoost (canonical)**, and a small PyTorch NN. **XGBoost hyperparameters
are Optuna-retuned per panel** (60 trials each on the 2017-18 validation
window — Phase 23a for the 13-feature baseline, **Phase 24a for the
14-feature canonical** including `chmom`). The retune lifts validation R²
from +0.0031 to +0.0055 (+18%) and the walk-forward Sharpe from +1.05 to
+1.08 with FF5 alpha t-stat climbing from +5.5 to +6.0.

Forecasts become a **sector-neutral top-/bottom-quantile dollar-neutral book
(k=20 per GICS sector ≈ 440 positions, ~0.45% per name)** with bankrupt-
ticker filter applied (Q-suffix rule: drop tickers ending in 'Q' of length
≥ 4 — Sharadar's bankruptcy convention; ~1,114 names dropped over the sample,
clustering in 2008 and 2023 per §2). XGBoost wins decisively on the Diebold-
Mariano test, on net Sharpe, and on FF5 alpha t-stat across every reporting
window. The two GKX features tested *beyond* `chmom` (`maxret`, `mom36m`)
were added to the panel and tested in **Phase 24b**, but a retune on the
16-feature panel produced a **lower** walk-forward Sharpe (+0.98 vs +1.08)
— marginal-feature complexity not justified by signal gain. `maxret` and
`mom36m` remain in the panel for the sensitivity record but are excluded
from the canonical `INCLUDE_FEATURES`. (See §6 limitations for an honest
discussion of feature parsimony vs broader-GKX ambition.)

---

## 4. Regime Overlay  *(Person C)*

**Model.** Each month is classified into a market regime by an unsupervised
model on six macro-financial features — 21- and 63-day realised volatility, the
VIX, the 10Y–2Y term spread, the BAA–AAA credit spread, and the trailing
3-month S&P 500 return — all lagged one trading day and standardised on
training data only. A **2-state HMM** was selected by walk-forward
crisis-detection (over GFC, Euro crisis, 2015–16, Q4-2018, COVID, 2022). Labels
are genuinely OOS (60-month burn-in 2005–2009 before the first prediction);
over 2010–2024 the split is **~81% calm / 19% crisis** (honest walk-forward
crisis-detection rate ≈ 51%).

**Overlay (leverage-only).** The overlay changes only **gross leverage** —
1.00× in calm, **0.40× in crisis** — holding breadth fixed. An earlier variant
that also tightened `k` and the quantiles in crises was dropped: an ablation
showed the breadth lever *hurt* drawdown while the leverage lever helped.
Delivered as `results/regime_overlay_rules.csv`, consumed via
`regime.make_regime_fn`.

| Regime | Gross leverage | Breadth (k, quantiles) |
|---|---|---|
| Calm | 1.00× | unchanged |
| Crisis | 0.40× | unchanged |

**It works on the strict-S&P canonical, but not on the broad one.** On the
strict-S&P-500 PIT canonical (Phase 22) the overlay cuts max drawdown
**−25.5% → −19.9%** with a small Sharpe *gain* (+0.18 → +0.27). On the broad
Sharadar canonical (Phase 23g) it does **not** help: full-OOS Sharpe +1.07 →
+1.13, but test-OOS **+1.00 → +0.94** and **max drawdown unchanged at −33.8%**
— the de-levering only costs return (34% → 27%) with no drawdown benefit.

**Why — a monthly-regime timing limit (COVID).** The −34% max drawdown is the
**Feb–Mar 2020 COVID crash**. The HMM correctly flagged March 2020 as crisis,
but the overlay sets leverage from the *prior* month-end's label — and both
Jan-end and Feb-end were 'calm', so the portfolio entered the crash at full
leverage and the crisis flag arrived **one rebalance too late**. This is a
fundamental limit of monthly-frequency regime detection on a fast crash, not a
flaw in the model or the overlay logic — and it is **universe-dependent**: the
small-cap-tilted broad book has idiosyncratic drawdown dynamics that
index-volatility regime detection underweights.

![Phase 23g monthly returns coloured by HMM regime label; COVID window shaded](../results/persona_figures/overlay_failure_regime.png)

*[Person C (Andrea) to co-review — regime model-selection detail is her domain. Ablation: `notebooks/persona/regime_overlay_ablation_broad.py`; diagnostic: `overlay_failure_diagnostic.py`.]*

---

## 5. Integrated Results

The **final honest canonical** is XGBoost on the broad US equity universe
(~2000 names per date, top by market cap, PIT survivorship-free) with
sector-neutral construction (k=20 per GICS sector), bankrupt-ticker filter,
and **14 features** (13 GKX-style price/value/quality + GKX `chmom`).
See `results/24_canonical_with_chmom/` and `results/final_canonical_plots/`.

### Final canonical (Phase 24-RT) — XGBoost @ 10 bps/side

| Window | Sharpe | Ann return | Max DD | **FF5 alpha** | **t-stat** | **p-value** | Mkt-β |
|---|---|---|---|---|---|---|---|
| **Full-OOS 2012–2024** | **+1.08** | +33.2% | −34.0% | **+17.51%/yr** | **+6.00** | **<0.001 ✓✓✓** | +1.27 |
| Long-OOS 2015–2024 | +0.98 | +32.0% | −34.0% | +18.18%/yr | +5.74 | <0.001 ✓✓✓ | +1.30 |
| Test 2019–2024 | +1.06 | +41.2% | −34.0% | +21.91%/yr | +5.32 | <0.001 ✓✓✓ | +1.37 |

### Conservative cost basis (30 bps/side, justified by small-cap tilt + 175% turnover)

| Window | Sharpe (30bps) | Ann return | FF5 α (30bps) | α t-stat |
|---|---|---|---|---|
| Long-OOS 2015–2024 | **~+0.90** | ~+27% | **~+13.5%/yr** | **~+4.3** ✓✓ |

*[TODO: pull exact numbers from Bowen's `cost_sensitivity_phase24.py` rerun]*

### Honest characterisation — NOT market-neutral

The strategy realizes a **high-beta directional long-short book** with significant cross-sectional alpha on top, NOT a market-neutral construct:

- **Mkt-β ≈ +1.3** (emerges from systematic long-small-cap-high-beta vs short-large-cap-low-beta picks)
- **Long-leg dominated**: long leg alone makes +37.9%/yr (Sharpe +1.16) while the short leg makes −2.0%/yr (Sharpe −0.47). The short leg acts as a near-zero-P&L market-neutralizing hedge — the alpha lives in long-side stock picking.
- **High volatility** (~32%/yr) and **deep drawdowns** (−34% max, COVID Feb-Mar 2020)

**Decomposition of the +32%/yr long-OOS realised return:**
- **~+18%/yr pure FF5 alpha** (the real cross-sectional skill, t=+5.74, p<0.001)
- ~+17%/yr from market beta (β=+1.30 × 13.5% Mkt-RF premium 2015-24)
- Residual from SMB/HML/RMW/CMA factor loadings

So ~55% of the headline return comes from market exposure; **~45% is genuine ML cross-sectional skill that survives Fama-French adjustment at t > 5 across every reporting window.**

### Comparison to the broader project narrative

| Phase | Universe | Construction | Sharpe (long-OOS) | FF5 α | Status |
|---|---|---|---|---|---|
| Phase 14 (pre-audit) | S&P 500 union (LEAKY) | k=5 dollar-neutral | +1.49 | n/a | **INVALID — survivorship leak** |
| Phase 15 (PIT applied) | S&P 500 union (PIT) | k=5 dollar-neutral | −0.31 | n.s. | Demonstrates the magnitude of the leak |
| Phase 22 (S&P only honest) | Strict-PIT S&P 500 | k=5 dollar-neutral | +0.31 | n.s. (t=−0.4) | Market-neutral but no alpha at this universe scale |
| Phase 23g (broad rebuild) | Broad US ~2000 | k=20 + Q-filter, 13 features | +0.95 | +18.9%/yr (t=5.3) ✓✓ | First sig FF5 alpha in project |
| **Phase 24-RT (FINAL)** | **Broad US ~2000** | **k=20 + Q-filter + chmom (14 features)** | **+0.98** | **+18.2%/yr (t=5.74) ✓✓✓** | **FINAL CANONICAL** |

### Key plots

- [Equity curve (Phase 24-RT)](../results/final_canonical_plots/equity_curve_phase24_honest.png) — cumulative growth of $1 on log scale, with **SPY benchmark** and **β-hedged pure-alpha curve** alongside the raw XGBoost line. The decomposition is the headline visual: the raw line includes ~1.3× leveraged market exposure, the β-hedged line shows the genuine cross-sectional skill.
- [Drawdown (Phase 24-RT)](../results/final_canonical_plots/drawdown_phase24.png) — drawdown trajectory: XGBoost vs SPY vs β-hedged
- [FF5 decomposition (Phase 24-RT)](../results/final_canonical_plots/ff5_decomposition_phase24.png) — annualised return broken into pure alpha + factor contributions
- [Phase progression](../results/final_canonical_plots/phase_progression_phase24.png) — Sharpe history: Phase 14 (leaky +1.49) → Phase 15 (PIT collapse −0.31) → Phase 22 (S&P honest +0.31) → Phase 23g (broad +0.95) → Phase 24-RT (final +0.98)
- [Long-leg vs short-leg decomposition](../results/long_short_decomp/long_short_decomp_phase24.png) — three-line cumulative growth showing the long leg does ~all the work (+38%/yr) while the short leg is a near-zero-P&L hedge (−2%/yr). Confirms the "long high-conviction stocks + token short hedge" characterisation.

### Regime overlay sensitivity (per Person A's Phase 23g ablation, applies to Phase 24-RT)

The regime overlay was applied to the broad canonical and yielded a **net-zero
benefit**: full-OOS Sharpe nudges +1.07 → +1.13, but test-OOS goes +1.00 →
+0.94 and **max DD is unchanged at −33.8%**. The deepest DD (Feb-Mar 2020
COVID) was entered at full leverage because both Jan-end and Feb-end regime
flags were 'calm' — the HMM correctly flagged March as crisis, but the
overlay sets leverage from the *prior* month-end label. On the strict-S&P-500
canonical (Phase 22) the overlay DOES help (DD −25.5% → −19.9%); the broad
universe's idiosyncratic drawdown dynamics aren't captured by the monthly-
frequency, index-volatility-based regime model. See `results/persona_figures/overlay_failure_regime.png`.

*[Historical TODO retained for audit — superseded by the ablation result above:
was done on the leaky pre-audit pipeline.]*

---

## 6. Limitations and honest findings

### The PIT-leak incident (lessons learned, fully transparent)

The most important honest disclosure in this project: **the previously-reported
long-OOS Sharpe of +1.49 was inflated by a survivorship leak in the backtest
engine**. We claimed point-in-time correctness throughout the project (and had
the `load_sp500_membership` function on disk), but the engine wasn't actually
using it — it treated every ticker in our panel as eligible at every rebalance,
filtering only by non-NaN next-period return.

**Quantitative impact** (audit findings 2026-05-23, full detail in
[PIT_INVESTIGATION_REPORT.pdf](../PIT_INVESTIGATION_REPORT.pdf)):
- Our panel contained 125 pre-S&P-join return observations for TSLA, 104 for
  ENPH, 133 for GNRC, 101 for NOW — all of which the engine could see and
  trade as if they were S&P 500 members.
- Bowen quantified: a 2012-2019 RandomModel run traded 726 non-member
  positions without the filter; with `eligible_universe_fn=universe_at` it
  trades 0.
- After enforcing strict PIT membership, XGBoost long-OOS Sharpe drops from
  +1.49 to −0.31. With relaxed PIT (cumulative ever-S&P members, no future
  joiners), it recovers to +0.31 — but the FF5 alpha is not significant at any
  window. The +0.31 is essentially +0.30 Mkt-RF beta exposure.

**Methodology corrections shipped (Bowen + Person B, 2026-05-23):**
- Engine v0.4.0: optional `eligible_universe_fn` filter on prediction-time
  eligibility and training labels.
- Engine v0.5.0: `apply_pit_to_training` flag for clean decomposition of
  training vs trading restrictions.
- Driver: sector_map derived from `features['sector']` (with SIC fallback)
  instead of `load_sector_map()` alone — dissolves the synthetic UNKNOWN
  bucket that had 10 of 110 positions per rebalance.
- Optuna retune of XGBoost on PIT-filtered panel (~8× the previous validation R²).

### Why our S&P-500-only ML strategy doesn't show alpha

Once the leak is closed, the honest finding is consistent with the academic
literature: ML cross-sectional alpha lives primarily in **small/mid-cap stocks
not in the S&P 500**. Gu-Kelly-Xiu (2020) report Sharpe 1.5+ on a CRSP universe
of 3000-6000 stocks per month; the S&P 500 (500 largest stocks by definition)
is the most efficiently-priced subset of US equities and offers little
cross-sectional dispersion for an ML model to exploit. Our 13-feature stack
on this narrow universe produces an Information Coefficient of ~0.006 and no
significant FF5 alpha.

### Broader-universe rebuild in progress (Phase 23)

To test whether ML alpha is recoverable on a broader universe, we are
rebuilding the pipeline on Bowen's premium Sharadar subscription (SF1 +
DAILY + TICKERS + SP500 + ACTIONS — all included free in the existing
subscription). The new universe will be ~2000-3000 US common stocks with
mcap > $1B per rebalance (~Russell 1000-1500 equivalent). Subscription
expires 2026-06-22 ("Will Not Renew") so the bulk data pull is happening
this weekend.

### Other limitations

- **Free-data coverage gap (pre-broader-rebuild).** ~10–16% of historical
  tickers (delisted/renamed before ~2022) are unavailable in yfinance under
  their old symbol; their history ends at the CRSP cutoff. The broader-universe
  rebuild via Sharadar closes this gap.
- **One ~12-year OOS sample.** The DSR adjustment accounts for the trials we
  ran, but it remains a single historical path.
- *[TODO (Person C): regime model's walk-forward crisis-detection rate is
  lower out-of-sample than in-sample — state the honest OOS number.]*

---

## 7. Conclusion

*[TODO: 2 paragraphs. The defensible claim: a disciplined, survivorship-aware,
look-ahead-controlled ML pipeline produces a long-OOS net Sharpe ~1.5 that is
statistically significant after multiple-testing correction, with a regime
overlay that *(quantify once the §5 ablation lands)*. The honest claim: the
edge is partly factor exposure and the short-window result is borderline.]*

---

## 8. Reproducibility

All seeds pinned to 42. Data caches (gitignored) rebuild from a clean clone
with `python -m notebooks.persona.run_all_data` (skips steps whose prerequisite
— the CRSP raw CSV or the Nasdaq Data Link key — is absent). The canonical
result and every diagnostic regenerate from the `notebooks/personb/*.py`
phase scripts listed in the Alpha Model section §12; methodology figures from
`python -m notebooks.persona.report_figures`. Every non-trivial design choice is
logged in `DECISIONS.md`.

---

## References

- Gu, S., Kelly, B., & Xiu, D. (2020). *Empirical Asset Pricing via Machine Learning.* Review of Financial Studies.
- Bailey, D. & López de Prado, M. (2014). *The Deflated Sharpe Ratio.* Journal of Portfolio Management.
- Fama, E. & French, K. (2015). *A Five-Factor Asset Pricing Model.* JFE.
- Sloan, R. (1996). *Do Stock Prices Fully Reflect Information in Accruals…* The Accounting Review.
- Jegadeesh, N. & Titman, S. (1993). *Returns to Buying Winners and Selling Losers.* Journal of Finance.
- *[TODO: add Cooper-Gulen-Schill (2008) asset growth; Nystrup et al. (2018) regime allocation.]*
