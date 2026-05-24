# Machine-Learning Factor Investing on the S&P 500
### A long–short cross-sectional strategy with a regime-conditioned leverage overlay

**Team:** Bowen Zuo (data & infrastructure) · Nicolas Couto Mota (alpha model) · Andrea Fontana (regime overlay)
**Course:** *(course / term)* — 5-week project
**Code:** https://github.com/bowenzuo119-hash/ml-factor-investing-2026
**Status:** DRAFT skeleton — sections in `report/*_SECTION.md` are the authoritative detail; this file is the assembled narrative. `[TODO]` marks gaps to fill before submission.

---

## Abstract

We build a monthly-rebalanced, dollar-neutral long–short equity strategy on the
S&P 500, following the machine-learning asset-pricing approach of Gu, Kelly &
Xiu (2020). A gradient-boosted model (XGBoost) forecasts the cross-section of
next-month returns from 13 firm features spanning price trend, liquidity,
volatility, and Fama-French value/quality; forecasts are turned into a
sector-neutral top-/bottom-quantile portfolio; and a Gaussian-mixture / HMM
regime model scales gross leverage down in detected crises. Evaluated under a
strict walk-forward backtest with transaction costs, the canonical XGBoost
strategy earns a net Sharpe of **1.49 (long-OOS) / 1.01 (2019–2024 test)** with
a maximum drawdown of **−7.9%**, and a long-OOS Deflated Sharpe Ratio of
**0.992** (clearing the 0.95 significance threshold after correcting for the
trials run during development). We are candid about the residual concerns: the
6-year test-window DSR (0.887) is just below significance, and Fama-French
alpha is not statistically significant — a meaningful fraction of the edge is
factor exposure rather than uncorrelated skill.

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

A monthly panel of S&P 500 constituents (2002–2024, ~929 unique tickers) with
**13 features**: momentum (12-1), short-term reversal, size, dollar volume,
return & idiosyncratic volatility, B/M, E/P, plus quality/investment factors
(ROE, ROA, D/E, asset growth, accruals). Three models share a `fit`/`predict`
interface — Lasso (linear baseline), **XGBoost (canonical)**, and a small NN.
Forecasts become a sector-neutral top-/bottom-quantile dollar-neutral book
(k = 5 names per sector per leg). XGBoost wins on Diebold-Mariano and on net
Sharpe; SHAP attributes most of the signal to momentum and the value factors.

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
and 10 bps per-side transaction costs. See `results/23g_canonical_qfiltered_orig_tune/`
and `results/final_canonical_plots/`.

### Final canonical (Phase 23g)

| Window | Sharpe | Ann return | Max DD | **FF5 alpha** | **t-stat** | **p-value** | Mkt-β |
|---|---|---|---|---|---|---|---|
| **Full-OOS 2012–2024** | **+1.05** | +34.1% | −34.3% | **+17.74%/yr** | **+5.52** | **<0.001 ✓✓** | +1.42 |
| Long-OOS 2015–2024 | +0.95 | +33.7% | −34.3% | +18.92%/yr | +5.31 | <0.001 ✓✓ | +1.46 |
| Test 2019–2024 | +1.02 | +43.2% | −34.3% | +22.64%/yr | +5.16 | <0.001 ✓✓ | +1.54 |

**Honest decomposition of the +34%/yr long-OOS realised return:**
- ~+19%/yr from market beta exposure (β = 1.42 × ≈13.5% Mkt-RF premium)
- ~+5%/yr from small-cap tilt (SMB)
- **~+18%/yr pure cross-sectional alpha** (FF5-adjusted, statistically significant)

So ~55-60% of the headline return comes from market exposure; the remaining
~40-45% is a genuine ML factor edge that survives Fama-French adjustment at
t > 5 across every reporting window. This is the first time in the project
we have statistically significant FF5 alpha.

### Comparison to the broader project narrative

| Phase | Universe | Construction | Sharpe (long-OOS) | FF5 α | Status |
|---|---|---|---|---|---|
| Phase 15 (pre-audit) | S&P 500 union (leaky) | k=5 dollar-neutral | +1.49 | n/a | **INVALID — survivorship leak** |
| Phase 22 (honest S&P) | Strict-PIT S&P 500 | k=5 dollar-neutral | +0.31 | n.s. (t=-0.4) | Market-neutral but no alpha |
| **Phase 23g (canonical)** | **Broad US ~2000 names** | **k=20 + Q-filter** | **+0.95** | **+18.9%/yr (t=5.3) ✓✓** | **HONEST FINAL** |

### Key plots

- [Equity curve](../results/final_canonical_plots/equity_curve_phase23g.png) — cumulative net return per model (XGBoost, NN, Lasso)
- [Drawdown](../results/final_canonical_plots/drawdown_phase23g.png) — drawdown trajectory
- [FF5 decomposition](../results/final_canonical_plots/ff5_decomposition_phase23g.png) — annualised return broken into pure alpha + factor contributions
- [Phase progression](../results/final_canonical_plots/phase_progression_phase23g.png) — Sharpe history from leaky pre-audit through honest final

### Regime overlay sensitivity

*[TODO: re-run regime overlay ablation on Phase 23g; previous regime work
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
