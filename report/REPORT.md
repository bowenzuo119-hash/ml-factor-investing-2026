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

> **Full detail: [`report/DATA_AND_ENGINE_SECTION.md`](DATA_AND_ENGINE_SECTION.md).**

Six point-in-time sources feed the panel: **CRSP MSF** monthly total returns
(1925–2022), spliced to **yfinance** for 2023–2025 (validated at 0.999999
median return correlation on the overlap window); **Sharadar SF1** fundamentals
for the value/quality factors (B/M from the as-reported quarterly dimension,
E/P from trailing-twelve-month — a distinction caught by validation, not
assumption); **yfinance daily** dollar volume for the liquidity factor;
**FRED** macro series for the regime model; and **fja05680** point-in-time
S&P 500 membership so the universe carries no survivorship bias. The
**walk-forward backtest engine** refits on a sliding 120-month window at each
test-block boundary, charges 10 bps per side on L1 turnover, supports a
three-layer sector-neutral construction, and is gated by a Random/Oracle/Uniform
sanity suite (Project Framework §4.6). Methodology figures:
`results/persona_figures/`.

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

**Model and feature set.** Each month is classified into a market regime using
an unsupervised model fitted on six macro-financial features: 21-day and
63-day realised S&P 500 volatility, the VIX level, the 10Y–2Y Treasury term
spread, the BAA–AAA corporate credit spread, and the trailing 3-month S&P 500
return. All inputs are lagged by one trading day to eliminate look-ahead, and
the scaler is refit on the training window only at each walk-forward step.

We compared Gaussian Mixture Models (K = 2, 3) and Hidden Markov Models
(n = 2, 3). The canonical overlay keeps the **2-state HMM** selected by the
walk-forward crisis-detection criterion rather than imposing extra granularity
by hand. This matters for two reasons. First, the 2-state model was the best
empirical performer out-of-sample among the candidates we tested. Second, a
binary overlay yields the cleanest economic interpretation for the report:
**full risk in calm periods, materially reduced risk in crises**.

**Walk-forward evaluation.** Regime labels are genuinely out-of-sample. A
60-month minimum training window (2005–2009) is used before the first
prediction in January 2010, and the model plus scaler are refit using prior
history only. Over the 2010–2024 OOS window, the selected 2-state HMM assigns
approximately **81% of months to calm** and **19% to crisis**. Its honest
walk-forward crisis-detection rate across the predefined stress episodes is
**51.1%**, which is substantially lower than the in-sample fit and is the
number that should be used when interpreting the overlay's realism. We view
that gap as expected: the regime model is useful, but not clairvoyant.

**Overlay rule.** The regime overlay does not change *which* stocks are chosen
by the alpha model; it changes portfolio aggressiveness and breadth:

| Regime | Gross leverage | k per sector | Long/short quantile |
|---|---|---|---|
| Calm | 1.00× | 5 | 10% / 10% |
| Crisis | 0.40× | 2 | 4% / 4% |

Operationally, these parameters are written to
`results/regime_overlay_rules.csv` and consumed by the engine through
`src.regime.make_regime_fn`. When the engine receives both `k_per_sector` and a
sector map, it applies top-k / bottom-k selection within sector, so the
overlay tightens both gross exposure and cross-sectional breadth in crisis
states.

**Interpretation.** The 2-regime specification is the canonical one in the
report because it is the walk-forward-selected model and because the economic
story is cleaner than a three-state variant. A 3-regime extension remains a
reasonable sensitivity check, but it is not needed to state the main result.

![Regime-shaded S&P 500 chart](../results/regime_walkforward_chart.png)

---

## 5. Integrated Results

The canonical configuration is **XGBoost + 13 features + 3-layer sector-neutral
(k = 5) + regime overlay**, trained from 2002-04 (`results/15_canonical_2002/`),
walk-forward with 10 bps/side costs.

| Window | Net Sharpe | Ann. return | Max drawdown | DSR |
|---|---|---|---|---|
| Long-OOS (2013–2024) | **+1.49** | +12.3% | −7.9% | **0.992** |
| Test-only (2019–2024) | **+1.01** | +9.5% | −7.9% | 0.887 |

Robustness (§8 of the Alpha Model section): 2015–2024 bootstrap Sharpe 95% CI
**[+0.82, +1.83]**, P(Sharpe ≤ 0) = 0.000; long-OOS **DSR = 0.992** after
correcting for the trials run during development. Diebold-Mariano selects
XGBoost over Lasso.

*[TODO: regime overlay ablation — report with-overlay vs no-overlay net Sharpe
and max drawdown side by side, to show the overlay earns its keep (the cleanest
single table to justify Person C's contribution). Compare `results/14_*` or a
no-regime run vs `results/15_canonical_2002/`.]*

*[TODO: insert the headline equity curve and drawdown plot —
`results/15_canonical_2002/` or `results/presentation_plots/`.]*

---

## 6. Limitations and honest findings

- **Test-window significance is borderline.** Long-OOS DSR (0.992) clears 0.95,
  but the stricter 6-year 2019–2024 test window sits at **0.887** — just below.
  The long-OOS result is the significant one; the short window is suggestive.
- **Alpha is largely factor exposure.** Fama-French alpha is not significant
  (t ≈ 0.8); the strategy loads negatively on HML (short-value tilt) and
  positively on the market. The edge is real but is partly compensated factor
  risk, not pure uncorrelated skill.
- **Free-data coverage gap.** ~10–16% of historical tickers (delisted/renamed
  before ~2022) are unavailable in yfinance under their old symbol; their
  history ends at the CRSP cutoff. The point-in-time universe limits the bias
  but does not erase it.
- **One 22-year sample.** The DSR adjustment accounts for the trials we ran, but
  it remains a single historical path.
- *[TODO (Person C): regime model's walk-forward crisis-detection rate is lower
  out-of-sample than in-sample — state the honest OOS number.]*

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
