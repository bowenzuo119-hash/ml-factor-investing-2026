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
regime model scales gross leverage down in detected crises. An initial
walk-forward backtest produced an apparent net Sharpe of **+1.50** (long-OOS) —
but a survivorship audit found the engine was trading stocks *before* they
entered the index (e.g. holding Tesla in 2012, eight years before it joined the
S&P 500). Enforcing a point-in-time investable universe **collapses the
out-of-sample Sharpe to roughly zero**: −0.27 under a strict train-and-trade PIT
filter, +0.18 under the more lenient train-on-full-cross-section / trade-PIT
setup that matches GKX. **The central result of this project is therefore
methodological**: most of the apparent alpha was look-ahead bias, and the
strategy shows no statistically meaningful out-of-sample edge once survivorship
is handled correctly. We document the full audit trail and the diagnostics that
attribute the collapse to its sources. (A final hyperparameter re-tune on the
PIT panel is pending; the prior across every honest configuration is no
material alpha.)

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
- **Headline result.** An apparent +1.5 Sharpe that **collapses to ~0 once a
  point-in-time universe is enforced** — the apparent edge was look-ahead bias
  (§5). The reusable infrastructure, not the alpha, is the deliverable.
- **What we got honestly wrong / right.** Forward-reference §6 limitations; the
  central one is that we caught and quantified our own survivorship leak.

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

> *Drafted from `report/week3_regime_summary.txt` and `regime_analysis_report.txt`; Person C to review/expand.*

**Model.** Each month from 2005 onward is classified into a market regime using
unsupervised models on six macro-financial features — 21- and 63-day realised
volatility, the VIX, the 10Y–2Y term spread, the BAA–AAA credit spread, and the
trailing 3-month S&P 500 return — all lagged one trading day to avoid
look-ahead and standardised on training data only. Gaussian Mixture Models
(K = 2, 3) and Hidden Markov Models (n = 2, 3) were compared; the **HMM with
n = 2 states** was selected for its crisis-detection rate across seven known
stress episodes (GFC, Euro crisis, 2015–16 China scare, Q4-2018, COVID, 2022
inflation). The HMM's learned transition matrix makes regimes "sticky,"
matching the empirical persistence of stress periods.

**Walk-forward.** Labels are genuinely out-of-sample: a 60-month minimum
training window (2005–2009) precedes the first prediction (Jan 2010), and the
model + scaler are refit on prior history only at each step. Over 2010–2024 the
OOS distribution is **81% calm / 19% crisis**.

**Overlay.** The regime sets gross leverage and sector breadth without changing
*which* stocks the alpha model holds:

| Regime | Gross leverage | k per sector | Long/short quantile |
|---|---|---|---|
| Calm | 1.00× | 5 | 10% / 10% |
| Crisis | 0.40× | 2 | 4% / 4% |

It is delivered as `results/regime_overlay_rules.csv` and consumed by the engine
via `regime.make_regime_fn`. Measured effect (§5 ablation): the leverage lever
improves Sharpe but the breadth-tightening lever worsens drawdown; net of both
the overlay does not help, and on the honest PIT panel there is no alpha to
protect.

*[TODO (Person C): expand model-selection detail; add the regime-shaded
S&P 500 chart (`results/regime_walkforward_chart.png`).]*

---

## 5. Integrated Results — the survivorship correction

Our first walk-forward run of the canonical configuration (XGBoost, 13 features,
3-layer sector-neutral, trained from 2002-04) produced an apparent **net Sharpe
of +1.50 long-OOS / +1.01 test**. A correctness audit then found the engine's
eligible cross-section was the *full union of every ticker that ever appeared in
the panel*, filtered only by data availability — so at a 2012 rebalance it could
(and did) trade names like Tesla, Enphase, Generac, and ServiceNow years before
they joined the S&P 500. Those names were *selected into the index because they
had already been winners*, so trading them early is pure look-ahead.

Enforcing a point-in-time investable universe (`load_sp500_membership` at every
rebalance; engine v0.4.0 / v0.5.0) removes the leak. The effect is decisive:

| Configuration | Sharpe (long-OOS) | Sharpe (test) | Ann. return | Max drawdown |
|---|---|---|---|---|
| No PIT (the biased +1.5) | +1.50 | +1.01 | +12.3% | −7.9% |
| **Full PIT** (train + trade) | **−0.27** | −0.54 | −2.3% | −36% |
| **Train-full / trade-PIT** (GKX-style, no look-ahead) | **+0.18** | −0.21 | +1.5% | −25% |

**Reading.** The survivorship leak accounted for essentially the entire apparent
edge. Even the most lenient *honest* configuration — train on the full
cross-section (legitimate: only *trading* non-members was the leak; learning
from their realised past returns is not) and trade only index members, which
matches GKX's own setup — yields a long-OOS Sharpe of just **+0.18** and a
*negative* test-window Sharpe, with deep drawdowns. A final Optuna re-tune on the
PIT panel is pending, but starting from +0.18 it is very unlikely to reach a
meaningful level.

**Decomposition.** Two sub-causes, separated with the `apply_pit_to_training`
flag: the *trading* restriction (can't hold future joiners) is the dominant
effect; the *training* restriction (learning on fewer stocks) explains the
−0.27 → +0.18 recovery between full-PIT and train-full — roughly a third of the
gap — with the residual still ~0. Reproduce: `python -m
notebooks.persona.honest_headline_check`.

**Regime overlay (ablation).** On the pre-PIT panel, decomposing Person C's
overlay shows the leverage lever *helps* (+1.50 → +1.56) but the
breadth-tightening lever (`k` 5→2 in crisis) *hurts* (max drawdown blows out to
−11.9%); the bundled overlay nets negative. On the honest PIT panel the question
is moot — there is no alpha to protect. Reproduce: `python -m
notebooks.persona.regime_ablation_check`.

---

## 6. Limitations and honest findings

- **The headline result is negative — and that is the finding.** Once
  survivorship is corrected, the strategy has no statistically meaningful
  out-of-sample alpha; the apparent +1.5 Sharpe was look-ahead. Everything below
  is secondary to this.
- **Two correctness bugs found and fixed (audit trail in DECISIONS).**
  (1) *Survivorship*: the engine traded any ticker with available data,
  including future S&P joiners — fixed with the point-in-time
  `eligible_universe_fn` (engine v0.4.0/v0.5.0). (2) *UNKNOWN-sector bucket*:
  ~440 delisted/renamed tickers collapsed into a single "UNKNOWN" pseudo-sector,
  distorting the sector-neutral construction — fixed on the feature side.
- **yfinance coverage gap.** The investable panel is ~700 tickers/month
  pre-splice (CRSP, 2002–2022) vs ~510 post-splice (yfinance, 2023–2024) — a
  ~200-name gap of delisted/renamed historicals yfinance can't reach under their
  old symbols (PIT retention ~60% vs ~95%). It also drops a few 2023–24 index
  members that delisted (SIVB, FRC) from the PIT panel. We chose not to pay for a
  Sharadar SEP upgrade: better data does not recover an alpha that isn't there
  (DECISIONS 2026-05-23).
- **Regime model.** HMM n=2 was selected by crisis-detection rate on *known*
  stress periods (some hindsight in model selection); the overlay covers
  2010–2024 (2005–09 was walk-forward burn-in). Its leverage lever helps and its
  breadth lever hurts (§5), but the question is moot on the honest panel.
- **One 22-year sample, US large-cap only.** A single historical path; results
  need not generalise to other universes or periods.

---

## 7. Conclusion

A disciplined ML factor pipeline on the S&P 500 shows **no statistically
meaningful out-of-sample alpha once survivorship and look-ahead are handled
correctly.** The apparent +1.5 Sharpe of our first canonical was almost entirely
the artifact of trading future index entrants; under a point-in-time universe
the honest out-of-sample Sharpe is roughly zero (−0.27 strict-PIT, +0.18
train-full / trade-PIT).

We regard this as the project's most valuable result. It is a concrete,
reproducible demonstration of how readily a naive ML backtest manufactures
spurious alpha, and of the audit discipline — point-in-time universe
enforcement, a Random/Oracle/Uniform sanity gate, and decomposed diagnostics —
required to catch it; precisely the failure mode López de Prado (2018)
documents. The *infrastructure* is sound and reusable (a point-in-time data
pipeline, a versioned walk-forward engine with the survivorship fix, and a
sanity-gated backtest); the *alpha* is not there. A genuine edge, if one exists,
would more plausibly come from a broader investable universe (GKX use the full
CRSP cross-section, not 500 names), richer features, or cost-aware portfolio
construction — the natural continuation of this work.

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
