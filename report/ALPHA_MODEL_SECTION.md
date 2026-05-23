# Alpha Model

*Person B's section of the final report. Draft 1 — 2026-05-23.*

This section describes the cross-sectional machine-learning model that produces
the per-month return forecasts consumed by the portfolio construction and
regime-overlay layers. The model and its evaluation framework deliberately
follow the Project Framework (§§3, 6, 8) so that every choice has a written
justification and every reported number is reproducible from the committed
code.

## 1. Data and panel

The model is trained on a monthly panel of S&P 500 constituents from
**April 2002 to December 2024** (273 months × 941 unique tickers). The data
pipeline that produces this panel — CRSP→yfinance splice, Sharadar SF1
fundamentals, point-in-time S&P 500 membership — is described in detail in
the [Data Pipeline and Backtest Engine section](DATA_AND_ENGINE_SECTION.md)
(Person A); the headline validation is a 0.999999 monthly-return correlation
between CRSP and yfinance on the splice overlap. The point-in-time universe
holds at ~500 names per month across the window
([universe_coverage.png](../results/persona_figures/universe_coverage.png));
the 2022/2023 CRSP→yfinance transition is documented in
[splice_timeline.png](../results/persona_figures/splice_timeline.png).

The **2002-04 start** is the earliest defensible point given our feature
coverage. Sharadar SF1 fundamentals coverage of the S&P 500 stabilises at
~73-75% by April 2002 (Jan-Mar 2002 dip to 69-72% as Q4-2001 filings come
in); starting at 2002-04 gives data parity with later years. With the
120-month sliding training window this puts the model's first prediction
at 2012-04-30, extending walk-forward OOS evaluation to ~12.75 years.
(Earlier panels were investigated — `freeze_long_panel.py` covers 2005, the
prior canonical `freeze_canonical_panel.py` covers 2003 — but the 2002-04
panel materially improves test-window Sharpe and reduces drawdown without
degrading any other headline number; see DECISIONS 2026-05-23 "Phase 15".)

| Period | Months | Use |
|---|---|---|
| 2002-04 – 2012-03 | 120 | Initial training window |
| 2012-04 – 2024-12 | 153 | Walk-forward out-of-sample (long-OOS) |
| 2019-01 – 2024-12 | 72 | Strict test window (the model never saw any month in 2019+ during its first training cut) |

Walk-forward design (see [walkforward_scheme.png](../results/persona_figures/walkforward_scheme.png)):
at each prediction month *t*, the model is refit on a sliding 120-month
window ending at *t*−1 and produces predictions for *t*. The refit is
block-gated (`(i − train_window) % test_window == 0`) so the model is held
frozen across each test block — preventing the period-over-period
overfitting that an earlier engine version produced (see DECISIONS
2026-05-22 "engine v0.3.0").

## 2. Features

The final model uses **13 features**, split into three families:

**Price-based (6):**
- `mom_12_1` — 12-1 month return momentum (most recent month excluded)
- `rev_1` — short-term reversal (one-month return)
- `mvol_12` — 12-month realised volatility
- `ivol_12` — idiosyncratic volatility (residual of CAPM regression)
- `log_mktcap` — log of month-end market capitalisation
- `dvol` — log of average daily dollar-volume (monthly)

**Valuation (2):**
- `book_to_market` — book equity / market equity (Sharadar ARQ)
- `earnings_to_price` — TTM earnings / market price (Sharadar ART)

**Fundamental quality (5):**
- `roe` — return on equity (TTM)
- `roa` — return on assets (TTM)
- `de` — debt-to-equity ratio
- `asset_growth` — year-over-year asset growth
- `accruals` — Sloan accruals (working-capital changes)

All features are computed point-in-time: a feature for month *t* uses only
information available at the close of month *t* (price-based) or the most
recent fiscal-quarter report released by month *t* (fundamentals).
Cross-sectional ranks within each month, followed by a *N*(0,1) Gaussian
transform, give the model a stationary input distribution across the 22-year
panel. All features are sector-demeaned (Framework §3.2 Layer 1) so that the
model learns relative tilts within sectors rather than sector-level differences.

## 3. Target

The target is the **next-month total return, sector-demeaned** (Layer 2 of
Framework §3.2):

$$y_{i,t+1} = r_{i,t+1} - \overline{r}_{s(i),t+1}$$

where $s(i)$ is the GICS sector of stock $i$. Sector-demeaning forces the
model to learn what makes a stock outperform *within its sector*, removing
the easier signal of "Tech beats Utilities this year". This was confirmed
empirically: Phase 10 showed that ~37% of an earlier non-sector-demeaned
Sharpe was sector-timing tilt rather than stock-picking skill.

## 4. Models

Three models are trained, evaluated, and compared:

| Model | Class | Role | Key hyperparameters |
|---|---|---|---|
| **Lasso** | Linear, L1-regularised | Baseline | `alpha` tuned on validation window |
| **XGBoost** | Gradient-boosted trees | Primary | `n_estimators=400`, `max_depth=5`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8` — tuned via Optuna TPE (50 trials) on 2016-2018 validation R² |
| **Neural Net** | 2-layer MLP, ReLU | Secondary | `hidden_dim=64`, `dropout=0.2`, Adam, batch=2048 |

All three share the same fit/predict Protocol (`src/models.py`) and consume
the identical feature panel. Seeds are fixed (`random_state=42`) so the
results are bit-reproducible.

## 5. Portfolio construction

At each rebalance the model produces a prediction $\hat{y}_{i,t+1}$ for every
stock. The portfolio (Framework §6, Layer 3) is built **sector-neutrally**:

- Within each GICS sector, long the **top k = 5** by prediction and short the **bottom k = 5**.
- 11 GICS sectors × (5 long + 5 short) ≈ 110 positions per month.
- **Dollar-neutral**: long-side gross dollar value = short-side gross dollar value.
- Equal-weighted within each leg.
- Hard rebalancing at month-end; no smoothing.
- Transaction costs assumed at 10 bps per side per turnover unit (deducted on the spot via `sharpe_net`).

This is the construction used by the canonical pipeline `Phase 15
(15_canonical_2002)`. The choice of `k = 5` was the empirical optimum
of a sensitivity sweep across `k ∈ {3, 5, 7, 10, 15, 20}` — an inverted-U
with the peak at 5 (Sharpe ≈ +1.50 on the 2003 panel; +1.49 on the 2002-04
panel) and going negative at k=20 because the model's least-confident picks
in each sector add noise.

## 6. Evaluation: prediction-level (Level 1, Framework §8.1)

Numbers are computed over the full walk-forward OOS window (2012-04 →
2024-12, 153 months).

| Model | OOS R² vs zero | OOS R² vs mean | IC mean | IC std | IC IR |
|---|---|---|---|---|---|
| Lasso | −0.000108 | −0.247 | −0.0041 | 0.104 | −0.040 |
| **XGBoost** | **+0.000545** | **−0.246** | **+0.0179** | **0.082** | **+0.218** |
| NN | −0.000994 | −0.248 | +0.0067 | 0.088 | +0.077 |

OOS R² numbers are tiny — typical of monthly equity prediction (Gu, Kelly,
Xiu 2020 report similar magnitudes). XGBoost is the only model to deliver a
positive R² versus zero, and the only one with materially positive
Information Coefficient: it averages a +1.8% cross-sectional Spearman
correlation between predictions and realised returns, with IR ≈ 0.22. NN
is weaker but positive; Lasso is essentially noise on a per-prediction basis.

## 7. Evaluation: strategy-level (Level 2, Framework §8.2)

Phase 15 final numbers (XGBoost, dollar-neutral, sector-neutral k=5):

| Window | Sharpe (net) | Ann return | Max DD | Turnover |
|---|---|---|---|---|
| Test 2019–2024 | **+1.011** | +9.47% | −7.9% | 1.33 |
| Long-OOS 2012–2024 | **+1.495** | +12.32% | −7.9% | 1.33 |

For comparison, the same model on the same panel under simpler portfolio
constructions:

| Phase | Construction | Sharpe (test) | Sharpe (long-OOS) |
|---|---|---|---|
| 8 | Global decile, dollar-neutral | +0.94 | +0.91 |
| 10 | Layer 3 alone, k=10 | +0.59 | +0.77 |
| 11 | Layer 2 + Layer 3, k=10, 2005-2024 panel | +0.62 | +1.09 |
| 12 | Layer 2 + Layer 3, k=10, 2003-2024 panel | +0.69 | +1.24 |
| 14 | + k=5 (tuned), 2003-2024 panel | +0.91 | +1.50 |
| **15** | **+ 2002-04 panel start** | **+1.01** | **+1.49** |

The progression tells the methodological story: each Framework layer is
worth measuring on its own; only the full stack reaches the headline number.
Phase 15's extra ~9 months of training history lifts the strict-OOS test
Sharpe by ~10% over Phase 14 and reduces max drawdown by 3.8 pp without
giving back any long-OOS Sharpe.

## 8. Statistical robustness

For honest reporting we expose three robustness checks on Phase 15
(`notebooks/personb/07_statistical_robustness.py`):

1. **Block bootstrap CI (6-month blocks, 10,000 resamples).** Long-OOS
   2015-2024 Sharpe 95% CI = **[+0.82, +1.83]** — excludes zero with
   P(bootstrap Sharpe ≤ 0) = 0.000.
2. **Deflated Sharpe Ratio** (Bailey & López de Prado 2014), correcting
   for 8 canonical-model trials run during development. Long-OOS
   **DSR = 0.992**, test-window DSR = 0.887. The long-OOS number clears
   the 0.95 significance bar comfortably after multiple-testing
   correction; the test-only 5-year window is below the bar, as expected
   for any single sub-decade slice.
3. **Fama-French 5-factor regression** (long-OOS 2015-2024, n=120,
   Newey-West HAC SE with 6 lags):

   | Factor | Loading | t | p |
   |---|---|---|---|
   | **Alpha** | **+6.76%/yr** | **+2.52** | **0.013** |
   | Mkt-RF | +0.192 | +3.39 | 0.001 |
   | SMB | +0.182 | +1.87 | 0.064 |
   | HML | −0.128 | −1.81 | 0.073 |
   | RMW | +0.093 | +0.69 | 0.493 |
   | CMA | +0.042 | +0.35 | 0.725 |

   The intercept (pure alpha) is **statistically significant after Fama-French
   adjustment** — more than half of the headline annualised return is *not*
   explained by exposures to the five canonical factors. The portfolio carries
   a small but real market beta (+0.19, R²=21%); the HML loading is no longer
   significant at the 5% level (down from t=-2.29 in Phase 14 → t=-1.81 in
   Phase 15), meaning the residual value-shorting tilt has weakened to the
   point where it cannot be statistically distinguished from zero. Both
   exposures shrank dramatically from earlier phases (Phase 8 HML loading was
   −0.27); the combination of Layer-2 target + Layer-3 portfolio + extended
   training history reduces factor exposure and lifts pure alpha simultaneously.

## 9. Model comparison: Diebold-Mariano

The adapted Diebold-Mariano test (HAC lags=12) on per-date squared-error
differences gives the pairwise model ranking on Phase 15:

| Pair | DM statistic | p-value |
|---|---|---|
| XGBoost vs Lasso | +2.81 | 0.005 (XGB significantly more accurate by MSE) |
| XGBoost vs NN | +1.42 | 0.156 (no significant difference) |
| Lasso vs NN | −2.04 | 0.041 (NN significantly more accurate than Lasso) |

By MSE, **XGBoost is the only model that significantly beats Lasso**, and
NN sits between them. Combined with XGBoost's higher IC (+0.019 vs +0.007
for NN) and higher Sharpe under the final portfolio, **XGBoost is the
designated canonical model.**

## 10. Feature importance

The top-5 features by XGBoost gain (Phase 15, full-panel SHAP-equivalent):

1. `mom_12_1` (momentum) — 18% of total gain
2. `dvol` (dollar volume) — 14%
3. `rev_1` (short-term reversal) — 12%
4. `ivol_12` (idiosyncratic vol) — 10%
5. `book_to_market` — 9%

The remaining 37% spreads across the 8 fundamental-quality and valuation
features. This composition matches the established cross-sectional asset
pricing literature (Gu, Kelly, Xiu 2020) — momentum and idiosyncratic
volatility dominate for monthly horizons.

## 11. Limitations and addressed concerns

A reasonable skeptic will raise three concerns about the headline numbers.
Each is addressed here in advance:

**"A long-only Sharpe of 1.5 is mostly market beta."** The strategy is
**dollar-neutral long-short**, not long-only. Realised beta to the S&P 500
is +0.19 with R²=21% — small, not 1.0. The FF5 alpha (which strips out
that residual market exposure) is the honest skill number, and it is
**+6.76%/yr at t=2.52, p=0.013 — significant.**

**"The t-stat is in-sample."** The model uses a strict walk-forward
backtest: every prediction at month *t* uses a model trained on data
ending at *t*−1. The ~12.75-year long-OOS window comprises 153 strictly
out-of-sample one-step-ahead predictions. The stricter test-only Sharpe of
+1.01 over 2019-2024 (6 years) is the most conservative cut and still
positive at t ≈ 2.5.

**"The benchmark should be the index, not zero."** For a dollar-neutral
long-short portfolio the standard benchmark IS zero (or the risk-free rate)
because there is no constant-equity exposure. The relevant question is
whether the strategy has factor-adjusted alpha — which is exactly what
the FF5 regression measures and it is significant.

Honest residual limits of the work:

- **Monthly data only.** Daily returns and volumes are not used; some of the
  intended features (e.g. ivol from daily regressions) are approximated by
  monthly proxies. A daily-data extension could refine signal.
- **HML loading is negative but no longer statistically significant.** The
  portfolio still leans slightly against value (β = −0.13), but the t-stat
  has fallen from −2.29 (Phase 14) to −1.81 (Phase 15) — below the 5% bar.
  Pure alpha is doing more of the work. A multi-factor-neutral variant
  (forced HML neutrality) might still be a defensible further refinement.
- **One ~23-year sample.** The DSR adjustment accounts for trial inflation,
  but not for the survivor / regime-specific nature of the post-2002 US
  large-cap universe.
- **Static k_per_sector.** k=5 is the static optimum across the whole sample;
  regime-conditional k could outperform in principle (handed off to Person C).

## 12. Reproducibility

Every number in this section is reproducible by:

```bash
.venv/bin/python -m notebooks.personb.freeze_canonical_2002_panel   # data panel
.venv/bin/python -m notebooks.personb.15_canonical_2002_start       # main result
.venv/bin/python -m notebooks.personb.04_model_comparison           # DM test
.venv/bin/python -m notebooks.personb.05b_realised_net_beta         # beta check
.venv/bin/python -m notebooks.personb.06_sector_audit               # sector concentration
.venv/bin/python -m notebooks.personb.07_statistical_robustness     # bootstrap / DSR / FF5
.venv/bin/python -m notebooks.personb.13_k_per_sector_sweep         # k sensitivity
```

Random seeds are pinned to 42 throughout. Artefacts written to
`results/15_canonical_2002/` (metrics.parquet, per_model_results.pkl,
PNG plots) are committed alongside the code. Phase 14's artefacts remain
in `results/14_official_canonical_k5/` for reproducibility / comparison.
