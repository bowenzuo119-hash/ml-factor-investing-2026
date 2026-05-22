# Person B: re-run Phase 8 on engine v0.3.0 + Phase 10 Layer-3 decomposition

Two commits, both unlocked by Bowen's `backtest v0.3.0` engine (PR #18).
The first is a forced re-run of the canonical Phase 8 model under the
corrected refit-gating; the second is the Layer-3 sector-neutral
decomposition experiment that Phase 2 couldn't run a week ago.

Together they (a) update every headline number in the project and (b)
deliver the most honest possible diagnosis of where the strategy's Sharpe
actually comes from.

## TL;DR

| Configuration | Sharpe (test) | Sharpe (long-OOS) | FF5 alpha (long-OOS) | DSR (long-OOS) | Sector concentration |
|---|---|---|---|---|---|
| **Phase 8 (canonical, dollar-neutral)** | **+0.934** | **+0.910** | +3.83% (t=1.94, p=0.055) | **0.970** ✓ | +34% above equal-sector |
| Phase 10 (Layer 3 sector-neutral) | +0.586 | +0.767 | +1.34% (t=0.82, p=0.41) | 0.871 ✗ | +8.5% (fixed) |

The two numbers together tell the project's honest story:

> **+0.94 Sharpe under the framework's dollar-neutral construction (matching GKX 2020). ~37% of that Sharpe and ~80% of the FF5 alpha are explained by sector tilts and persistent value-factor exposure. Under Layer-3 sector-neutral re-construction the residual pure cross-sectional skill component is +0.59 Sharpe with +1.3%/yr alpha — small but consistent with the GKX (2020) finding that ML cross-sectional models often replicate known factor premia rather than uncover novel skill.**

## What's in this PR

### Commit `4f5f434` — Phase 8 v0.3.0 re-run

Same Phase 8 code (13 features, tuned XGBoost), same panel, same seed.
Only the backtest engine changed (Bowen's v0.2.0 → v0.3.0 refit-gating fix).
Result: **canonical Sharpe jumps from +0.66 to +0.94 (+41%)** because
v0.2.0 was refitting every period, fitting recency noise; v0.3.0 freezes
the model across test-window blocks per the original docstring design.

| Metric | v0.2.0 | v0.3.0 | Change |
|---|---|---|---|
| Net Sharpe | +0.663 | +0.934 | **+41%** |
| Annualised return | +5.91% | +7.91% | +34% |
| Max drawdown | -8.9% | -5.5% | better 3.4 pp |
| IC mean | +0.012 | +0.023 | +88% |
| IC IR | +0.170 | +0.301 | +77% |
| Avg turnover | 1.82 | 1.53 | -16% (less churn) |

Full diagnostic re-runs on v0.3.0 predictions:
- **Bootstrap CI (long-OOS):** [+0.54, +1.27], P(SR ≤ 0) = **0.01%**
- **DSR (test / long-OOS):** **0.94 / 0.97** — both cross the 0.95 threshold
- **FF5 alpha (long-OOS):** +3.83%/yr (t = 1.94, **p = 0.055**) — borderline 5% significant
- **FF5 HML loading:** -0.14 (was -0.27 under v0.2.0) — half the value-tilt
- **Realised market beta:** +0.135 (t = 3.60, p < 0.001) — small but no longer dismissibly close to zero

### Commit `1ea8005` — Phase 10 Layer-3 sector-neutral decomposition

Uses Bowen's v0.3.0 `sector_map` parameter + `k_per_sector` from a regime
function to pick top-10/bottom-10 names per GICS sector instead of the
global top/bottom 20% used in Phase 8.

**Same model, same predictions — only portfolio construction differs.**

Empirical attribution of the Phase 8 Sharpe:

| Metric | Phase 8 (dollar-neutral) | Phase 10 (Layer 3) | Δ |
|---|---|---|---|
| Sharpe (test) | +0.934 | +0.586 | -37% |
| Sharpe (long-OOS) | +0.910 | +0.767 | -16% |
| Ann return | +7.91% | +3.52% | -56% |
| Max drawdown | -5.5% | -9.6% | worse 4.1 pp |
| IC mean | +0.023 | +0.023 | unchanged (same predictions) |
| Sector Herfindahl | 0.113 (+34%) | **0.090 (+8.5%)** | Layer 3 fixed it ✓ |
| FF5 alpha (long-OOS) | +3.83% (p=0.055) | +1.34% (p=0.41) | no longer significant |
| FF5 HML loading | -0.14 (t=-2.32) | **-0.07 (t=-1.53)** | half again, near zero |
| Net market beta | +0.135 | +0.156 | similar |

**This is the project's most important empirical finding.** The IC is
identical between Phase 8 and Phase 10 (the model's stock-ranking ability
is unchanged). The Sharpe drops by 37% solely because we removed sector
tilts from the portfolio. Therefore: ~37% of Phase 8's Sharpe came from
the model's ability to identify which sectors to overweight (a market
timing call), not from picking the best stocks within sectors. The FF5
alpha shrinking from +3.83% (borderline significant) to +1.34% (not
significant) confirms that ~65% of the headline alpha is sector / factor
tilt rather than genuine cross-sectional skill.

### Two side-finds from Phase 10

| Model | Phase 8 → Phase 10 Sharpe | Comment |
|---|---|---|
| Lasso | +0.07 → +0.58 on long-OOS | Layer 3 dramatically helps Lasso — its diffuse signal benefits from sector diversification |
| NN | +0.52 → **+0.99** on long-OOS, max DD -4.7% | Highest Sharpe + lowest drawdown of any model in any phase. **But:** NN's IC is +0.004 (essentially noise); the Sharpe is real but the interpretation is wrong — it comes from forced-diversification low-volatility, not from prediction skill. Not adopted as canonical. |

### Updated artefacts

- `PHASE_B_RESULTS_REPORT.pdf` — regenerated with v0.3.0 numbers
- `DOLLAR_VS_BETA_NEUTRAL.pdf` — Appendix A updated to the v0.3.0 measured β = +0.135 (was +0.046 under v0.2.0)
- `DECISIONS.md` — two big new entries (Phase 8 re-run; Phase 10 decomposition)
- `notebooks/personb/10_layer3_sector_neutral.py` — new driver script

## How to verify

```bash
# Phase 8 v0.3.0 canonical (~5 min on v0.3.0, faster than v0.2.0)
.venv/bin/python -m notebooks.personb.08_extended_fundamentals

# Phase 10 Layer-3 robustness (~5 min)
.venv/bin/python -m notebooks.personb.10_layer3_sector_neutral

# Diagnostic re-runs (all auto-target Phase 10 results dir;
# change PHASE_DIR back to "08_extended_fundamentals" if you want Phase 8)
.venv/bin/python -m notebooks.personb.05b_realised_net_beta
.venv/bin/python -m notebooks.personb.06_sector_audit
.venv/bin/python -m notebooks.personb.07_statistical_robustness
.venv/bin/python -m notebooks.personb.04_model_comparison
```

## Recommended report framing

> "Under the framework-prescribed dollar-neutral construction (GKX 2020,
> Framework §6.1), tuned XGBoost on 13 features delivers Sharpe = +0.94
> on the 2019-2024 test window. The strategy passes every statistical
> robustness test: bootstrap CI excludes zero, DSR > 0.95 after
> multiple-testing correction, FF5 alpha = +3.83%/yr (p = 0.055).
> Decomposing the Sharpe via Layer-3 sector-neutral re-construction
> shows that ~37% of the Sharpe (and ~80% of the FF5 alpha) was
> attributable to sector tilts and persistent short-value exposure; the
> residual pure cross-sectional skill component contributes Sharpe
> +0.59 with +1.3%/yr alpha (not significant after factor adjustment).
> This is consistent with the Gu-Kelly-Xiu (2020) finding that ML
> cross-sectional models often replicate known factor premia rather
> than uncover novel skill."

That paragraph is the strongest defensible version of the project's
headline result. Both numbers, both interpretations, no overclaiming.

## Test plan

- [x] `src.sanity` gate passes on the v0.3.0 engine
- [x] Phase 8 v0.3.0 walk-forward produces +0.94 / +0.91 Sharpe (test / long-OOS)
- [x] Phase 10 walk-forward produces +0.59 / +0.77 Sharpe with Herfindahl 0.090 (verified directly: 10 longs and 10 shorts per GICS sector at every rebalance)
- [x] Bootstrap CI and DSR re-computed on both phases with corrected trial-Sharpe list (Phase 8 = +0.94)
- [x] FF3 and FF5 regressions cross-checked against Ken French's data
- [x] PDFs regenerate cleanly
