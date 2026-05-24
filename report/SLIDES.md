---
title: "ML Factor Investing on a Survivorship-Free US Equity Universe"
subtitle: "Walk-forward backtest, audit-driven canonical, honest counterweights"
author: "Bowen Zuo, Nicolas Couto Mota"
date: "Team 6"
theme: "Madrid"
colortheme: "seahorse"
fonttheme: "professionalfonts"
aspectratio: 169
header-includes: |
  \setbeamertemplate{navigation symbols}{}
  \setbeamercolor{frametitle}{fg=white,bg=structure.fg}
---

## (N) Question — does ML cross-sectional alpha survive when the backtest is honest?

- **GKX 2020:** reports Sharpe ~1.5 on a broad CRSP ML factor strategy.
- **Three classic failure modes** for any single-OOS backtest:
  - Survivorship bias
  - Look-ahead leakage
  - Multiple-testing on a single historical path
- **Our project:** build the pipeline from scratch under strict audit controls; report whatever it produces — including the failures.
- **Methodological scope:** every ML concept lives here — walk-forward CV, hyperparameter tuning (Optuna/TPE), regularisation, gradient boosting, neural nets, statistical model comparison (Diebold-Mariano), bias-variance trade-off in feature choice.

## (B) Data lane — broad survivorship-free Sharadar universe, ~4,400 names/month

- **Single source: Sharadar/Nasdaq Data Link** (SF1 fundamentals, SEP prices, DAILY mcap, TICKERS, SP500, ACTIONS).
- **Universe:** alive set of historical top-2,000 by market cap, filtered by `firstpricedate ≤ asof ≤ lastpricedate` — survivorship-free by construction (delisted names retain their *historical* symbols: LEHMQ, ENRNQ, SIVBQ).
- ~4,400 names/month median; 5,897 unique tickers union 2002-2024.
- **Bankrupt-ticker filter:** `endswith('Q') AND isdelisted=='Y'` (gated on TICKERS metadata, corrected after audit found NDAQ + IONQ were being wrongly dropped).

![Universe coverage over time](results/persona_figures/universe_coverage_broad.png){width=70%}

## (B) Engine — walk-forward, PIT-correct, sanity-gated

- `run_walk_forward_backtest`: **120-month sliding training window, block-gated refit every 12 months**, 10 bps/side L1-turnover cost.
- **Point-in-time enforced** on both training and trading via `eligible_universe_fn`; `apply_pit_to_training=True`.
- **Sector-neutral construction** (top-k / bottom-k per GICS sector, k=20 → ~440 positions, dollar-neutral).
- **Sanity gate (Project Framework §4.6) — 3 synthetic-panel checks must pass every commit:**
  - RandomModel Sharpe ≈ 0
  - OracleModel Sharpe → ∞
  - UniformModel mean ≈ 0
- ✅ **3/3 PASS** on the canonical (random −0.51, oracle +99, uniform flat).

## (N) Alpha model — 14 features, 3 models share an interface, XGBoost wins decisively

- **Three models, same `fit`/`predict` Protocol:**
  - **Lasso** — regularised linear baseline
  - **XGBoost** — gradient-boosted trees, canonical (Optuna-TPE, 60 trials)
  - **PyTorch NN** — small 2-layer MLP with dropout, secondary
- **14 features** = GKX 2020 top-13 (12-1 momentum, reversal, vol, ivol, log mcap, log dvol, B/M, E/P, ROE, ROA, D/E, asset growth, accruals) + **`chmom`** (GKX rank-4 feature, |corr|<0.06 with existing momentum).
- **Target:** sector-relative monthly return (GKX Layer 2 — removes sector-level common signal).
- **Model selection:** **Diebold-Mariano test** with Newey-West HAC SE. XGBoost > Lasso (p<0.01), XGBoost > NN (p<0.05) on both MSE and Sharpe.

## (N) Walk-forward result — XGBoost full-OOS Sharpe +1.15, FF5 α +18.7%/yr at t=+6.85

\begin{center}
\large
\begin{tabular}{lccc}
\hline
\textbf{Window} & \textbf{Sharpe} & \textbf{FF5 $\alpha$/yr} & \textbf{t-stat} \\
\hline
\textbf{Full-OOS 2012–24} & \textbf{+1.15} & \textbf{+18.73\%} & \textbf{+6.85} \\
Long-OOS 2015–24 & +0.97 & +19.10\% & +6.00 \\
Test-OOS 2019–24 & +1.00 & +21.17\% & +5.00 \\
\hline
\end{tabular}
\end{center}

\vspace{0.4cm}

- All α t-stats ≥ +5, all p < 0.001
- ~440 positions/rebalance, 175% monthly turnover, 10 bps/side
- Max DD −34% (Feb-Mar 2020 COVID)
- Visual on next slide: equity curve with SPY benchmark + β-hedged pure-alpha line

## (N) The honest cumulative comparison vs S&P 500 — pure alpha is +614%, not +4,600%

\footnotesize

\begin{center}
\begin{tabular}{lccc}
\hline
\textbf{Strategy (Feb 2012 -- Dec 2024, net of 10 bps/side)} & \textbf{Cumulative} & \textbf{\$1 grows to} & \textbf{CAGR} \\
\hline
S\&P 500 passive (Mkt-RF + RF) & $+463\%$ & \$5.63 & $+14.3\%/$yr \\
\textbf{$\beta$-hedged pure alpha (honest headline)} & $\mathbf{+614\%}$ & \textbf{\$7.13} & $\mathbf{+16.5\%/}$\textbf{yr} \\
XGBoost canonical (gross, incl. $+1.3$ Mkt-$\beta$) & $+4{,}600\%$ & \$47.00 & $+34.7\%/$yr \\
\hline
\end{tabular}
\end{center}

\vspace{0.2cm}

**Three honest reads (key to the whole defence):**

\begin{enumerate}
\item \textbf{S\&P itself made $+463\%$ over 2012--24} (Sharpe $+0.99$ -- strong bull market). \$5.6 is the passive baseline.
\item \textbf{Pure alpha is $1.26\times$ S\&P, not $8\times$.} The $8.3\times$ gross ratio is \emph{leverage $\times$ compounding}, not skill -- our Sharpe $+1.15$ is only $+0.16$ above passive.
\item \textbf{At realistic 30 bps/side:} annual return $\sim+12\%/$yr $\Rightarrow$ cumulative $\sim+340\%$ ($\$4.4$), \emph{below} S\&P. Capacity at deployable AUM shrinks it further.
\end{enumerate}

\vspace{0.2cm}

\normalsize

\textbf{Bottom line:} we defend the $\beta$-hedged $+614\%$ as the honest deployable headline. The gross $+4{,}600\%$ is a backtest artefact of leverage in a 13-year bull market.

## (N) Equity curve — three-line decomposition (read the RED line)

![](results/final_canonical_plots/equity_curve_phase24_honest.png){width=85%}

\footnotesize **Read the chart bottom-to-top, NOT top-to-bottom.** The honest cross-sectional skill we should be defending is the \textcolor{red}{red $\beta$-hedged line (+614\%, $1.26\times$ S\&P)} — not the green gross line. The green line is the gross XGBoost canonical with $+1.3$ leveraged Mkt-$\beta$ in a 13-year bull market; the red line is what genuine ML alpha looks like after that exposure is stripped. \normalsize

## (N) Robustness battery — alpha survives every rigor check

\small

| Check | Result | Interpretation |
|---|---|---|
| **Feature-shuffle placebo** | +1.15 → −0.94 | rules out engine / target / cost leakage |
| Carhart 6F (FF5+UMD) | α rises to +20.1%/yr at t=+7.4, UMD β=−0.43 | momentum-AVERSE |
| Block bootstrap (6-mo blocks, 10k iter) | P(SR ≤ 0) = 0.0002 long-OOS | excludes zero >99% confidence |
| Deflated Sharpe Ratio (N=25) | DSR = 0.85–0.88 | survives multiple-testing penalty |
| Cost grid stress | α significant up to ~50 bps/side | survives conservative cost assumptions |
| Dense k-sweep + plateau zoom (CIs) | k=20 indistinguishable from every k in [10,20] | empirically defensible hyperparameter |

\normalsize

## (N) Placebo — the cleanest leakage test we have

![](results/qa_figures/placebo_vs_real.png){width=70%}

Shuffling the feature → ticker mapping within each rebalance date kills the edge: **+1.15 Sharpe collapses to −0.94** (mean of 2 seeds). The strategy *needs* the real feature content; the alpha is not a backtest-plumbing artefact.

## (B) Regime overlay — works on strict-S&P, net-zero on broad book

\small

- Andrea's workstream (she is not presenting today). **Gaussian HMM**, 2 states (`calm`/`crisis`), walk-forward expanding window with 60-mo burn-in, `StandardScaler` refit each step.
- **Overlay rule:** calm → 1.00× leverage, crisis → 0.40× (leverage-only).
- **Universe-dependent result:**
  - **Strict-S&P canonical:** DD improves −25.5% → −19.9% with small Sharpe gain ✅
  - **Broad canonical:** DD unchanged at −33.8% ❌

**Why:** the deepest drawdown is the Feb-Mar 2020 COVID crash; HMM flags March correctly but the overlay de-levers off the *prior* month-end label — January and February were both `calm`, so we entered the crash at full leverage. **Honest monthly-frequency timing limit, not a model flaw.**

\normalsize

![Monthly returns coloured by regime, COVID DD window shaded](results/persona_figures/overlay_failure_regime.png){width=72%}

## (B) The audit story — Sharpe +1.49 → −0.31 → +1.15

- **Phase 14 (pre-audit):** S&P 500 union, k=5, apparent Sharpe **+1.49**. Looked great.
- **PIT audit week 4:** engine wasn't filtering by point-in-time S&P 500 membership. 726 non-member positions traded in a 2012-19 RandomModel run.
- **Phase 15 (PIT applied to S&P):** Sharpe **−0.31**, α n.s. on every window. The +1.49 was 100% survivorship leak.
- **Rebuild on broad survivorship-free Sharadar universe:** Phase 23 → 24-RT, Sharpe **+1.15**, α significant at t > 5.
- **Two more bugs caught in week 5:**
  - Q-filter dropping NDAQ + IONQ (alive common stock) → corrected gate on `isdelisted=='Y'`
  - INCLUDE_FEATURES subset not applied → canonical was silently 16-feature → corrected to declared 14
- The **journey from +1.49 → +1.15** is the methodological contribution.

![Phase progression: long-OOS Sharpe history](results/final_canonical_plots/phase_progression_phase24.png){width=72%}

## (N) Honest counterweights — not market-neutral; alpha lives down-cap

- **Mkt-β ≈ +1.3** — strategy is *not* market-neutral despite dollar-neutral construction. Long-leg makes +37.9%/yr; short-leg is a near-zero-P&L hedge (−2%/yr). **~55% of headline return is leveraged market exposure; ~45% is genuine cross-sectional skill** (FF5-residual).
- **Where the alpha lives** — strict rolling top-2,000 (large-cap end) gives α = +1.8%/yr at **t=0.96 (n.s.)**. SMB-β also collapses (+1.26 → +0.15). Confirms **GKX 2020 §IV.D**: ML alpha lives in the small/mid-cap tail.
- **Capacity binding at deployable AUM** — 175% turnover × ~440 small/mid-cap positions = ~770% NAV/mo throughput; realistic small-cap costs likely above 30 bps/side.
- **Single-name fragility** (LO study, n=11): top name (LSCG) moves Sharpe by +0.089. Suggestive, underpowered.

![Where the alpha lives: broad vs strict top-2,000](results/qa_figures/where_alpha_lives.png){width=80%}

## (N) Conclusion + what we'd do differently

- **Defensible claim:** under realistic survivorship controls + PIT eligibility + 10 bps/side costs, the Phase 24-RT canonical produces statistically significant cross-sectional alpha on the 2012–2024 OOS sample, robust to **every** rigor check (Carhart, DSR, bootstrap, placebo, cost stress, k-sweep, sanity gates).
- **Honest bounds:** not market-neutral; capacity-constrained at deployable AUM; single-name fragility; regime overlay net-zero on broad book.
- **What we'd do differently** (in priority):
  1. Size-impact-aware cost modelling (Almgren-Chriss) on the actual cap-bucket distribution
  2. Sub-monthly regime detection (weekly/daily HMM) to catch fast crashes
  3. Cliff-style leave-one-out on full universe (n>>11) with bootstrap CIs
- **What we learned:** the audit journey (+1.49 → −0.31 → +1.15) is the methodological contribution; the alpha number is the empirical contribution. Build the audit into the pipeline, run it relentlessly, publish the journey honestly.

## Questions?

\Large

**Reference materials in the appendix:**

- Full report `report/REPORT.md`
- Decision log `DECISIONS.md` (1,233 lines, complete chronological provenance)
- Q&A prep doc with 25 anticipated questions
- All robustness artefacts under `results/`

\normalsize

\vfill

*Repository:* https://github.com/bowenzuo119-hash/ml-factor-investing-2026
