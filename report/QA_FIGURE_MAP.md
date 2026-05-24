# Q&A Figure Map — which slide / backup figure to pull up for each question

> **Use case:** during the live Q&A, the named speaker pulls up the corresponding figure (either already in the slide deck, or as a backup figure in the appendix PDF) while delivering the answer. Visuals are more persuasive than spoken numbers.
>
> **Sources:**
> - `results/qa_figures/` — 4 new defense figures purpose-built for Q&A (see `notebooks/personb/qa_figures.py`)
> - `results/final_canonical_plots/` — the 4 headline canonical plots
> - `results/persona_figures/` — Bowen's data-lane + presentation visuals
> - `results/27_k_sweep_dense/`, `results/27b_k_sweep_plateau/` — k-sweep figures
> - `results/25_statistical_robustness_broad/` — bootstrap distribution
> - `results/long_short_decomp/` — long-leg vs short-leg
>
> **Have all of these as backup slides at the end of the deck**, hidden from the main flow, ready to jump to with the slide-picker if asked.

---

## Tier-1 Q&A figures (must have ready, asked >90% likely)

| Q | Question | Speaker | Primary figure | File path |
|---|---|---|---|---|
| Q1 | Is the +1.15 a backtest artefact? | N | **Placebo bar chart** | `results/qa_figures/placebo_vs_real.png` |
| Q1 | (supporting) sanity 3/3 | N | Sanity-gate 3-panel | `results/persona_figures/sanity_3panel.png` |
| Q2 | Why XGBoost over NN / Lasso? | N | **Model comparison** (Sharpe + α t-stat × 3 models × 3 windows) | `results/qa_figures/model_comparison.png` |
| Q3 | Hyperparameter tuning strategy? | N | *(no figure — best-params table from REPORT.md §3)* | — |
| Q4 | How handle look-ahead bias? | B | Universe coverage over time | `results/persona_figures/universe_coverage_broad.png` |
| Q4 | (supporting) survivorship comparison | B | Survivorship-correction visual | `results/persona_figures/universe_survivorship_comparison.png` |
| Q5 | What baseline are you comparing against? | N | **Model comparison** (same as Q2) | `results/qa_figures/model_comparison.png` |
| Q5 | (supporting) phase-progression | B | Phase progression | `results/final_canonical_plots/phase_progression_phase24.png` |
| Q6 | Why 14 features? | N | Feature-correlation heatmap | `results/persona_figures/feature_correlation.png` |
| Q7 | What's the DSR? | N | Bootstrap-distribution histogram | `results/25_statistical_robustness_broad/bootstrap_distribution.png` |
| Q8 | Why is the alpha so high (~+18%/yr)? | N | **Where the alpha lives** (broad vs strict top-2000, 4-panel) | `results/qa_figures/where_alpha_lives.png` |
| Q9 | How do you know the alpha isn't just momentum? | N | **Momentum control** (FF5 vs Carhart α + UMD loading) | `results/qa_figures/momentum_control.png` |
| Q10 | Walk through the audit story | B | Phase progression | `results/final_canonical_plots/phase_progression_phase24.png` |
| Q10 | (supporting) leaky vs honest equity | B | Leaky vs honest equity-curve | `results/persona_figures/leaky_vs_honest_equity.png` |

---

## Tier-2 Q&A figures (have ready, asked ~50/50)

| Q | Question | Speaker | Figure | File path |
|---|---|---|---|---|
| Q11 | Why sector-neutral + k=20? | N | k-sweep dense curve | `results/27_k_sweep_dense/k_sweep_dense.png` |
| Q11 | (supporting) k=10..20 plateau with bootstrap CIs | N | k-sweep plateau zoom | `results/27b_k_sweep_plateau/k_sweep_plateau_zoom.png` |
| Q11 | (supporting) sector exposure of canonical | N | Sector exposure heatmap | `results/persona_figures/sector_exposure.png` |
| Q12 | What about transaction costs? | B | *(no dedicated figure — cite the 10/30/50 bps grid table from REPORT.md §5)* | — |
| Q12 | (supporting) gross vs net equity | B | Gross-vs-net equity | `results/persona_figures/gross_vs_net_equity.png` |
| Q13 | How do you do inference on a single OOS path? | N | Bootstrap distribution | `results/25_statistical_robustness_broad/bootstrap_distribution.png` |
| Q14 | Regime overlay mechanism? | B | Overlay failure regime | `results/persona_figures/overlay_failure_regime.png` |
| Q15 | Why long-short, not long-only? | N | **Long-leg vs short-leg P&L decomposition** | `results/long_short_decomp/long_short_decomp_phase24.png` |
| Q15 | (supporting) long-vs-short scatter | N | Monthly scatter | `results/long_short_decomp/long_short_scatter_phase24.png` |
| Q16 | How big is the dataset? | B | Universe coverage over time | `results/persona_figures/universe_coverage_broad.png` |
| Q17 | What's the loss function? | N | *(no figure — verbal explanation)* | — |
| Q18 | Did you check for feature leakage? | N | **Placebo bar chart** + feature correlation | `results/qa_figures/placebo_vs_real.png` + `results/persona_figures/feature_correlation.png` |

---

## Tier-3 / adversarial Q&A

| Q | Question | Speaker | Figure | File path |
|---|---|---|---|---|
| Q19 | Generalisation to non-US? | N | *(no figure — verbal)* | — |
| Q20 | If you had another month? | N | *(no figure — verbal extension list)* | — |
| Q21 | Why monthly rebalancing? | B | Cost grid in REPORT.md §5 | (text only) |
| Q22 | How do you reproduce results? | B | *(verbal: cite repository + DECISIONS.md)* | (text only) |
| Q23 | Lucky path / multiple-testing? | N | Bootstrap CI + DSR figure | `results/25_statistical_robustness_broad/bootstrap_distribution.png` |
| Q23 | (supporting) phase progression | B | Phase progression | `results/final_canonical_plots/phase_progression_phase24.png` |
| Q24 | Mkt-β ≈ +1.3 — just leveraged market? | N | **FF5 decomposition** (alpha vs factor exposure) | `results/final_canonical_plots/ff5_decomposition_phase24.png` |
| Q24 | (supporting) equity curve with β-hedged line | N | Honest equity curve | `results/final_canonical_plots/equity_curve_phase24_honest.png` |
| Q25 | Conflict of interest in self-audit? | B | Phase progression + leaky vs honest | `results/final_canonical_plots/phase_progression_phase24.png` + `results/persona_figures/leaky_vs_honest_equity.png` |

---

## Deck logistics

**Recommended slide-deck structure for backups:**

- **Main flow (slides 1-11):** the speaking outline in `PRESENTATION_OUTLINE.md` — uses only the 8-10 most impactful figures.
- **Appendix slides (12+, hidden):** all the Tier-1 + Tier-2 figures above as one-figure-per-slide, with the headline question as the slide title. Examiner asks Q8 → slide-picker jumps directly to the "Where the alpha lives" appendix slide.
- **Backup PDF** (the optional appendix submission): the full `report/REPORT.md` typeset + all figures listed above + `DECISIONS.md` extract — single PDF, ≤ 30 pages.

**Figure-quality checklist before printing/exporting slides:**

- [ ] All 4 new Q&A figures (`results/qa_figures/*.png`) export cleanly at 170 dpi (already done by the script).
- [ ] Each backup-slide figure has the question text as the slide title (not just the figure title).
- [ ] Each backup slide has 1-2 bullets of speaker-note text reminding the speaker of the verbal answer.
- [ ] Sanity check: every figure on every slide reads cleanly from the back of the room (test by projecting at ~50% reduced size).

---

## Quick visual sanity-check — what each new Q&A figure shows

### `results/qa_figures/placebo_vs_real.png` (Q1)

4-bar chart: REAL features Sharpe +1.15 (navy) vs SHUFFLED seed=0 −1.03 / seed=1 −0.85 / mean −0.94 (red). Big "Δ Sharpe ≈ 2.09" annotation between REAL and shuffled-mean. Single-figure visual answer to "is this leakage?".

### `results/qa_figures/model_comparison.png` (Q2, Q5)

Two side-by-side panels: (a) Sharpe ratio for Lasso / XGBoost / NN across full-OOS / long-OOS / test-OOS; (b) FF5 alpha t-stat across the same 3×3 grid. XGBoost clearly dominates on both. Reference line at t=2 on the right panel.

### `results/qa_figures/where_alpha_lives.png` (Q8)

Four-panel small-multiples: FF5 α / α t-stat / Mkt-β / SMB-β for Broad (~4,400 names) vs Strict-top-2,000. The collapse from +18.7% to +1.8% α and from +1.26 SMB-β to +0.15 are both visible at a glance. The "n.s. ↓" annotation on the top-2,000 t-stat panel makes the lack-of-significance explicit.

### `results/qa_figures/momentum_control.png` (Q9)

Two panels: (a) FF5 α (+17.7%) vs Carhart-6F α (+20.1%) bars with the green "α RISES by +2.4 pp" annotation arrow; (b) UMD coefficient = −0.43 with "MOMENTUM-AVERSE" label in red. The visual makes "alpha is NOT repackaged momentum" obvious.
