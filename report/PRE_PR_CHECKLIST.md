# Pre-PR checklist — `personb-models` → `main`

Use this list before opening the final PR. Each item is independently
verifiable; tick them off in the PR body.

## Blocking (must pass before merge)

- [ ] **Bowen re-froze `results/24_canonical_with_chmom/per_model_results.pkl`** with the corrected `is_bankruptcy_ticker` filter. New file is committed; old buggy-filter pkl no longer reflected in any number we publish.
- [ ] **Per-window numbers in REPORT.md §5 final-canonical table replaced** with authoritative values from the re-frozen pkl. The "delta-shifted estimate" footnote on long-OOS and test-OOS rows is removed.
- [ ] **Phase 25 re-run** on the new pkl:
  ```
  .venv/bin/python -m notebooks.personb.25_statistical_robustness_broad
  ```
  DSR table in §6 updated with the new bootstrap CIs + DSR figures.
- [ ] **Andrea (Person C) signed off §4** — leave a comment or thumbs-up on the PR confirming the regime-overlay write-up matches her understanding. The §4 caveat that says "Person C to co-review" must be removed once signed off.
- [ ] **Bowen signed off §2 + data lane** — same.
- [ ] **`.venv/bin/python -m src.sanity` passes 3/3** on the broad panel (Random/Oracle/Uniform). Re-run as the last step before merging.

## Pre-flight — no recompute needed

- [ ] Run `git diff main personb-models -- report/REPORT.md | wc -l`. Should be **< 1500 lines** of report diff. If much larger, something is off.
- [ ] All figures referenced in REPORT.md resolve. Quick check:
  ```
  grep -oE '\(\.\./[^)]+\.png\)' report/REPORT.md | sort -u | while read p; do
    f=${p#(\.\./}; f=${f%)}
    test -f "$f" && echo "OK $f" || echo "MISSING $f"
  done
  ```
- [ ] No `[TODO`, `[Person`, or `*[*` placeholder markers left in REPORT.md (Abstract / §1 / §3 / §5 / §6 / §7 cleared).
- [ ] DECISIONS.md tail entries (2026-05-24 universe-audit, Q-filter bug, Q-fix re-baseline) are coherent and dated.
- [ ] References section §9 includes every paper cited in body text. Grep for parenthetical citations:
  ```
  grep -oE '\([A-Z][a-z]+(\-[A-Z][a-z]+)*[, ]+(\d{4})\)' report/REPORT.md | sort -u
  ```
  Cross-reference against §9.

## Quality / regression

- [ ] `import` smoke-test all 8 migrated personb scripts + 4 Bowen scripts:
  ```
  for m in notebooks.personb.{24_canonical_with_chmom,24a_retune_xgb_with_chmom,24b_canonical_all_gkx,24b_retune_xgb_all_gkx,23c_k1_qfilter_canonical,23d_retune_xgb_qfiltered,23e_canonical_qfiltered_retuned,23g_canonical_qfiltered_orig_tune} notebooks.persona.{canonical_qfix_validate,decompose_qfix,canonical_true_top2000,canonical_broad_16feat}; do
    .venv/bin/python -c "import importlib; importlib.import_module('$m'); print('OK $m')" || echo "FAIL $m"
  done
  ```
- [ ] Regime overlay slim CSV loads cleanly:
  ```
  .venv/bin/python -c "from src.regime import make_regime_fn; import pandas as pd; print(make_regime_fn('results/regime_overlay_rules.csv')(pd.Timestamp('2020-03-31')))"
  ```
  Should print `{'leverage': 0.4}`.
- [ ] Untracked `results/0*_*/per_model_results.pkl` files either added to `.gitignore` or committed. `git status` should show no untracked pkls.
- [ ] `.DS_Store` is gitignored (mac artefact).

## Numbers consistency

- [ ] Abstract full-OOS Sharpe matches §5 full-OOS row matches DECISIONS.md re-baseline entry. (After Bowen's re-freeze, all three should read the same authoritative number.)
- [ ] Phase 25 DSR table in §6 uses N=25 (the bumped trial count).
- [ ] FF5 alpha t-stat in Abstract / §5 / §6 Momentum control / §6 DSR all from the same window (long-OOS) within ±0.3 t.
- [ ] Universe descriptions in Abstract / §2 / §3 / §5 all say "broad survivorship-free" + "~4,400 names/month median" (consistent with Bowen's relabel).

## PR body template

```markdown
## Summary

Final personb-models → main merge for the 5-week course project's
ML Factor Investing report. Locks in the Phase 24-RT canonical
(XGBoost on broad survivorship-free US universe, 14 features, k=20
per sector, corrected bankrupt-ticker filter, 10 bps/side costs).

Headline: full-OOS Sharpe +1.15, FF5 alpha +18.7%/yr at t=+6.85
(<0.001), DSR at N=25 trials = 0.87, Carhart momentum-controlled
alpha +20.1%/yr at t=+7.40. Alpha concentrates in the down-cap tail
(strict top-2,000 alpha is n.s. at t=0.96).

Honest caveats centred in §6: capacity binding at deployable AUM;
single-name fragility (LSCG +0.087 Sharpe, IONQ +0.042); regime
overlay net-zero on broad book due to COVID monthly-frequency lag.

## Test plan

- [ ] All 6 blocking items above completed
- [ ] sanity 3/3 pass
- [ ] All figure paths resolve
- [ ] At least one teammate approving review (Bowen + Andrea)
```

## Post-merge

- [ ] Tag the merge commit: `git tag v1.0-report-submitted` and push.
- [ ] Move `personb-models` branch to deleted/archived state.
- [ ] Delete obsolete spec docs: `report/PHASE_23_SPEC.md` (already
  marked superseded), `report/ALPHA_MODEL_SECTION.md` if redundant
  with REPORT.md §3.
