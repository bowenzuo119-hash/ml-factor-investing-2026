"""regime_crisis_detection_rate.py - honest IS-vs-OOS crisis-detection rate.

Person C selected the 2-state HMM by its walk-forward crisis-detection rate over
seven known stress episodes (see notebooks/personc/week3_regime_finalise.py). The
REPORT §6 limitation needs the *honest* number: how much weaker is detection out
-of-sample (walk-forward, scaler refit each step) than in-sample (one full-sample
fit)? This script reproduces both from C's frozen label files so the §6 figures
(IS 91.7% / OOS 51.1%) are auditable, not copied from her write-up.

Method matches C's wf_crisis_score exactly: per stress window, the % of non-burn-
in months labelled 'crisis', then the macro-average across windows (equal weight
per episode). We also report month-weighted pooled recall for transparency.

    python -m notebooks.persona.regime_crisis_detection_rate
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WF_FILE = ROOT / "results" / "regime_walkforward_labels.csv"   # OOS (walk-forward)
IS_FILE = ROOT / "results" / "regime_labels_final.csv"         # in-sample (full fit)

# Identical to STRESS_PERIODS in notebooks/personc/week3_regime_finalise.py
STRESS_PERIODS = [
    ("2007-06", "2009-06", "GFC"),
    ("2010-04", "2010-07", "Euro I"),
    ("2011-07", "2011-10", "Euro II"),
    ("2015-08", "2016-02", "China/Oil"),
    ("2018-10", "2018-12", "Q4 2018"),
    ("2020-02", "2020-04", "COVID"),
    ("2022-01", "2022-10", "Inflation"),
]


def crisis_detection(labels: pd.Series, name: str) -> tuple[float, float]:
    """C's macro-average detection rate + month-weighted pooled recall."""
    labels = labels.copy()
    labels.index = pd.to_datetime(labels.index)
    per_window, pooled_hit, pooled_total = [], 0, 0
    print(f"\n  {name}")
    print(f"  {'episode':<12}{'months':>8}{'crisis-labelled':>18}")
    print("  " + "-" * 36)
    for start, end, lab in STRESS_PERIODS:
        win = labels.loc[start:end]
        win = win[win != "burn_in"].dropna()
        if len(win) == 0:
            print(f"  {lab:<12}{0:>8}{'(burn-in)':>18}")
            continue
        hit = int((win == "crisis").sum())
        pct = 100 * hit / len(win)
        per_window.append(pct)
        pooled_hit += hit
        pooled_total += len(win)
        print(f"  {lab:<12}{len(win):>8}{pct:>16.0f}%")
    macro = float(np.mean(per_window)) if per_window else 0.0
    pooled = 100 * pooled_hit / pooled_total if pooled_total else 0.0
    print(f"  --> macro-avg {macro:.1f}%   pooled recall {pooled:.1f}% "
          f"({pooled_hit}/{pooled_total} mo)")
    return macro, pooled


def main() -> int:
    wf = pd.read_csv(WF_FILE, parse_dates=["month_end"]).set_index("month_end")
    fin = pd.read_csv(IS_FILE, parse_dates=["month_end"]).set_index("month_end")

    m_is, p_is = crisis_detection(fin["regime_final"], "IN-SAMPLE (full-sample fit)")
    m_oos, p_oos = crisis_detection(wf["regime_final_wf"], "OUT-OF-SAMPLE (walk-forward)")

    print("\n" + "=" * 52)
    print("  REGIME CRISIS-DETECTION -- honest IS vs OOS")
    print("=" * 52)
    print(f"  in-sample  : macro {m_is:.1f}%   pooled {p_is:.1f}%")
    print(f"  walk-fwd OOS: macro {m_oos:.1f}%   pooled {p_oos:.1f}%")
    print(f"  macro drop  : {m_is - m_oos:+.1f} pts (the §6 honest finding)")
    print("\n  REPORT §6 cites macro 91.7% IS -> 51.1% OOS (selection metric),"
          "\n  pooled OOS 64.5%. The OOS loss is in the short fast wobbles"
          "\n  (Euro I 0/4, Q4-2018 0/3), the monthly-timing limit of §4.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
