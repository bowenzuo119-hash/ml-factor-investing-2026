"""decompose_qfix.py - which name drives the +0.116 Sharpe from the q-filter fix?

The corrected filter un-drops NDAQ (Nasdaq Inc, a real large-cap exchange) and
IONQ (a 2021 high-vol quantum SPAC) and moves full-OOS Sharpe 1.037 -> 1.153.
If NDAQ drives it, the higher headline is legitimate (we were wrongly excluding
a good large-cap); if IONQ drives it, the gain is fragile (one volatile recent
SPAC). Runs the NDAQ-only arm (keep NDAQ, still drop IONQ) to split it.

    KMP_DUPLICATE_LIB_OK=TRUE python -m notebooks.persona.decompose_qfix
"""

from __future__ import annotations

from src.data_loader import is_bankruptcy_ticker
from notebooks.persona.canonical_qfix_validate import run_arm

OLD = {"sharpe": 1.037, "alpha_t": 5.38}   # symbol-only (drops NDAQ + IONQ)
NEW = {"sharpe": 1.153, "alpha_t": 6.85}   # corrected (keeps both)


def ndaq_only(t):
    # drop the real bankrupts AND IONQ, but keep NDAQ
    return is_bankruptcy_ticker(t) or str(t).upper().strip() == "IONQ"


def main() -> int:
    r = run_arm(ndaq_only, "NDAQ-only (keep NDAQ, drop IONQ)")
    print("\n" + "=" * 60)
    print("Q-FILTER FIX DECOMPOSITION (full-OOS Sharpe / FF5 alpha t)")
    print("=" * 60)
    print(f"  {'arm':<22}{'Sharpe':>10}{'alpha t':>10}")
    print(f"  {'OLD (drop both)':<22}{OLD['sharpe']:>+10.3f}{OLD['alpha_t']:>+10.2f}")
    print(f"  {'+ NDAQ only':<22}{r['sharpe']:>+10.3f}{r['alpha_t']:>+10.2f}")
    print(f"  {'+ NDAQ + IONQ (NEW)':<22}{NEW['sharpe']:>+10.3f}{NEW['alpha_t']:>+10.2f}")
    print()
    print(f"  NDAQ contribution: {r['sharpe']-OLD['sharpe']:+.3f} Sharpe, {r['alpha_t']-OLD['alpha_t']:+.2f} t")
    print(f"  IONQ contribution: {NEW['sharpe']-r['sharpe']:+.3f} Sharpe, {NEW['alpha_t']-r['alpha_t']:+.2f} t")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
