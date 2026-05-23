"""regime_alignment_check.py - Verify the regime overlay actually fires.

Run with:
    python -m notebooks.persona.regime_alignment_check

The overlay CSV is keyed by calendar month-ends (2015-01-31); the backtest
rebalances on trading-day month-ends (2015-01-30 when the 31st is a weekend).
`regime.make_regime_fn` now matches by month PERIOD, so both resolve to the
same rule. This script proves it, and shows what the old exact-date lookup
would have missed (the silent ~30% no-op this fix closes).

Pass criterion: every backtest rebalance date inside the overlay's coverage
window resolves to non-empty RegimeParams.
"""

from __future__ import annotations

import pandas as pd

from src.data_loader import load_prices_spliced
from src.regime import make_regime_dict, make_regime_fn

OVERLAY = "results/regime_overlay_rules.csv"


def main() -> int:
    regime_fn = make_regime_fn(OVERLAY)              # period-matched (fixed)
    exact = make_regime_dict(OVERLAY)                # exact-date dict (old behaviour)
    cov = pd.to_datetime(pd.read_csv(OVERLAY)["month_end"])
    cov_lo, cov_hi = cov.min(), cov.max()

    # Real backtest rebalance dates over the overlay coverage window.
    spliced = load_prices_spliced(
        start=cov_lo.strftime("%Y-%m-%d"), end=cov_hi.strftime("%Y-%m-%d")
    )
    rebal = sorted(spliced.index.get_level_values("date").unique())
    rebal = [d for d in rebal if cov_lo <= d <= cov_hi]

    period_hits = sum(1 for d in rebal if regime_fn(d))      # fixed lookup
    exact_hits = sum(1 for d in rebal if exact.get(pd.Timestamp(d)))  # old lookup

    n = len(rebal)
    print("=" * 64)
    print("Regime overlay alignment check")
    print("=" * 64)
    print(f"Coverage window      : {cov_lo.date()} -> {cov_hi.date()}")
    print(f"Backtest rebalances  : {n} months in window")
    print(f"Period-match (fixed) : {period_hits}/{n} resolve to a regime "
          f"({100*period_hits/n:.0f}%)")
    print(f"Exact-date (old bug) : {exact_hits}/{n} resolve "
          f"({100*exact_hits/n:.0f}%)  <- the silent no-op the fix closes")
    print()
    sample = rebal[len(rebal) // 2]
    print(f"Spot check regime_fn({sample.date()}) = {regime_fn(sample)}")

    if period_hits == n:
        print("\nPASS: every in-window rebalance resolves to a regime.")
        return 0
    print(f"\nFAIL: {n - period_hits} rebalance month(s) still miss the overlay.")
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
