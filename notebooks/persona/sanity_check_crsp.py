"""sanity_check_crsp.py - End-to-end smoke test of the backtest engine
on real CRSP data.

Run with:
    python -m notebooks.persona.sanity_check_crsp

This is the validation gate that any backtest-engine change must clear.
The synthetic-panel tests in `python -m src.sanity` are necessary but
not sufficient -- real CRSP data has NaN-filled ragged panels, alpha
RET codes ("B"/"C"), wildly different volatilities across stocks, and
real corporate actions. If the engine works here, it works.

What this script does
---------------------
1. Loads 6 years of CRSP monthly data (2010-01 to 2015-12).
2. Restricts the universe to the 200 PERMNOs that appear in the most
   months (a crude proxy for "the big liquid names"). NOTE: this is
   biased toward survivors WITHIN the test window -- acceptable for
   a smoke test, NOT acceptable for a real backtest. Real backtests
   filter via `load_sp500_membership` at each rebalance.
3. Pivots the `ret` column to wide format (rows = month-ends, cols = PERMNO).
4. Builds a trivial 3-feature panel (random N(0,1)) -- the sanity
   models ignore feature content; only the (date, permno) shape matters.
5. Runs `run_sanity_checks` and prints the verdict.

If all three checks pass on real data, the engine is ready for Person B
to plug in a real model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import load_prices
from src.sanity import run_sanity_checks


def main() -> int:
    print("=" * 70)
    print("Backtest engine sanity check on REAL CRSP data (2010-2015)")
    print("=" * 70)

    # 1. Load CRSP
    print("\n[1/4] Loading CRSP monthly prices...")
    prices = load_prices(start="2010-01-01", end="2015-12-31")
    print(f"      {len(prices):,} rows, "
          f"{prices.index.get_level_values('permno').nunique():,} unique PERMNOs, "
          f"{prices.index.get_level_values('date').nunique()} months")

    # 2. Pick the 200 most-frequently-appearing PERMNOs as a smoke universe
    print("\n[2/4] Selecting universe (top 200 by month-count)...")
    counts = prices.groupby(level="permno").size()
    top_permnos = counts.sort_values(ascending=False).head(200).index.tolist()
    sub = prices.loc[(slice(None), top_permnos), :]
    print(f"      Kept {len(top_permnos)} PERMNOs across "
          f"{sub.index.get_level_values('date').nunique()} months "
          f"({len(sub):,} rows total)")

    # 3. Pivot ret -> wide format expected by the backtest
    print("\n[3/4] Pivoting returns to wide format...")
    returns_wide = (
        sub["ret"]
        .unstack(level="permno")
        .sort_index()
    )
    print(f"      returns_wide shape: {returns_wide.shape}  "
          f"(NaN count: {returns_wide.isna().sum().sum():,})")

    # Build a trivial 3-feature panel. The sanity models all ignore the
    # feature values (RandomModel uses RNG, OracleModel reads future
    # returns, UniformModel returns a constant); only the (date, asset)
    # MultiIndex shape matters.
    print("\n[4/4] Building trivial feature panel...")
    rng = np.random.default_rng(0)
    feat_idx = pd.MultiIndex.from_product(
        [returns_wide.index, returns_wide.columns], names=["date", "asset"]
    )
    features = pd.DataFrame(
        rng.standard_normal((len(feat_idx), 3)),
        index=feat_idx,
        columns=["feat1", "feat2", "feat3"],
    )
    print(f"      features shape: {features.shape}  "
          f"(MultiIndex: {features.index.names})")

    # Run the three sanity checks
    print("\nRunning sanity checks (this may take ~30-60 sec on real data)...\n")
    results = run_sanity_checks(returns=returns_wide, features=features)

    print(f"{'Check':>10s}  {'Sharpe':>10s}  {'Mean ret %':>12s}  {'Pass':>6s}  Message")
    print("-" * 110)
    for name, r in results.items():
        flag = "PASS" if r["pass"] else "FAIL"
        print(
            f"{name:>10s}  "
            f"{r['sharpe']:>+10.3f}  "
            f"{r['mean_return']*100:>+12.4f}  "
            f"{flag:>6s}  {r['message']}"
        )

    n_pass = sum(1 for r in results.values() if r["pass"])
    print()
    print(f"Summary: {n_pass} / 3 sanity checks passed on real CRSP data.")
    if n_pass == 3:
        print("\nThe backtest engine is ready for Person B's real model.")
        return 0
    else:
        print("\nDO NOT TRUST any subsequent backtest output until all 3 checks pass.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
