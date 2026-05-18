"""sanity_check_spliced.py - End-to-end smoke test of the backtest engine
on the CRSP+yfinance spliced panel.

Run with:
    python -m notebooks.persona.sanity_check_spliced

This is the validation gate that any change touching either the splice
or the backtest engine must clear. `sanity_check_crsp.py` covers the
engine on pure CRSP data; this one covers it on the full project test
window (2019-2024) where CRSP and yfinance both feed in.

Per Project Framework §4.6, a green run here is the precondition for
trusting any model output produced by Person B on the 2019-2024 window.

What this script does
---------------------
1. Loads 6 years (2019-01 to 2024-12) via `load_prices_spliced`. The
   first call bootstraps the yfinance cache (~50 sec); subsequent
   runs are served from parquet (~5 sec).
2. Restricts to the 250 tickers with the most month-observations
   across the full window -- a survivor-friendly proxy for "liquid
   names that span the splice". This is fine for a smoke test;
   real backtests apply `load_sp500_membership(asof)` per rebalance.
3. Pivots `ret` from the long (date, ticker) format to wide format
   (rows = month-ends, cols = ticker).
4. Builds a trivial 3-feature panel (random N(0,1)). The sanity
   models all ignore feature content; only the (date, asset) shape matters.
5. Runs `run_sanity_checks` and prints the verdict.

If all three checks pass, the splice is engine-safe AND the engine is
splice-safe -- Person B may proceed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import load_prices_spliced
from src.sanity import run_sanity_checks


def main() -> int:
    print("=" * 70)
    print("Backtest engine sanity check on SPLICED data (2019-2024)")
    print("=" * 70)

    # 1. Load spliced returns
    print("\n[1/4] Loading spliced CRSP+yfinance prices (2019-2024)...")
    panel = load_prices_spliced(start="2019-01-01", end="2024-12-31")
    src_counts = panel.groupby("source").size().to_dict()
    print(
        f"      {len(panel):,} rows, "
        f"{panel.index.get_level_values('ticker').nunique():,} unique tickers, "
        f"{panel.index.get_level_values('date').nunique()} months"
    )
    print(f"      sources: {src_counts}")

    # 2. Pick the 250 tickers with the most months observed across the
    #    full window (proxy for "liquid names that span the splice").
    print("\n[2/4] Selecting universe (top 250 by month-count)...")
    counts = panel.groupby(level="ticker").size()
    top_tickers = counts.sort_values(ascending=False).head(250).index.tolist()
    sub = panel.loc[(slice(None), top_tickers), :]
    print(
        f"      Kept {len(top_tickers)} tickers across "
        f"{sub.index.get_level_values('date').nunique()} months "
        f"({len(sub):,} rows total)"
    )

    # 3. Pivot ret -> wide (the format expected by the backtest engine)
    print("\n[3/4] Pivoting returns to wide format...")
    returns_wide = sub["ret"].unstack(level="ticker").sort_index()
    nan_frac = returns_wide.isna().sum().sum() / returns_wide.size
    print(
        f"      returns_wide shape: {returns_wide.shape}  "
        f"(NaN fraction: {nan_frac:.1%})"
    )

    # 4. Build trivial 3-feature panel
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
    print(
        f"      features shape: {features.shape}  "
        f"(MultiIndex: {features.index.names})"
    )

    # Run sanity checks
    print("\nRunning sanity checks (this may take ~30-60 sec)...\n")
    results = run_sanity_checks(returns=returns_wide, features=features)

    print(
        f"{'Check':>10s}  {'Sharpe':>10s}  {'Mean ret %':>12s}  "
        f"{'Pass':>6s}  Message"
    )
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
    print(
        f"Summary: {n_pass} / 3 sanity checks passed on spliced 2019-2024 data."
    )
    if n_pass == 3:
        print(
            "\nThe splice + backtest engine combo is ready for Person B. "
            "Both data sources (CRSP 2019-2022, yfinance 2023-2024) flow "
            "through the engine without breaking sanity."
        )
        return 0
    else:
        print(
            "\nDO NOT TRUST any backtest run on the spliced panel until "
            "all 3 checks pass. The splice may be introducing a regime "
            "break the engine treats as signal."
        )
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
