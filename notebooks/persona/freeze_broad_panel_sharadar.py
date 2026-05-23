"""freeze_broad_panel_sharadar.py - Block C: freeze the broad-universe panel.

C1. Build the survivorship-free broad universe = union of load_universe_at(t)
    (top-2000 by marketcap, US common stock, alive at t) over every month-end
    in 2002-2024, then compute monthly total returns (SEP closeadj) for that
    union and freeze to:
        data/processed/returns_broad_sharadar_2002_2024.parquet
C2. Gate the frozen panel through the Random/Oracle/Uniform sanity suite. The
    pseudo-models don't read feature *values* (only the (date, asset) index),
    so a dummy feature frame built from the panel's own non-null cross-sections
    is enough to verify the engine is correctly wired on this panel
    (Random ~0, Oracle >5, Uniform ~0).

    python -m notebooks.persona.freeze_broad_panel_sharadar
"""

from __future__ import annotations

import sys

import pandas as pd

from src.data_loader import (
    PROCESSED_DIR,
    load_universe_at,
    compute_monthly_returns_sharadar,
)

START, END = "2002-01-01", "2024-12-31"
TOP_N = 2000
OUT = PROCESSED_DIR / "returns_broad_sharadar_2002_2024.parquet"


def build_union_universe() -> list[str]:
    month_ends = pd.date_range(START, END, freq="ME")
    union: set[str] = set()
    for i, me in enumerate(month_ends):
        u = load_universe_at(me, top_n_by_marketcap=TOP_N)
        union |= set(u["ticker"])
        if i % 24 == 0:
            print(f"  {me.date()}: month universe {len(u)}, running union {len(union)}")
    return sorted(union)


def run_sanity(returns: pd.DataFrame) -> bool:
    from src.sanity import run_sanity_checks

    # (date, asset) index over only the non-null cross-sections = the real
    # per-month investable universe. Feature values are irrelevant to the
    # pseudo-models, so a single dummy column suffices.
    feat_idx = returns.stack().index.rename(["date", "asset"])
    features = pd.DataFrame({"dummy": 0.0}, index=feat_idx)
    res = run_sanity_checks(returns=returns, features=features, train_window=12)

    print(f"\n{'check':>8s} {'sharpe':>9s} {'mean ret':>10s}  pass  message")
    print("-" * 90)
    for name, r in res.items():
        print(f"{name:>8s} {r['sharpe']:>+9.3f} {r['mean_return']*100:>+9.4f}%  "
              f"{'OK' if r['pass'] else 'FAIL':>4s}  {r['message']}")
    return all(r["pass"] for r in res.values())


def main() -> int:
    print(f"C1. Building broad union universe (top-{TOP_N}/mo) over {START}..{END} ...")
    union = build_union_universe()
    print(f"\nUnion universe: {len(union)} distinct tickers over the window")

    print("\nComputing monthly returns (SEP closeadj) for the union ...")
    returns = compute_monthly_returns_sharadar(START, END, tickers=union)
    # keep columns that actually have at least one return in-window
    returns = returns.dropna(axis=1, how="all")
    print(f"Panel: {returns.shape[0]} months x {returns.shape[1]} tickers "
          f"({returns.index.min().date()}..{returns.index.max().date()})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    returns.to_parquet(OUT)
    print(f"Froze -> {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")

    print("\nC2. Sanity gate on the frozen broad panel ...")
    ok = run_sanity(returns)
    print(f"\nSanity gate: {'PASS (3/3)' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
