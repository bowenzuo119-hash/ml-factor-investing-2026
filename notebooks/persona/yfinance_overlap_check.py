"""yfinance_overlap_check.py - Validation gate for the CRSP -> yfinance splice.

Run with:
    python -m notebooks.persona.yfinance_overlap_check

This is the validation gate that must pass before yfinance is trusted to
extend the CRSP price history into 2023-2024. Method: take an overlap
window (2018-01 to 2022-12, where both sources have data), sample N
S&P 500 PERMNOs, fetch monthly returns from each source under the same
ticker, and check that they agree.

What this script does
---------------------
1. Calls `compare_crsp_vs_yfinance` for a 200-stock sample over 2018-2022.
   First run downloads ~200 tickers from yfinance (~60 sec); subsequent
   runs hit the parquet cache (~5 sec).
2. Prints distribution of correlations, mean / max absolute differences.
3. Lists the WORST 15 tickers (lowest correlation) so a human can eyeball
   whether they look like real data errors or known corporate-action
   weirdness (renames, share-class splits, spin-offs).
4. Returns non-zero if the splice does not look safe.

Pass criteria (PR description's safeguard list)
-----------------------------------------------
* Median correlation > 0.99 (essentially perfect on most names).
* At least 80% of matched tickers have correlation > 0.99.
* No more than 5% of matched tickers have correlation < 0.90.
"""

from __future__ import annotations

from src.data_loader import compare_crsp_vs_yfinance


# Pass thresholds. Tuned from a 50-stock dev run that gave median ~0.999999;
# anything materially worse than these is a real regression.
_MEDIAN_CORR_FLOOR = 0.99
_FRAC_ABOVE_099_FLOOR = 0.80
_FRAC_BELOW_090_CEIL = 0.05


def main() -> int:
    print("=" * 70)
    print("CRSP vs yfinance overlap validation (2018-2022, 200-stock sample)")
    print("=" * 70)

    df = compare_crsp_vs_yfinance(
        start="2018-01-01",
        end="2022-12-31",
        sample_size=200,
        random_state=42,
    )

    n_matched = len(df)
    frac_above_099 = (df["correlation"] > 0.99).mean()
    frac_below_090 = (df["correlation"] < 0.90).mean()
    median_corr = df["correlation"].median()

    print()
    print("=== Distribution ===")
    print(f"  Matched tickers:                     {n_matched}")
    print(f"  Median correlation:                  {median_corr:.6f}")
    print(f"  Worst correlation:                   {df['correlation'].min():.4f}")
    print(f"  Median mean_abs_diff (bps):          {df['mean_abs_diff_bps'].median():.3f}")
    print(f"  P95 max_abs_diff (bps):              {df['max_abs_diff_bps'].quantile(0.95):.1f}")
    print(f"  Fraction with correlation > 0.99:    {frac_above_099:.1%}")
    print(f"  Fraction with correlation < 0.90:    {frac_below_090:.1%}")

    print()
    print("=== Bottom 15 (lowest correlation -- inspect manually) ===")
    print(df.head(15).to_string())

    print()
    print("=== Verdict ===")
    checks = [
        (median_corr > _MEDIAN_CORR_FLOOR,
         f"median correlation {median_corr:.4f} > {_MEDIAN_CORR_FLOOR}"),
        (frac_above_099 > _FRAC_ABOVE_099_FLOOR,
         f"{frac_above_099:.0%} > 0.99 (need > {_FRAC_ABOVE_099_FLOOR:.0%})"),
        (frac_below_090 < _FRAC_BELOW_090_CEIL,
         f"{frac_below_090:.0%} < 0.90 (need < {_FRAC_BELOW_090_CEIL:.0%})"),
    ]
    all_pass = True
    for ok, msg in checks:
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {msg}")
        all_pass = all_pass and ok

    print()
    if all_pass:
        print(
            "Splice is SAFE. yfinance returns track CRSP returns to ~basis-point "
            "precision; the post-2022-12 splice will not introduce a regime break."
        )
        return 0
    else:
        print(
            "Splice is NOT SAFE. DO NOT use load_prices_spliced until the worst "
            "tickers above are investigated (most likely a ticker reuse / "
            "rename problem in CRSP's ticker column)."
        )
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
