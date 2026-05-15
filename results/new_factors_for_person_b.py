
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Additional factors â€” Person C contribution (Week 3, Days 19â€“21)
# Add these functions to src/factors.py
# Format: each function returns a DataFrame (index=dates, cols=tickers)
#         cross-sectionally z-scored (mean=0, std=1) each month.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

import numpy as np
import pandas as pd


def zscore_cross_section(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectionally z-score each row.
    Subtract the row mean and divide by the row std.
    This is applied to every factor so they are all on the same scale.
    """
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)


def factor_short_term_reversal(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Short-Term Reversal Factor
    â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    Definition:
        Negative of the previous calendar month's total return.
        Sign is flipped because last month's losers tend to mean-revert
        and outperform next month (and vice versa).

    Formula:
        rev_t = - (price_t / price_{t-1} - 1)

    Academic basis:
        Jegadeesh (1990) â€” short-horizon reversals are well-documented.
        This is distinct from the 12-month momentum factor (Person B)
        which skips the most recent month precisely to avoid this effect.

    Parameters:
        prices_df : pd.DataFrame
            Monthly adjusted closing prices.
            Index = dates (month-end), columns = ticker symbols.

    Returns:
        pd.DataFrame â€” same shape as prices_df, cross-sectionally z-scored.
    """
    # Monthly simple return
    monthly_ret = prices_df.pct_change(1)

    # Sign-flip: yesterday's losers are today's longs
    reversal = -monthly_ret

    return zscore_cross_section(reversal)


def factor_accruals(net_income: pd.DataFrame,
                    operating_cf: pd.DataFrame,
                    total_assets: pd.DataFrame) -> pd.DataFrame:
    """
    Accruals Factor (Earnings Quality)
    â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    Definition:
        Accruals = (Net Income - Operating Cash Flow) / Total Assets

        High accruals â†’ earnings are driven by accounting adjustments
        rather than actual cash. These firms tend to underperform.
        So we NEGATE: negative accruals (cash-driven earnings) â†’ positive signal.

    Formula:
        accruals_t = -(net_income_t - operating_cf_t) / total_assets_t

    Academic basis:
        Sloan (1996) â€” "Do Stock Prices Fully Reflect Information in
        Accruals and Cash Flows about Future Earnings?" The Accounting Review.
        One of the most replicated anomalies in the literature.

    Parameters:
        net_income   : pd.DataFrame â€” quarterly net income, same index/cols as prices
        operating_cf : pd.DataFrame â€” quarterly operating cash flow
        total_assets : pd.DataFrame â€” quarterly total assets

    Returns:
        pd.DataFrame â€” cross-sectionally z-scored accruals factor.

    Note:
        Fundamental data should be lagged by at least 2 months before use
        (10-Q filings arrive ~45 days after quarter end). Person A enforces
        this in the backtest infrastructure.
    """
    accruals_raw = (net_income - operating_cf) / total_assets.replace(0, np.nan)

    # Negate: low accruals (high cash earnings) â†’ high predicted return
    accruals_signal = -accruals_raw

    return zscore_cross_section(accruals_signal)


def factor_investment_growth(total_assets: pd.DataFrame,
                             lag_months: int = 12) -> pd.DataFrame:
    """
    Investment Growth Factor (Asset Growth)
    â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    Definition:
        Year-over-year percentage change in total assets, sign-flipped.
        Firms that aggressively grow their asset base tend to subsequently
        underperform (overinvestment / empire building hypothesis).

    Formula:
        inv_growth_t = -(total_assets_t / total_assets_{t-12} - 1)

    Academic basis:
        Cooper, Gulen & Schill (2008) â€” "Asset Growth and the Cross-Section
        of Stock Returns." Journal of Finance. Robust across markets.

    Parameters:
        total_assets : pd.DataFrame â€” monthly or quarterly total assets
        lag_months   : int â€” how many months back to compare (default 12 = 1 year)

    Returns:
        pd.DataFrame â€” cross-sectionally z-scored investment growth factor.
    """
    yoy_growth = total_assets.pct_change(lag_months)

    # Negate: high asset growth â†’ negative expected return
    inv_growth_signal = -yoy_growth

    return zscore_cross_section(inv_growth_signal)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# QUICK SMOKE TEST â€” run this block to verify functions work
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == "__main__":
    import pandas as pd, numpy as np

    np.random.seed(42)
    dates   = pd.date_range("2010-01-31", periods=60, freq="ME")
    tickers = [f"T{i}" for i in range(50)]

    # Fake prices
    prices = pd.DataFrame(
        100 * np.cumprod(1 + np.random.randn(60, 50) * 0.05, axis=0),
        index=dates, columns=tickers
    )

    # Fake fundamentals
    ni  = pd.DataFrame(np.random.randn(60, 50) * 1e8, index=dates, columns=tickers)
    ocf = pd.DataFrame(np.random.randn(60, 50) * 1e8, index=dates, columns=tickers)
    ta  = pd.DataFrame(np.abs(np.random.randn(60, 50)) * 1e9 + 1e9,
                       index=dates, columns=tickers)

    rev  = factor_short_term_reversal(prices)
    acc  = factor_accruals(ni, ocf, ta)
    invg = factor_investment_growth(ta)

    for name, f in [("short_term_reversal", rev),
                    ("accruals", acc),
                    ("investment_growth", invg)]:
        row_means = f.mean(axis=1).abs().mean()
        row_stds  = f.std(axis=1).mean()
        status    = "âœ“" if row_means < 0.05 and abs(row_stds - 1.0) < 0.1 else "âš "
        print(f"  {status}  {name:<25}  row_meanâ‰ˆ{row_means:.3f}  row_stdâ‰ˆ{row_stds:.3f}")
