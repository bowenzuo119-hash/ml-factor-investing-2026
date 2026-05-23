"""Phase 5b: Realised net market beta of the canonical dollar-neutral portfolio.

The DOLLAR_VS_BETA_NEUTRAL.pdf report claims the canonical Phase-3c
portfolio has a "small but non-zero net beta, probably +0.2 to +0.4".
This script puts a number on it by:

  1. Loading Phase 3c's BacktestResult (per-rebalance weights + per-period
     portfolio returns) for each of Lasso / XGBoost / NN.
  2. Pulling the S&P 500 monthly index return as the market benchmark
     (yfinance ^GSPC, resampled to month-end).
  3. Regressing portfolio_returns on market_returns over the test window
     2019-2024:  r_p,t = alpha + beta * r_m,t + eps
     The slope coefficient is the realised net beta. Robust (HAC) standard
     errors so the t-stat respects autocorrelation.

Run with:
    .venv/bin/python -m notebooks.personb.05b_realised_net_beta
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


RESULTS_DIR = (
    Path(__file__).resolve().parents[2] / "results" / "05b_realised_net_beta"
)
PHASE_DIR = (
    Path(__file__).resolve().parents[2] / "results"
    / "14_official_canonical_k5"
)
PANEL_FILE = (
    Path(__file__).resolve().parents[2] / "data" / "processed"
    / "returns_spliced_2005_2024.parquet"
)

TEST_START = pd.Timestamp("2019-01-01")
TEST_END = pd.Timestamp("2024-12-31")


def fetch_spy_monthly() -> pd.Series:
    """Monthly S&P 500 total returns from yfinance ^GSPC, month-end aligned."""
    print("  fetching ^GSPC monthly via yfinance...")
    raw = yf.download("^GSPC", start="2018-11-01", end="2025-01-31",
                      interval="1d", auto_adjust=True, progress=False)
    # Resample to last trading day of each month, take adj close, pct_change
    monthly = (
        raw.assign(_month=raw.index.to_period("M"))
        .groupby("_month", group_keys=False)
        .tail(1)
        .drop(columns="_month")
    )
    px = monthly["Close"].squeeze()
    rets = px.pct_change().dropna()
    rets.name = "spy"
    return rets


def newey_west_se(x: np.ndarray, y: np.ndarray, lags: int = 6) -> dict:
    """OLS y on x (with intercept) + Newey-West HAC standard errors.

    Returns dict with alpha, beta, alpha_se, beta_se, t_beta, p_beta, r2.
    """
    n = len(x)
    X_mat = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X_mat.T @ X_mat)
    beta_hat = XtX_inv @ X_mat.T @ y
    resid = y - X_mat @ beta_hat

    # HAC variance: XtX_inv @ S @ XtX_inv, where S is the Newey-West kernel.
    S = np.zeros((2, 2))
    for k in range(0, min(lags, n - 1) + 1):
        if k == 0:
            g_k = (X_mat.T * resid) @ ((X_mat.T * resid).T)
            # That builds a 2x2 outer-sum; simpler form below
            inner = np.zeros((2, 2))
            for i in range(n):
                v = X_mat[i] * resid[i]
                inner += np.outer(v, v)
            S += inner
        else:
            inner = np.zeros((2, 2))
            for i in range(k, n):
                v_t = X_mat[i] * resid[i]
                v_l = X_mat[i - k] * resid[i - k]
                inner += np.outer(v_t, v_l) + np.outer(v_l, v_t)
            weight = 1.0 - k / (lags + 1)
            S += weight * inner

    cov = XtX_inv @ S @ XtX_inv
    ses = np.sqrt(np.diag(cov))
    from scipy.stats import t as t_dist
    t_beta = beta_hat[1] / ses[1] if ses[1] > 0 else 0.0
    p_beta = 2.0 * (1.0 - t_dist.cdf(abs(t_beta), df=n - 2))
    ss_res = (resid ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "alpha": float(beta_hat[0]),
        "beta": float(beta_hat[1]),
        "alpha_se": float(ses[0]),
        "beta_se": float(ses[1]),
        "t_beta": float(t_beta),
        "p_beta": float(p_beta),
        "r2": float(r2),
        "n": int(n),
    }


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("Phase 5b: realised net market beta of the canonical portfolio")
    print("=" * 72)

    # 1. Market benchmark series ----------------------------
    print("\n[1/3] Market benchmark (S&P 500 monthly return)...")
    spy = fetch_spy_monthly()
    spy_test = spy[(spy.index >= TEST_START) & (spy.index <= TEST_END)]
    print(f"  {len(spy_test)} months: {spy_test.index.min().date()} -> "
          f"{spy_test.index.max().date()}")

    # 2. Portfolio returns per model -----------------------
    print("\n[2/3] Loading Phase 3c portfolios...")
    with open(PHASE_DIR / "per_model_results.pkl", "rb") as f:
        results = pickle.load(f)
    print(f"  models: {list(results.keys())}")

    rows = []
    for model_name, res in results.items():
        pf_rets = res.portfolio_returns
        # Restrict to test window and align with SPY by month-period
        pf_rets_test = pf_rets[(pf_rets.index >= TEST_START)
                               & (pf_rets.index <= TEST_END)]

        # Align on month-period to handle CRSP-Dec-30 vs yfinance-Dec-31 etc.
        # Use a left join on (year, month).
        df = pd.DataFrame({
            "portfolio": pf_rets_test.to_numpy(),
            "month": pf_rets_test.index.to_period("M"),
        })
        spy_df = pd.DataFrame({
            "market": spy_test.to_numpy(),
            "month": spy_test.index.to_period("M"),
        })
        merged = df.merge(spy_df, on="month").dropna()

        if len(merged) < 12:
            print(f"  [warn] {model_name}: only {len(merged)} aligned months")
            continue

        x = merged["market"].to_numpy()
        y = merged["portfolio"].to_numpy()
        res_reg = newey_west_se(x, y, lags=6)
        annualised_alpha = res_reg["alpha"] * 12

        rows.append({
            "model": model_name,
            "n_months": res_reg["n"],
            "beta": res_reg["beta"],
            "beta_se": res_reg["beta_se"],
            "t_beta": res_reg["t_beta"],
            "p_beta": res_reg["p_beta"],
            "alpha_monthly": res_reg["alpha"],
            "alpha_annualised": annualised_alpha,
            "r2_market": res_reg["r2"],
        })

    summary_df = pd.DataFrame(rows).set_index("model")
    print("\n[3/3] Regression results (portfolio_t on market_t, "
          "HAC SE with 6 lags):\n")
    fmt = summary_df.copy()
    fmt["beta"] = fmt["beta"].map(lambda v: f"{v:+.3f}")
    fmt["beta_se"] = fmt["beta_se"].map(lambda v: f"{v:.3f}")
    fmt["t_beta"] = fmt["t_beta"].map(lambda v: f"{v:+.2f}")
    fmt["p_beta"] = fmt["p_beta"].map(lambda v: f"{v:.4f}")
    fmt["alpha_monthly"] = fmt["alpha_monthly"].map(lambda v: f"{v*100:+.3f}%")
    fmt["alpha_annualised"] = fmt["alpha_annualised"].map(lambda v: f"{v*100:+.2f}%")
    fmt["r2_market"] = fmt["r2_market"].map(lambda v: f"{v:.3f}")
    print(fmt.to_string())

    summary_df.to_parquet(RESULTS_DIR / "net_beta_summary.parquet")

    # Plot: scatter of portfolio vs market with regression lines per model
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, (model_name, res) in zip(axes, results.items()):
        pf_rets = res.portfolio_returns
        pf_rets_test = pf_rets[(pf_rets.index >= TEST_START)
                               & (pf_rets.index <= TEST_END)]
        df = pd.DataFrame({
            "portfolio": pf_rets_test.to_numpy(),
            "month": pf_rets_test.index.to_period("M"),
        })
        spy_df = pd.DataFrame({
            "market": spy_test.to_numpy(),
            "month": spy_test.index.to_period("M"),
        })
        merged = df.merge(spy_df, on="month").dropna()
        beta_row = summary_df.loc[model_name]

        ax.scatter(merged["market"] * 100, merged["portfolio"] * 100,
                   s=18, alpha=0.7, color="#1F3864")
        # Regression line
        xx = np.linspace(merged["market"].min(), merged["market"].max(), 50)
        yy = beta_row["alpha_monthly"] + beta_row["beta"] * xx
        ax.plot(xx * 100, yy * 100, color="#DC2626", lw=1.5,
                label=f"β = {beta_row['beta']:+.3f}")
        ax.axhline(0, color="black", lw=0.4)
        ax.axvline(0, color="black", lw=0.4)
        ax.set_title(f"{model_name}\n"
                     f"β={beta_row['beta']:+.3f}  α={beta_row['alpha_annualised']*100:+.2f}%/yr",
                     fontsize=10, weight="bold")
        ax.set_xlabel("S&P 500 monthly return (%)")
        if ax is axes[0]:
            ax.set_ylabel("Portfolio monthly return (%)")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9)
    fig.suptitle("Phase 3c portfolio returns vs S&P 500 — "
                 "2019-2024 test window\n"
                 "β > 0 = portfolio is net long the market",
                 fontsize=11, weight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "net_beta_scatter.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote {RESULTS_DIR.name}/net_beta_summary.parquet, "
          f"net_beta_scatter.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
