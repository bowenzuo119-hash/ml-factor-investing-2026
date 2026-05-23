"""Phase 7: statistical robustness checks on the canonical Phase-3c portfolio.

Addresses the three caveats that legitimate criticism would raise about a
+0.59 Sharpe over 5 years of OOS data:

  1. The 5-year t-stat is only ~1.3 -- not statistically significant.
     -> Block bootstrap CIs (Framework section 8.3 explicit ask).
     -> Deflated Sharpe per Bailey & Lopez de Prado (2014), adjusting
        for the 5 model variants we tried.
     -> Re-slice metrics on the longer 2015-2024 OOS window for
        comparison.

  2. Some of the Sharpe is market-beta drift, not skill.
     -> Fama-French 3-factor regression (Mkt-RF, SMB, HML).
     -> Fama-French 5-factor regression (+ RMW, CMA).
     -> Report alpha with Newey-West HAC standard errors.

  3. Tiny IC + decent Sharpe is suspicious (could be sector tilts).
     -> Sector audit in Phase 6 already showed +34% concentration.
     -> Here: report alpha t-stat after FF -- if it survives, it is
        residual cross-sectional skill, not factor-replication.

Run with:
    .venv/bin/python -m notebooks.personb.07_statistical_robustness
"""
from __future__ import annotations

import io
import json
import pickle
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


RESULTS_DIR = (
    Path(__file__).resolve().parents[2] / "results" / "07_statistical_robustness"
)
PHASE_DIR = (
    Path(__file__).resolve().parents[2] / "results"
    / "15_canonical_2002"
)

TEST_START = pd.Timestamp("2019-01-01")
TEST_END = pd.Timestamp("2024-12-31")
LONG_OOS_START = pd.Timestamp("2015-01-01")  # full walk-forward range

# We tried 6 distinct model/feature configurations along the way:
# Phase 1, 1.5, 2, 3b, 3c, 8, 14, 15: 8 canonical-model trials.
# Used for deflation -- more trials = bigger penalty.
N_TRIALS = 8
BLOCK_BOOTSTRAP_BLOCK_SIZE = 6   # 6-month blocks
BLOCK_BOOTSTRAP_N_ITERS = 10_000
RANDOM_STATE = 42


# --------------------------------------------------------------------------
# 1. Block bootstrap on Sharpe
# --------------------------------------------------------------------------

def sharpe(rets: np.ndarray, periods_per_year: int = 12) -> float:
    """Annualised Sharpe of a return series."""
    r = rets[~np.isnan(rets)]
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))


def block_bootstrap_sharpe(
    rets: np.ndarray, *, block_size: int = 6,
    n_iters: int = 10_000, seed: int = 42,
) -> dict[str, float]:
    """Block-bootstrap CIs on the Sharpe ratio.

    Resample contiguous blocks of `block_size` months (with replacement)
    until we have a series of the original length, recompute Sharpe,
    repeat `n_iters` times. Block resampling preserves the autocorrelation
    structure of monthly returns; standard iid bootstrap would underestimate
    the CI width by assuming independence we don't have.
    """
    r = rets[~np.isnan(rets)]
    n = len(r)
    if n < block_size:
        return {"sharpe": sharpe(r), "ci_5": float("nan"),
                "ci_50": float("nan"), "ci_95": float("nan"),
                "p_le_zero": float("nan"), "n_obs": int(n)}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    sharpes = np.empty(n_iters)
    max_start = n - block_size + 1
    for i in range(n_iters):
        starts = rng.integers(0, max_start, size=n_blocks)
        sample = np.concatenate([r[s:s + block_size] for s in starts])[:n]
        if sample.std(ddof=1) > 0:
            sharpes[i] = sample.mean() / sample.std(ddof=1) * np.sqrt(12)
        else:
            sharpes[i] = 0.0
    return {
        "sharpe": sharpe(r),
        "ci_5": float(np.percentile(sharpes, 5)),
        "ci_50": float(np.percentile(sharpes, 50)),
        "ci_95": float(np.percentile(sharpes, 95)),
        "ci_mean": float(np.mean(sharpes)),
        "p_le_zero": float((sharpes <= 0).mean()),
        "n_obs": int(n),
    }


# --------------------------------------------------------------------------
# 2. Deflated Sharpe (Bailey & Lopez de Prado 2014)
# --------------------------------------------------------------------------

def deflated_sharpe(
    rets: np.ndarray, *, n_trials: int, sr_variance_across_trials: float,
) -> dict[str, float]:
    """Deflated Sharpe per Bailey & Lopez de Prado (2014).

    Adjusts observed Sharpe down by:
    * the maximum Sharpe expected from `n_trials` random configurations
      (Bonferroni-like correction for multiple testing), and
    * the higher-order moments (skewness, kurtosis) of the actual return
      series (heavy-tailed series gets a bigger haircut).

    The deflated Sharpe is reported as a probability that the true Sharpe
    is positive. DSR > 0.95 means significant at 5% after deflation.
    """
    r = rets[~np.isnan(rets)]
    T = len(r)
    if T < 4:
        return {"sharpe": sharpe(r), "DSR": float("nan"),
                "expected_max_sr": float("nan"),
                "skew": float("nan"), "ex_kurt": float("nan")}

    sr_obs = sharpe(r) / np.sqrt(12)  # non-annualised for DSR formula
    # Sample skewness and excess kurtosis
    centred = r - r.mean()
    sd = r.std(ddof=0)
    g1 = float(((centred / sd) ** 3).mean()) if sd > 0 else 0.0
    g2 = float(((centred / sd) ** 4).mean() - 3.0) if sd > 0 else 0.0

    # Expected maximum Sharpe from n_trials trials. Uses the standard
    # extreme-value approximation: E[max] = sqrt(V[SR]) * z_eff where
    # z_eff = (1 - gamma) Phi^-1(1 - 1/n) + gamma Phi^-1(1 - 1/(n*e)),
    # gamma = Euler-Mascheroni = 0.5772.
    if n_trials <= 1:
        sr_expected = 0.0
    else:
        gamma = 0.5772156649
        z_eff = (
            (1 - gamma) * norm.ppf(1 - 1.0 / n_trials)
            + gamma * norm.ppf(1 - 1.0 / (n_trials * np.e))
        )
        sr_expected = np.sqrt(max(sr_variance_across_trials, 1e-12)) * z_eff

    denom = np.sqrt(max(1 - g1 * sr_obs + (g2 / 4.0) * sr_obs ** 2, 1e-12))
    dsr_z = (sr_obs - sr_expected) * np.sqrt(T - 1) / denom
    DSR = float(norm.cdf(dsr_z))

    return {
        "sharpe": sharpe(r),
        "sharpe_non_annualised": float(sr_obs),
        "DSR": DSR,
        "DSR_z": float(dsr_z),
        "expected_max_sr_non_annualised": float(sr_expected),
        "skew": g1,
        "ex_kurt": g2,
        "n_obs": int(T),
    }


# --------------------------------------------------------------------------
# 3. Fama-French factor data from Ken French's website
# --------------------------------------------------------------------------

FF_3F_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
FF_5F_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"


def _fetch_ff_csv(url: str) -> bytes:
    """Download and unzip a Ken French research data CSV."""
    print(f"  fetching {url.split('/')[-1]}...")
    with urllib.request.urlopen(url, timeout=30) as resp:
        zip_bytes = resp.read()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        name = [n for n in zf.namelist() if n.endswith(".CSV")
                or n.endswith(".csv")][0]
        return zf.read(name)


def fetch_ff_monthly(*, five_factor: bool = False) -> pd.DataFrame:
    """Monthly Fama-French factor returns, indexed by month-end."""
    raw = _fetch_ff_csv(FF_5F_URL if five_factor else FF_3F_URL)
    text = raw.decode("latin-1")
    # The file format is documented but quirky: a preamble, then the
    # monthly block (header row + numeric rows), then a blank line, then
    # the annual block. Parse the monthly block only.
    lines = text.splitlines()
    # Find the header row: starts with comma and contains "Mkt-RF"
    header_idx = None
    for i, line in enumerate(lines):
        if "Mkt-RF" in line:
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError("Could not find Mkt-RF header row in FF CSV")
    # Numeric rows are those where the first field is a 6-digit YYYYMM
    rows = []
    for line in lines[header_idx + 1:]:
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0]:
            break
        if not (parts[0].isdigit() and len(parts[0]) == 6):
            break
        rows.append(parts)
    cols = ["date"] + [c.strip() for c in lines[header_idx].split(",")[1:]]
    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m") + pd.offsets.MonthEnd(0)
    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0  # % -> fraction
    df = df.set_index("date")
    return df


def regress_with_hac(y: np.ndarray, X: np.ndarray, *, lags: int = 6) -> dict:
    """OLS y = X beta + eps with Newey-West HAC standard errors.

    X must already include the constant column (or not, depending on caller).
    Returns dict with beta, se, t, p, r2.
    """
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta_hat = XtX_inv @ X.T @ y
    resid = y - X @ beta_hat
    # Newey-West "sandwich"
    S = np.zeros((k, k))
    for lag in range(0, min(lags, n - 1) + 1):
        gamma = np.zeros((k, k))
        for i in range(lag, n):
            v_t = X[i] * resid[i]
            v_l = X[i - lag] * resid[i - lag]
            if lag == 0:
                gamma += np.outer(v_t, v_t)
            else:
                gamma += np.outer(v_t, v_l) + np.outer(v_l, v_t)
        weight = 1.0 if lag == 0 else 1.0 - lag / (lags + 1)
        S += weight * gamma
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    t = beta_hat / np.where(se > 0, se, 1)
    from scipy.stats import t as t_dist
    p = 2.0 * (1.0 - t_dist.cdf(np.abs(t), df=n - k))
    ss_res = (resid ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "beta": beta_hat, "se": se, "t": t, "p": p, "r2": float(r2),
        "n": int(n), "k": int(k),
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("Phase 7: statistical robustness on the canonical Phase-3c portfolio")
    print("=" * 72)

    # Load canonical portfolio
    print("\n[1/4] Loading Phase 3c portfolios...")
    with open(PHASE_DIR / "per_model_results.pkl", "rb") as f:
        results = pickle.load(f)

    xgb_rets = results["XGBoost"].portfolio_returns.sort_index()
    print(f"  XGBoost portfolio returns: {len(xgb_rets)} months, "
          f"{xgb_rets.index.min().date()} -> {xgb_rets.index.max().date()}")

    test_rets = xgb_rets[(xgb_rets.index >= TEST_START)
                         & (xgb_rets.index <= TEST_END)]
    long_oos_rets = xgb_rets[(xgb_rets.index >= LONG_OOS_START)
                             & (xgb_rets.index <= TEST_END)]
    print(f"  test slice (2019-2024): {len(test_rets)} months")
    print(f"  long-OOS slice (2015-2024): {len(long_oos_rets)} months")

    # ============== A. SHARPE + BOOTSTRAP ============================
    print("\n[2/4] Block-bootstrap Sharpe CIs...")
    bb_test = block_bootstrap_sharpe(
        test_rets.to_numpy(), block_size=BLOCK_BOOTSTRAP_BLOCK_SIZE,
        n_iters=BLOCK_BOOTSTRAP_N_ITERS, seed=RANDOM_STATE,
    )
    bb_long = block_bootstrap_sharpe(
        long_oos_rets.to_numpy(), block_size=BLOCK_BOOTSTRAP_BLOCK_SIZE,
        n_iters=BLOCK_BOOTSTRAP_N_ITERS, seed=RANDOM_STATE,
    )
    print(f"\n  Test window 2019-2024 ({bb_test['n_obs']} months):")
    print(f"    Sharpe observed = {bb_test['sharpe']:+.4f}")
    print(f"    Block-bootstrap 5-95% CI = "
          f"[{bb_test['ci_5']:+.4f}, {bb_test['ci_95']:+.4f}]")
    print(f"    P(bootstrap Sharpe <= 0) = {bb_test['p_le_zero']:.4f}")
    print(f"\n  Long-OOS window 2015-2024 ({bb_long['n_obs']} months):")
    print(f"    Sharpe observed = {bb_long['sharpe']:+.4f}")
    print(f"    Block-bootstrap 5-95% CI = "
          f"[{bb_long['ci_5']:+.4f}, {bb_long['ci_95']:+.4f}]")
    print(f"    P(bootstrap Sharpe <= 0) = {bb_long['p_le_zero']:.4f}")

    # ============== B. DEFLATED SHARPE ===============================
    # Sharpe across all canonical trials run during model development.
    print("\n[3/4] Deflated Sharpe (Bailey-Lopez de Prado 2014)...")
    trial_sharpes = {
        "Phase 1":   -0.032,
        "Phase 1.5": +0.556,
        "Phase 2":   +0.432,
        "Phase 3b":  +0.526,
        "Phase 3c":  +0.589,
        # Phase 8 onward use the v0.3.0 engine (block-gated refit). Prior
        # phases were measured on the old refit-every-period engine and
        # haven't been re-run on v0.3.0; the DSR is robust to this slight
        # mixture.
        "Phase 8":   +0.942,
        "Phase 14":  +0.913,
        "Phase 15":  +1.011,
    }
    print(f"  Sharpe across {N_TRIALS} trials: "
          f"{list(trial_sharpes.values())}")
    # Per BLdP, V[SR] is the variance of SR estimates across trials,
    # NON-annualised. Annual SR -> monthly SR = annual / sqrt(12).
    trial_sr_monthly = np.array(list(trial_sharpes.values())) / np.sqrt(12)
    sr_variance_across_trials = float(trial_sr_monthly.var(ddof=1))
    print(f"  V[SR_monthly] across trials = {sr_variance_across_trials:.5f}")

    dsr_test = deflated_sharpe(
        test_rets.to_numpy(), n_trials=N_TRIALS,
        sr_variance_across_trials=sr_variance_across_trials,
    )
    dsr_long = deflated_sharpe(
        long_oos_rets.to_numpy(), n_trials=N_TRIALS,
        sr_variance_across_trials=sr_variance_across_trials,
    )
    print(f"\n  Test window 2019-2024:")
    print(f"    Sharpe = {dsr_test['sharpe']:+.4f}, "
          f"DSR = {dsr_test['DSR']:.4f}  "
          f"(P(true SR > expected max from {N_TRIALS} trials | observed))")
    print(f"    skew = {dsr_test['skew']:+.3f}, "
          f"excess kurtosis = {dsr_test['ex_kurt']:+.3f}")
    print(f"\n  Long-OOS window 2015-2024:")
    print(f"    Sharpe = {dsr_long['sharpe']:+.4f}, "
          f"DSR = {dsr_long['DSR']:.4f}")
    print(f"    skew = {dsr_long['skew']:+.3f}, "
          f"excess kurtosis = {dsr_long['ex_kurt']:+.3f}")

    # ============== C. FAMA-FRENCH FACTOR REGRESSION ===================
    print("\n[4/4] Fama-French factor regressions...")
    try:
        ff3 = fetch_ff_monthly(five_factor=False)
        ff5 = fetch_ff_monthly(five_factor=True)
    except Exception as e:
        print(f"  [warn] Ken French download failed: {e}; skipping FF block")
        ff3 = ff5 = None

    ff_summary: dict[str, dict] = {}
    if ff3 is not None and ff5 is not None:
        for window_name, rets_window in [
            ("test_only", test_rets),
            ("long_oos", long_oos_rets),
        ]:
            # Align by month-end period (handle CRSP-Dec-30 vs FF-Dec-31)
            port_df = pd.DataFrame({
                "y": rets_window.to_numpy(),
                "month": rets_window.index.to_period("M"),
            })

            for ff_name, ff_data in [("FF3", ff3), ("FF5", ff5)]:
                ff_df = ff_data.copy()
                ff_df["month"] = ff_df.index.to_period("M")
                joint = port_df.merge(ff_df, on="month").dropna()
                if len(joint) < 12:
                    continue

                # Excess return = portfolio - RF
                y_excess = (joint["y"] - joint["RF"]).to_numpy()
                if ff_name == "FF3":
                    factors = ["Mkt-RF", "SMB", "HML"]
                else:
                    factors = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
                # Note: column names from FF have minus signs; pandas keeps them
                X = np.column_stack(
                    [np.ones(len(joint))]
                    + [joint[f].to_numpy() for f in factors]
                )
                reg = regress_with_hac(y_excess, X, lags=6)
                names = ["alpha"] + factors
                row = {"window": window_name, "model": ff_name,
                       "n_months": reg["n"], "r2": reg["r2"]}
                for j, nm in enumerate(names):
                    row[f"{nm}_coef"] = float(reg["beta"][j])
                    row[f"{nm}_se"] = float(reg["se"][j])
                    row[f"{nm}_t"] = float(reg["t"][j])
                    row[f"{nm}_p"] = float(reg["p"][j])
                ff_summary[f"{window_name}_{ff_name}"] = row

        # Pretty-print FF results
        for key, row in ff_summary.items():
            print(f"\n  [{key}]  n = {row['n_months']}, R^2 = {row['r2']:.3f}")
            for nm in row:
                if nm.endswith("_coef"):
                    factor = nm[:-5]
                    coef = row[nm]
                    t = row[f"{factor}_t"]
                    p = row[f"{factor}_p"]
                    star = (
                        "***" if p < 0.01 else
                        "**" if p < 0.05 else
                        "*" if p < 0.10 else ""
                    )
                    label = factor
                    if factor == "alpha":
                        label = "alpha (annualised)"
                        coef_disp = f"{coef * 12 * 100:+.2f}%"
                    else:
                        coef_disp = f"{coef:+.3f}"
                    print(f"    {label:24s}  {coef_disp:>12s}  "
                          f"t={t:+.2f}  p={p:.4f}  {star}")

    # ============== Persist all results ==============================
    out = {
        "bootstrap_test": bb_test,
        "bootstrap_long_oos": bb_long,
        "deflated_test": dsr_test,
        "deflated_long_oos": dsr_long,
        "trial_sharpes": trial_sharpes,
        "n_trials": N_TRIALS,
        "fama_french": ff_summary,
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(out, f, indent=2, default=float)

    # ============== Plot: bootstrap distribution ==================
    rng = np.random.default_rng(RANDOM_STATE)
    n_blocks_test = int(np.ceil(len(test_rets) / BLOCK_BOOTSTRAP_BLOCK_SIZE))
    r_arr = test_rets.to_numpy()
    sharpes = np.empty(BLOCK_BOOTSTRAP_N_ITERS)
    max_start = len(r_arr) - BLOCK_BOOTSTRAP_BLOCK_SIZE + 1
    for i in range(BLOCK_BOOTSTRAP_N_ITERS):
        starts = rng.integers(0, max_start, size=n_blocks_test)
        sample = np.concatenate(
            [r_arr[s:s + BLOCK_BOOTSTRAP_BLOCK_SIZE] for s in starts]
        )[:len(r_arr)]
        if sample.std(ddof=1) > 0:
            sharpes[i] = sample.mean() / sample.std(ddof=1) * np.sqrt(12)
        else:
            sharpes[i] = 0.0

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(sharpes, bins=60, color="#3B82F6", edgecolor="black",
            alpha=0.85)
    ax.axvline(bb_test["sharpe"], color="#DC2626", lw=2,
               label=f"Observed = {bb_test['sharpe']:+.3f}")
    ax.axvline(bb_test["ci_5"], color="#10B981", lw=1.4, linestyle="--",
               label=f"5% CI = {bb_test['ci_5']:+.3f}")
    ax.axvline(bb_test["ci_95"], color="#10B981", lw=1.4, linestyle="--",
               label=f"95% CI = {bb_test['ci_95']:+.3f}")
    ax.axvline(0, color="black", lw=0.7)
    ax.set_title("Block-bootstrap distribution of test-window Sharpe\n"
                 "10,000 resamples, 6-month blocks, 2019-2024",
                 fontsize=11, weight="bold")
    ax.set_xlabel("Bootstrapped Sharpe ratio (annualised)")
    ax.set_ylabel("Frequency")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "bootstrap_distribution.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote summary.json + bootstrap_distribution.png to "
          f"{RESULTS_DIR.relative_to(Path.cwd())}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
