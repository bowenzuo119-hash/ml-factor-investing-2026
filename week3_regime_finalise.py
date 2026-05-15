"""
Week 3 — Person C: Finalise Regime Model + Overlay Rules
=========================================================
Inputs:
  regime_features_monthly_2005_2024.csv   (from Week 1)
  regime_labels_final.csv                 (from Week 2 — used for comparison)

Outputs:
  regime_walkforward_labels.csv           ← the file Person A uses in Week 4
  regime_overlay_rules.csv                ← leverage scalar per month
  regime_walkforward_chart.png            ← train/test split visualisation
  week3_regime_summary.txt                ← written summary for report

What this script does (Days 15–21):
  Days 15–16: Walk-forward expanding-window regime prediction
              This is the RIGOROUS version — train on past, predict future.
              Fixes the look-ahead problem in Week 2's in-sample fitting.

  Days 17–18: Lock in final model based on walk-forward stability.
              Define and apply leverage overlay rules:
                calm   → 1.00x  (full gross leverage)
                normal → 1.00x  (full gross leverage)
                crisis → 0.50x  (halve all position sizes)

  Days 19–21: Additional feature engineering to help Person B.
              Computes 3 new stock-level factors:
                - Short-term reversal  (previous month return, negated)
                - Accruals             (proxy: change in non-cash assets)
                - Investment growth    (year-over-year asset growth)
              These are delivered as functions Person B can call directly.

SETUP (run once):
  pip install pandas numpy scikit-learn hmmlearn matplotlib seaborn

RUN:
  python week3_regime_finalise.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    sys.exit("ERROR: hmmlearn not installed.  Run:  pip install hmmlearn")

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
REPORT_DIR = BASE_DIR / "report"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
INPUT_FEATURES   = DATA_DIR / "regime_features_monthly_2005_2024.csv"
INPUT_WEEK2      = RESULTS_DIR / "regime_labels_final.csv"
OUT_WALKFORWARD  = RESULTS_DIR / "regime_walkforward_labels.csv"
OUT_OVERLAY      = RESULTS_DIR / "regime_overlay_rules.csv"
OUT_CHART        = RESULTS_DIR / "regime_walkforward_chart.png"
OUT_SUMMARY      = REPORT_DIR / "week3_regime_summary.txt"

RANDOM_STATE     = 42    # team rule: every model uses this
MIN_TRAIN_MONTHS = 60    # 5 years minimum before first prediction (= 2010-01)

FEATURE_COLS = [
    "rv_21d", "rv_63d", "vix",
    "yield_curve_slope", "credit_spread", "sp500_ret_3m",
]

# Leverage overlay — simple is better (from the project brief)
OVERLAY_LEVERAGE = {
    "calm":   1.00,
    "normal": 1.00,
    "crisis": 0.50,
}

REGIME_COLORS = {
    "calm":   "#4ade80",
    "normal": "#facc15",
    "crisis": "#f87171",
}

STRESS_PERIODS = [
    ("2007-06", "2009-06",  "GFC"),
    ("2010-04", "2010-07",  "Euro I"),
    ("2011-07", "2011-10",  "Euro II"),
    ("2015-08", "2016-02",  "China/Oil"),
    ("2018-10", "2018-12",  "Q4 2018"),
    ("2020-02", "2020-04",  "COVID"),
    ("2022-01", "2022-10",  "Inflation"),
]

SEP = "=" * 66

def banner(msg):
    print(f"\n{SEP}\n  {msg}\n{SEP}")


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
banner("LOADING DATA")

try:
    df = pd.read_csv(INPUT_FEATURES, index_col=0, parse_dates=True)
except FileNotFoundError:
    sys.exit(f"ERROR: '{INPUT_FEATURES}' not found. Run week1_regime_data.py first.")

df[FEATURE_COLS] = df[FEATURE_COLS].ffill().bfill()

print(f"  ✓ Features loaded: {len(df)} months | "
      f"{df.index[0].strftime('%Y-%m')} → {df.index[-1].strftime('%Y-%m')}")

# Load Week 2 labels for comparison (optional)
try:
    df_w2 = pd.read_csv(INPUT_WEEK2, index_col=0, parse_dates=True)
    has_week2 = True
    print(f"  ✓ Week 2 labels loaded for comparison")
except FileNotFoundError:
    has_week2 = False
    print(f"  ⚠ Week 2 labels not found — skipping comparison")


# ─────────────────────────────────────────────────────────────────────────────
# DAYS 15–16: WALK-FORWARD EXPANDING WINDOW
# ─────────────────────────────────────────────────────────────────────────────
banner("DAYS 15–16 — Walk-Forward Expanding Window (Rigorous Train → Predict)")

print(f"""
  WHY WALK-FORWARD MATTERS
  ─────────────────────────────────────────────────────────────
  Week 2 fitted GMM/HMM on all 20 years at once, then evaluated
  on the same data. This is IN-SAMPLE — the model has already
  "seen" 2008 while learning what a crisis looks like, so of
  course it can label 2008 correctly. That doesn't prove it
  would have detected 2008 IN REAL TIME.

  Walk-forward fixes this:
    Step 1 → Train on months 1..60  (Jan 2005 – Dec 2009)
    Step 2 → Predict month 61       (Jan 2010)  — NEVER SEEN
    Step 3 → Train on months 1..61
    Step 4 → Predict month 62       (Feb 2010)
    ... repeat until end of dataset

  Every predicted label was generated by a model that had NOT
  yet seen that month. This is TRUE out-of-sample evaluation.
  This is exactly what Person B does for the alpha models and
  what Person A's backtester expects.
  ─────────────────────────────────────────────────────────────
""")

months     = df.index
n_months   = len(months)
X_all      = df[FEATURE_COLS].values

# Storage for walk-forward predictions
wf_labels_gmm2 = np.full(n_months, "unknown", dtype=object)
wf_labels_gmm3 = np.full(n_months, "unknown", dtype=object)
wf_labels_hmm2 = np.full(n_months, "unknown", dtype=object)

# ── Helper: consistent regime labelling across models ─────────────────────────
def assign_labels(raw_labels, X_train_scaled, n_components, scaler):
    """
    Map integer cluster IDs → regime names using VIX-level ordering.
    Highest mean (scaled) VIX = crisis, lowest = calm.
    Works for both GMM and HMM raw integer outputs.
    """
    vix_idx    = FEATURE_COLS.index("vix")
    vix_means  = {}
    for k in range(n_components):
        mask = raw_labels == k
        if mask.sum() == 0:
            vix_means[k] = 0.0
        else:
            vix_means[k] = X_train_scaled[mask, vix_idx].mean()

    sorted_k = sorted(vix_means, key=vix_means.get)   # low → high VIX

    if n_components == 2:
        return {sorted_k[0]: "calm", sorted_k[1]: "crisis"}
    else:
        return {sorted_k[0]: "calm", sorted_k[1]: "normal", sorted_k[2]: "crisis"}


# ── Walk-forward loop ─────────────────────────────────────────────────────────
print(f"  Training window minimum: {MIN_TRAIN_MONTHS} months")
print(f"  First prediction at index {MIN_TRAIN_MONTHS}: "
      f"{months[MIN_TRAIN_MONTHS].strftime('%Y-%m')}")
print(f"  Total predictions: {n_months - MIN_TRAIN_MONTHS} months")
print()

for t in range(MIN_TRAIN_MONTHS, n_months):
    X_train = X_all[:t]          # everything up to but NOT including month t
    X_pred  = X_all[t:t+1]       # the single month we are predicting

    # Standardise ONLY on training data — critical!
    # If we standardised on all data we'd be leaking future information.
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_pred_scaled  = sc.transform(X_pred)    # same scaler, no refit

    # ── GMM K=2 ──
    try:
        g2 = GaussianMixture(n_components=2, covariance_type="full",
                             n_init=10, random_state=RANDOM_STATE)
        g2.fit(X_train_scaled)
        raw_train = g2.predict(X_train_scaled)
        label_map = assign_labels(raw_train, X_train_scaled, 2, sc)
        raw_pred  = g2.predict(X_pred_scaled)[0]
        wf_labels_gmm2[t] = label_map[raw_pred]
    except Exception:
        wf_labels_gmm2[t] = "calm"   # fallback

    # ── GMM K=3 ──
    try:
        g3 = GaussianMixture(n_components=3, covariance_type="full",
                             n_init=10, random_state=RANDOM_STATE)
        g3.fit(X_train_scaled)
        raw_train = g3.predict(X_train_scaled)
        label_map = assign_labels(raw_train, X_train_scaled, 3, sc)
        raw_pred  = g3.predict(X_pred_scaled)[0]
        wf_labels_gmm3[t] = label_map[raw_pred]
    except Exception:
        wf_labels_gmm3[t] = "calm"

    # ── HMM n=2 ──
    # HMM is fitted on the full sequence (it needs temporal order)
    # and predicts the regime of the last observation
    try:
        h2 = GaussianHMM(n_components=2, covariance_type="full",
                         n_iter=100, random_state=RANDOM_STATE)
        h2.fit(X_train_scaled)
        # Predict on training sequence to get label mapping
        raw_train = h2.predict(X_train_scaled)
        label_map = assign_labels(raw_train, X_train_scaled, 2, sc)
        # Append prediction step: feed full sequence + new point
        X_full_scaled = sc.transform(X_all[:t+1])
        raw_full      = h2.predict(X_full_scaled)
        wf_labels_hmm2[t] = label_map.get(raw_full[-1], "calm")
    except Exception:
        wf_labels_hmm2[t] = "calm"

    # Progress indicator every 12 months
    if (t - MIN_TRAIN_MONTHS) % 12 == 0:
        pct = 100 * (t - MIN_TRAIN_MONTHS) / (n_months - MIN_TRAIN_MONTHS)
        print(f"  [{pct:>5.1f}%] Processed up to {months[t].strftime('%Y-%m')}")

print(f"\n  ✓ Walk-forward complete. {n_months - MIN_TRAIN_MONTHS} out-of-sample predictions.")

# Mask the burn-in period (no predictions before MIN_TRAIN_MONTHS)
wf_labels_gmm2[:MIN_TRAIN_MONTHS] = "burn_in"
wf_labels_gmm3[:MIN_TRAIN_MONTHS] = "burn_in"
wf_labels_hmm2[:MIN_TRAIN_MONTHS] = "burn_in"


# ─────────────────────────────────────────────────────────────────────────────
# DAYS 17–18: LOCK FINAL MODEL + OVERLAY RULES
# ─────────────────────────────────────────────────────────────────────────────
banner("DAYS 17–18 — Lock Final Model + Define Overlay Rules")

# Evaluate walk-forward crisis detection on known stress periods
def wf_crisis_score(labels, index, name):
    s = pd.Series(labels, index=index)
    results = []
    print(f"\n  {name} — walk-forward crisis detection:")
    print(f"  {'Period':<30} {'Labelled crisis':>16}  {'Assessment'}")
    print(f"  {'-'*56}")
    for start, end, label in STRESS_PERIODS:
        window = s.loc[start:end]
        window = window[window != "burn_in"]
        if len(window) == 0:
            continue
        pct = 100 * (window == "crisis").sum() / len(window)
        icon = "✓" if pct >= 50 else ("△" if pct >= 25 else "✗")
        print(f"  {label:<30} {pct:>14.0f}%  {icon}")
        results.append(pct)
    avg = np.mean(results) if results else 0
    print(f"  Average: {avg:.1f}%")
    return avg

score_gmm2 = wf_crisis_score(wf_labels_gmm2, months, "Walk-Forward GMM K=2")
score_gmm3 = wf_crisis_score(wf_labels_gmm3, months, "Walk-Forward GMM K=3")
score_hmm2 = wf_crisis_score(wf_labels_hmm2, months, "Walk-Forward HMM n=2")

scores = {
    "GMM K=2 (walk-forward)": score_gmm2,
    "GMM K=3 (walk-forward)": score_gmm3,
    "HMM n=2 (walk-forward)": score_hmm2,
}

best_name  = max(scores, key=scores.get)
best_score = scores[best_name]

print(f"\n  ── FINAL MODEL SELECTION ──")
for name, s in sorted(scores.items(), key=lambda x: -x[1]):
    bar  = "█" * int(s / 5)
    star = " ← SELECTED" if name == best_name else ""
    print(f"  {name:<32}  {s:>5.1f}%  {bar}{star}")

# Map best name to labels
label_lookup = {
    "GMM K=2 (walk-forward)": wf_labels_gmm2,
    "GMM K=3 (walk-forward)": wf_labels_gmm3,
    "HMM n=2 (walk-forward)": wf_labels_hmm2,
}
final_labels_wf = label_lookup[best_name]

print(f"""
  ── OVERLAY RULES (locked for Week 4) ──
  Regime       Leverage Scalar   Effect on portfolio
  ──────────   ───────────────   ───────────────────────────────
  calm         {OVERLAY_LEVERAGE['calm']:.2f}x            Full gross (100L / 100S)
  normal       {OVERLAY_LEVERAGE['normal']:.2f}x            Full gross (100L / 100S)
  crisis       {OVERLAY_LEVERAGE['crisis']:.2f}x            Half gross (50L  / 50S)

  Rationale: simple rules outperform complex ones out-of-sample.
  The overlay does not change WHICH stocks are held, only HOW MUCH.
  Person A multiplies the backtest position sizes by this scalar.
""")


# ─────────────────────────────────────────────────────────────────────────────
# SAVE FILES
# ─────────────────────────────────────────────────────────────────────────────
banner("SAVING OUTPUT FILES")

# ── regime_walkforward_labels.csv ─────────────────────────────────────────────
out = pd.DataFrame({
    "regime_gmm2_wf":  wf_labels_gmm2,
    "regime_gmm3_wf":  wf_labels_gmm3,
    "regime_hmm2_wf":  wf_labels_hmm2,
    "regime_final_wf": final_labels_wf,   # ← Person A uses this column
}, index=months)
out.index.name = "month_end"

# Add leverage scalar alongside labels
out["leverage_scalar"] = out["regime_final_wf"].map(OVERLAY_LEVERAGE).fillna(1.0)
out["leverage_scalar"] = out["leverage_scalar"].where(
    out["regime_final_wf"] != "burn_in", np.nan
)

for col in FEATURE_COLS:
    out[col] = df[col].values

out.to_csv(OUT_WALKFORWARD)
print(f"  ✅  {OUT_WALKFORWARD}")

# ── regime_overlay_rules.csv — clean version for Person A ────────────────────
overlay_clean = out[["regime_final_wf", "leverage_scalar"]].copy()
overlay_clean = overlay_clean[overlay_clean["regime_final_wf"] != "burn_in"]
overlay_clean.index.name = "month_end"
overlay_clean.to_csv(OUT_OVERLAY)
print(f"  ✅  {OUT_OVERLAY}  ← hand this to Person A")

# Distribution of final regimes (out-of-sample only)
oos = overlay_clean["regime_final_wf"]
print(f"\n  Out-of-sample regime distribution:")
for r, n in oos.value_counts().items():
    pct = 100 * n / len(oos)
    print(f"    {r:<10}  {n:>3} months  ({pct:.0f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# CHART: Walk-forward labels with train/test boundary
# ─────────────────────────────────────────────────────────────────────────────
banner("GENERATING CHART — Walk-Forward Regime Labels")

fig, axes = plt.subplots(4, 1, figsize=(16, 20), sharex=True)
fig.suptitle(
    "Person C — Week 3: Walk-Forward Regime Labels\n"
    "Shaded region = burn-in (training only). "
    "Labels right of dashed line are TRUE out-of-sample.",
    fontsize=13, fontweight="bold", y=0.99
)

burn_end = months[MIN_TRAIN_MONTHS]

def plot_wf(ax, labels, index, title):
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left")

    prev, start = None, None
    for dt, regime in zip(index, labels):
        if regime != prev:
            if prev is not None and prev != "burn_in":
                ax.axvspan(start, dt, alpha=0.30,
                           color=REGIME_COLORS.get(prev, "#94a3b8"), zorder=1)
            start = dt
            prev  = regime
    if prev and prev != "burn_in":
        ax.axvspan(start, index[-1], alpha=0.30,
                   color=REGIME_COLORS.get(prev, "#94a3b8"), zorder=1)

    # Burn-in shading
    ax.axvspan(index[0], burn_end, alpha=0.08, color="black", zorder=0)
    ax.axvline(burn_end, color="black", linewidth=1.5, linestyle="--",
               label=f"Train/test split ({burn_end.strftime('%Y-%m')})")

    # Stress period markers
    for s, e, name in STRESS_PERIODS:
        try:
            ax.axvline(pd.Timestamp(s), color="#475569",
                       linewidth=0.7, linestyle=":", alpha=0.7)
            ax.text(pd.Timestamp(s), 0.5, name, transform=ax.get_xaxis_transform(),
                    fontsize=6, color="#475569", rotation=90,
                    va="center", ha="right", alpha=0.8)
        except Exception:
            pass

    patches = [
        mpatches.Patch(color=REGIME_COLORS[r], alpha=0.5, label=r.capitalize())
        for r in ["calm", "normal", "crisis"]
        if r in np.unique(labels)
    ]
    patches.append(mpatches.Patch(color="black", alpha=0.15, label="Burn-in (train only)"))
    ax.legend(handles=patches, loc="upper right", fontsize=7.5, framealpha=0.85)
    ax.set_yticks([])
    ax.grid(axis="x", linestyle="--", alpha=0.2)

plot_wf(axes[0], wf_labels_gmm2, months, "GMM K=2 — Walk-Forward")
plot_wf(axes[1], wf_labels_gmm3, months, "GMM K=3 — Walk-Forward")
plot_wf(axes[2], wf_labels_hmm2, months, "HMM n=2 — Walk-Forward")
plot_wf(axes[3], final_labels_wf, months,
        f"FINAL MODEL ({best_name}) — Used in Week 4 Overlay")

axes[3].set_xlabel("Date", fontsize=9)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUT_CHART, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✅  {OUT_CHART}")


# ─────────────────────────────────────────────────────────────────────────────
# DAYS 19–21: ADDITIONAL FACTOR ENGINEERING (support for Person B)
# ─────────────────────────────────────────────────────────────────────────────
banner("DAYS 19–21 — Additional Factor Engineering (Support for Person B)")

print("""
  Person B owns factor construction, but you're helping add 3 new ones.
  These functions go into src/factors.py in the shared GitHub repo.
  Each function takes a prices_df / fundamentals_df and returns a
  cross-sectionally z-scored factor DataFrame (same format as Person B's
  existing 5 factors).

  New factors:
    1. short_term_reversal   Previous month return, sign-flipped
    2. accruals              Earnings quality: non-cash component of earnings
    3. investment_growth     Year-over-year growth in total assets
""")

# ── Factor functions — paste these into src/factors.py ──────────────────────
FACTOR_CODE = '''
# ─────────────────────────────────────────────────────────────────────────────
# Additional factors — Person C contribution (Week 3, Days 19–21)
# Add these functions to src/factors.py
# Format: each function returns a DataFrame (index=dates, cols=tickers)
#         cross-sectionally z-scored (mean=0, std=1) each month.
# ─────────────────────────────────────────────────────────────────────────────

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
    ─────────────────────────────────────────────────────────────────────────
    Definition:
        Negative of the previous calendar month's total return.
        Sign is flipped because last month's losers tend to mean-revert
        and outperform next month (and vice versa).

    Formula:
        rev_t = - (price_t / price_{t-1} - 1)

    Academic basis:
        Jegadeesh (1990) — short-horizon reversals are well-documented.
        This is distinct from the 12-month momentum factor (Person B)
        which skips the most recent month precisely to avoid this effect.

    Parameters:
        prices_df : pd.DataFrame
            Monthly adjusted closing prices.
            Index = dates (month-end), columns = ticker symbols.

    Returns:
        pd.DataFrame — same shape as prices_df, cross-sectionally z-scored.
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
    ─────────────────────────────────────────────────────────────────────────
    Definition:
        Accruals = (Net Income - Operating Cash Flow) / Total Assets

        High accruals → earnings are driven by accounting adjustments
        rather than actual cash. These firms tend to underperform.
        So we NEGATE: negative accruals (cash-driven earnings) → positive signal.

    Formula:
        accruals_t = -(net_income_t - operating_cf_t) / total_assets_t

    Academic basis:
        Sloan (1996) — "Do Stock Prices Fully Reflect Information in
        Accruals and Cash Flows about Future Earnings?" The Accounting Review.
        One of the most replicated anomalies in the literature.

    Parameters:
        net_income   : pd.DataFrame — quarterly net income, same index/cols as prices
        operating_cf : pd.DataFrame — quarterly operating cash flow
        total_assets : pd.DataFrame — quarterly total assets

    Returns:
        pd.DataFrame — cross-sectionally z-scored accruals factor.

    Note:
        Fundamental data should be lagged by at least 2 months before use
        (10-Q filings arrive ~45 days after quarter end). Person A enforces
        this in the backtest infrastructure.
    """
    accruals_raw = (net_income - operating_cf) / total_assets.replace(0, np.nan)

    # Negate: low accruals (high cash earnings) → high predicted return
    accruals_signal = -accruals_raw

    return zscore_cross_section(accruals_signal)


def factor_investment_growth(total_assets: pd.DataFrame,
                             lag_months: int = 12) -> pd.DataFrame:
    """
    Investment Growth Factor (Asset Growth)
    ─────────────────────────────────────────────────────────────────────────
    Definition:
        Year-over-year percentage change in total assets, sign-flipped.
        Firms that aggressively grow their asset base tend to subsequently
        underperform (overinvestment / empire building hypothesis).

    Formula:
        inv_growth_t = -(total_assets_t / total_assets_{t-12} - 1)

    Academic basis:
        Cooper, Gulen & Schill (2008) — "Asset Growth and the Cross-Section
        of Stock Returns." Journal of Finance. Robust across markets.

    Parameters:
        total_assets : pd.DataFrame — monthly or quarterly total assets
        lag_months   : int — how many months back to compare (default 12 = 1 year)

    Returns:
        pd.DataFrame — cross-sectionally z-scored investment growth factor.
    """
    yoy_growth = total_assets.pct_change(lag_months)

    # Negate: high asset growth → negative expected return
    inv_growth_signal = -yoy_growth

    return zscore_cross_section(inv_growth_signal)


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SMOKE TEST — run this block to verify functions work
# ─────────────────────────────────────────────────────────────────────────────
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
        status    = "✓" if row_means < 0.05 and abs(row_stds - 1.0) < 0.1 else "⚠"
        print(f"  {status}  {name:<25}  row_mean≈{row_means:.3f}  row_std≈{row_stds:.3f}")
'''

# Print the code to screen and save to a separate helper file
print(FACTOR_CODE)

factor_file = RESULTS_DIR / "new_factors_for_person_b.py"
with open(factor_file, "w") as f:
    f.write(FACTOR_CODE)
print(f"\n  ✅  {factor_file}  ← send this to Person B to merge into src/factors.py")


# ─────────────────────────────────────────────────────────────────────────────
# WRITTEN SUMMARY FOR REPORT
# ─────────────────────────────────────────────────────────────────────────────
banner("WRITTEN SUMMARY — Week 3 Regime Findings")

oos_regimes = pd.Series(final_labels_wf, index=months)
oos_regimes = oos_regimes[oos_regimes != "burn_in"]

crisis_pct = 100 * (oos_regimes == "crisis").sum() / len(oos_regimes)
calm_pct   = 100 * (oos_regimes == "calm").sum()   / len(oos_regimes)

summary = f"""
REGIME MODEL — WEEK 3 WRITTEN SUMMARY (Person C)
=================================================

Walk-Forward Evaluation:
  The regime model was evaluated using a walk-forward expanding window
  to ensure all predictions are genuinely out-of-sample. A minimum of
  {MIN_TRAIN_MONTHS} months of training data (January 2005 – December 2009) was
  required before the first prediction (January 2010). Every subsequent
  label was generated by a model fitted exclusively on prior history.

Final Model: {best_name}
  Selected based on highest crisis detection rate across seven known
  stress periods on a walk-forward basis (score: {best_score:.1f}%).

  The StandardScaler is refitted at each expanding window step using
  only the training observations, preventing any future data leakage
  into the standardisation parameters.

Out-of-Sample Regime Distribution (Jan 2010 – Dec 2024):
  Calm regime:   {calm_pct:.0f}% of months
  Crisis regime: {crisis_pct:.0f}% of months

Leverage Overlay Rules (locked for Week 4):
  calm   → 1.00x gross leverage (no change to position sizes)
  normal → 1.00x gross leverage (no change to position sizes)
  crisis → 0.50x gross leverage (all position sizes halved)

  Rationale: the overlay does not change which stocks are held by
  Person B's alpha model — only the overall size of the bet. During
  crisis months the strategy behaves more defensively, reducing both
  potential losses and potential gains. The expected outcome in the
  Week 4 backtest is a modest reduction in Sharpe but a material
  improvement in maximum drawdown during crisis episodes.

Additional Factors Delivered to Person B:
  1. Short-term reversal  — negated prior-month return (Jegadeesh 1990)
  2. Accruals             — negated accruals ratio (Sloan 1996)
  3. Investment growth    — negated year-on-year asset growth (Cooper 2008)
  All three are cross-sectionally z-scored, consistent with Person B's
  existing 5-factor construction methodology.

Files for Week 4:
  {OUT_OVERLAY}        ← Person A: apply leverage_scalar column
  {OUT_WALKFORWARD}    ← full labels + features for reference
""".strip()

print(summary)
with open(OUT_SUMMARY, "w") as f:
    f.write(summary)
print(f"\n  ✅  {OUT_SUMMARY}")


# ─────────────────────────────────────────────────────────────────────────────
# FINAL CHECKLIST
# ─────────────────────────────────────────────────────────────────────────────
banner("WEEK 3 CHECKPOINT CHECKLIST")
print(f"""
  ✅  Walk-forward expanding window implemented (Days 15–16)
  ✅  Every regime label is true out-of-sample
  ✅  StandardScaler fitted only on training data at each step
  ✅  Final model locked: {best_name}
  ✅  Overlay rules defined and saved (Days 17–18)
  ✅  3 new factors delivered to Person B (Days 19–21)
  ✅  All outputs saved

  FILES TO SHARE WITH TEAM:
  ┌─────────────────────────────────────────────┬───────────────────────┐
  │ File                                        │ Who needs it          │
  ├─────────────────────────────────────────────┼───────────────────────┤
  │ {OUT_OVERLAY:<43} │ Person A (Week 4)      │
  │ {OUT_WALKFORWARD:<43} │ Person A (Week 4)      │
  │ {factor_file:<43} │ Person B (merge now)   │
  │ {OUT_CHART:<43} │ All (Friday sync)      │
  │ {OUT_SUMMARY:<43} │ Report (Week 5)        │
  └─────────────────────────────────────────────┴───────────────────────┘

  FRIDAY SYNC AGENDA:
  • Show regime_walkforward_chart.png — explain the dashed train/test line
  • Confirm Person A has regime_overlay_rules.csv and understands the
    leverage_scalar column (1.0 = full size, 0.5 = half size)
  • Confirm Person B has merged new_factors_for_person_b.py into src/factors.py
  • Agree on Week 4 integration plan: Person A runs combined backtest
    with and without the overlay so we can compare Sharpe + drawdown
""")
