"""27b_k_sweep_plateau.py - zoom into k=10..20 with bootstrap CIs + FF5 alpha.

Phase 27 dense sweep showed a flat plateau between k=10 and k=20 with full-OOS
Sharpe within +-0.02 of peak (k=16 +1.174 vs k=20 +1.153). The differences look
real but are likely within sampling noise. This script answers definitively:

  * Block-bootstrap Sharpe CI at each k in [10, 20]
  * FF5 alpha + t-stat at each k in [10, 20]
  * Zoomed plot showing the plateau with error bars

If the bootstrap CIs at k=10..20 overlap, the "best k" call is in the noise
floor and the k=20 canonical lock is empirically justified.

Run with:
    .venv/bin/python -m notebooks.personb.27b_k_sweep_plateau
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import sharpe_ratio, annualised_return, max_drawdown
from notebooks.persona.verify_phase23_headline import fetch_ff5, nw_ols


# --- Inlined helpers (duplicated from notebooks/personb/27_k_sweep_dense.py;
#     can't `import` because the module name starts with a digit.) ---

def build_weights_for_k(
    preds: pd.Series,
    sector_map: dict[str, str],
    eligible_at_date: dict[pd.Timestamp, set[str]],
    k: int,
) -> pd.DataFrame:
    weights_records = []
    dates = sorted(preds.index.get_level_values("date").unique())
    for t in dates:
        scores_t = preds.loc[t].dropna()
        elig = eligible_at_date.get(t, set())
        if elig:
            scores_t = scores_t[scores_t.index.isin(elig)]
        if scores_t.empty:
            continue
        long_list, short_list = [], []
        sectors = pd.Series(
            {tk: sector_map.get(tk, "UNKNOWN") for tk in scores_t.index},
            name="sector",
        )
        for _sec, grp in scores_t.groupby(sectors):
            ranked = grp.sort_values(ascending=False)
            k_eff = min(int(k), len(ranked) // 2)
            if k_eff < 1:
                continue
            long_list.extend(ranked.head(k_eff).index.tolist())
            short_list.extend(ranked.tail(k_eff).index.tolist())
        w = pd.Series(0.0, index=scores_t.index)
        if long_list:
            w.loc[long_list] = 1.0 / len(long_list)
        if short_list:
            w.loc[short_list] = -1.0 / len(short_list)
        weights_records.append((t, w))
    all_tickers = sorted({tk for _, w in weights_records for tk in w.index})
    weights = pd.DataFrame(0.0, index=[t for t, _ in weights_records], columns=all_tickers)
    for t, w in weights_records:
        weights.loc[t, w.index] = w.values
    return weights


def portfolio_returns(
    weights: pd.DataFrame, next_returns: pd.DataFrame, cost_rate: float
) -> pd.Series:
    rets = pd.Series(index=weights.index, dtype=float)
    prev_w = pd.Series(0.0, index=weights.columns)
    for t in weights.index:
        if t not in next_returns.index:
            continue
        wt = weights.loc[t]
        rt = next_returns.loc[t].reindex(wt.index)
        valid = rt.dropna().index
        gross = float((wt.loc[valid] * rt.loc[valid]).sum())
        turnover = float((wt - prev_w.reindex(wt.index, fill_value=0.0)).abs().sum())
        rets.loc[t] = gross - cost_rate * turnover
        prev_w = wt
    return rets.dropna()


ROOT = Path(__file__).resolve().parents[2]
PREDS = ROOT / "results" / "24_canonical_with_chmom" / "predictions.parquet"
RETURNS_FILE = ROOT / "data" / "processed" / "returns_broad_sharadar_2002_2024.parquet"
FEATURES_FILE = ROOT / "data" / "processed" / "features_broad_sharadar_with_chmom_maxret.parquet"
OUT_DIR = ROOT / "results" / "27b_k_sweep_plateau"

K_GRID = list(range(10, 21))  # 10, 11, ..., 20
BLOCK_SIZE = 6
N_BOOT = 2000
COST_BPS = 10.0
SEED = 42
LONG_START = pd.Timestamp("2015-01-01")
TEST_START = pd.Timestamp("2019-01-01")


def block_bootstrap_sharpe(
    rets: np.ndarray, block_size: int, n_iters: int, seed: int
) -> dict:
    rng = np.random.default_rng(seed)
    r = rets[~np.isnan(rets)]
    if len(r) < block_size:
        return {"sharpe": np.nan, "ci_5": np.nan, "ci_95": np.nan, "n": len(r)}
    n_blocks = int(np.ceil(len(r) / block_size))
    sharpes = np.empty(n_iters)
    for i in range(n_iters):
        starts = rng.integers(0, len(r) - block_size + 1, size=n_blocks)
        sample = np.concatenate([r[s:s + block_size] for s in starts])[:len(r)]
        sd = sample.std(ddof=1)
        sharpes[i] = sample.mean() / sd * np.sqrt(12) if sd > 0 else 0.0
    return {
        "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(12)),
        "ci_5": float(np.percentile(sharpes, 5)),
        "ci_95": float(np.percentile(sharpes, 95)),
        "n": len(r),
    }


def ff5_alpha(net: pd.Series) -> tuple[float, float]:
    ff = fetch_ff5(); ff.index = ff.index.to_period("M")
    s = net.copy(); s.index = s.index.to_period("M")
    common = s.index.intersection(ff.index)
    y = s.loc[common].values - ff.loc[common, "RF"].values
    f = ff.loc[common]
    X = np.column_stack(
        [np.ones(len(y))] + [f[c].values for c in ["Mkt-RF","SMB","HML","RMW","CMA"]]
    )
    b, _, t = nw_ols(y, X)
    return float(b[0] * 12), float(t[0])


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("Phase 27b: plateau-zoom k-sweep with bootstrap CIs + FF5 alpha")
    print("k in [10..20]; %d bootstrap iters per k" % N_BOOT)
    print("=" * 70)

    preds_all = pd.read_parquet(PREDS)
    preds_xgb = preds_all["XGBoost"].copy()
    preds_xgb.index = preds_xgb.index.set_names(["date", "ticker"])
    valid_tickers = set(preds_xgb.index.get_level_values("ticker"))

    returns_wide = pd.read_parquet(RETURNS_FILE)
    returns_wide = returns_wide[[c for c in returns_wide.columns if c in valid_tickers]]
    next_returns = returns_wide.shift(-1)

    features = pd.read_parquet(FEATURES_FILE)
    features = features.loc[features.index.get_level_values("ticker").isin(valid_tickers)]
    sector_map = features.reset_index().groupby("ticker")["sector"].first().to_dict()
    fd = features.index.get_level_values("date"); ft = features.index.get_level_values("ticker")
    elig = {d: set(ft[fd == d].unique()) for d in fd.unique()}

    print(f"\nRunning {len(K_GRID)} values with {N_BOOT} bootstrap reps each...")
    rows = []
    for i, k in enumerate(K_GRID):
        weights = build_weights_for_k(preds_xgb, sector_map, elig, k)
        net = portfolio_returns(weights, next_returns, COST_BPS / 1e4)
        long_oos = net[net.index >= LONG_START]
        test = net[net.index >= TEST_START]
        bb_full = block_bootstrap_sharpe(net.values, BLOCK_SIZE, N_BOOT, SEED)
        bb_long = block_bootstrap_sharpe(long_oos.values, BLOCK_SIZE, N_BOOT, SEED)
        a_full, t_full = ff5_alpha(net)
        a_long, t_long = ff5_alpha(long_oos)
        a_test, t_test = ff5_alpha(test)
        rows.append({
            "k": k,
            "n_pos_per_rebal": int((weights != 0).sum(axis=1).median()),
            "sharpe_full": bb_full["sharpe"],
            "ci5_full": bb_full["ci_5"],
            "ci95_full": bb_full["ci_95"],
            "alpha_full_pct_yr": a_full * 100,
            "alpha_t_full": t_full,
            "sharpe_long": bb_long["sharpe"],
            "ci5_long": bb_long["ci_5"],
            "ci95_long": bb_long["ci_95"],
            "alpha_long_pct_yr": a_long * 100,
            "alpha_t_long": t_long,
            "sharpe_test": sharpe_ratio(test),
            "alpha_test_pct_yr": a_test * 100,
            "alpha_t_test": t_test,
        })
        print(f"  k={k:2d} pos={rows[-1]['n_pos_per_rebal']:3d}  "
              f"full Sh {bb_full['sharpe']:+.3f} [{bb_full['ci_5']:+.2f},{bb_full['ci_95']:+.2f}]  "
              f"alpha {a_full*100:+5.1f}%/yr t={t_full:+.2f}  "
              f"long Sh {bb_long['sharpe']:+.3f}  alpha t={t_long:+.2f}",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "plateau_metrics.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'plateau_metrics.csv'}")

    print("\nGenerating plateau-zoom figure...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # Top: Sharpe with bootstrap CIs (full + long)
    for col_sh, col_lo, col_hi, label, color in [
        ("sharpe_full", "ci5_full", "ci95_full", "Full-OOS 2012-2024", "#1F3864"),
        ("sharpe_long", "ci5_long", "ci95_long", "Long-OOS 2015-2024", "#22C55E"),
    ]:
        ax1.errorbar(df["k"], df[col_sh],
                     yerr=[df[col_sh] - df[col_lo], df[col_hi] - df[col_sh]],
                     fmt="o-", capsize=4, label=label, color=color, lw=1.6,
                     markersize=6, alpha=0.95)
        # Mark peak
        peak_i = df[col_sh].idxmax()
        ax1.scatter(df.loc[peak_i, "k"], df.loc[peak_i, col_sh], s=120,
                    facecolor="none", edgecolor=color, lw=2, zorder=10)
    ax1.scatter(df["sharpe_test"].values, df["sharpe_test"].values, alpha=0)  # placeholder
    ax1.plot(df["k"], df["sharpe_test"], "s--", label="Test-OOS 2019-2024 (no CI)",
             color="#DC2626", lw=1.2, markersize=4.5, alpha=0.85)
    ax1.axvline(20, color="grey", ls="--", lw=0.7, alpha=0.7, label="k=20 canonical")
    ax1.set_ylabel("Sharpe ratio", fontsize=11)
    ax1.set_title("Phase 27b - plateau zoom k=10..20 with 90% block-bootstrap CIs\n"
                  "Circled = per-window peak; CI overlap == statistically indistinguishable",
                  fontsize=11.5, weight="bold")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower right", fontsize=9.5, framealpha=0.92)

    # Bottom: FF5 alpha t-stat
    for col, label, color in [
        ("alpha_t_full", "Full-OOS", "#1F3864"),
        ("alpha_t_long", "Long-OOS", "#22C55E"),
        ("alpha_t_test", "Test-OOS", "#DC2626"),
    ]:
        ax2.plot(df["k"], df[col], "o-", label=label, color=color, lw=1.7,
                 markersize=5)
    ax2.axhline(5.0, color="grey", ls=":", lw=0.7, alpha=0.7,
                label="t = 5 reference")
    ax2.axvline(20, color="grey", ls="--", lw=0.7, alpha=0.7)
    ax2.set_xlabel("k (long/short picks per GICS sector)", fontsize=11)
    ax2.set_ylabel("FF5 alpha t-stat (Newey-West)", fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.legend(loc="lower right", fontsize=9.5, framealpha=0.92)
    ax2.set_xticks(K_GRID)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "k_sweep_plateau_zoom.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'k_sweep_plateau_zoom.png'}")

    # Print plateau summary
    print()
    print("=" * 70)
    print("PLATEAU READOUT (full-OOS)")
    print("=" * 70)
    print(f"  {'k':>3}  {'Sharpe':>7}  {'CI 5/95%':<18}  {'FF5 alpha':>10}  {'t-stat':>7}")
    for _, r in df.iterrows():
        print(f"  {int(r['k']):>3}  {r['sharpe_full']:>+7.3f}  "
              f"[{r['ci5_full']:+.2f},{r['ci95_full']:+.2f}]    "
              f"{r['alpha_full_pct_yr']:>+8.2f}%  {r['alpha_t_full']:>+6.2f}")

    # Overlap analysis vs k=20
    k20_row = df[df["k"] == 20].iloc[0]
    overlap_count = ((df["ci5_full"] <= k20_row["sharpe_full"]) &
                     (df["ci95_full"] >= k20_row["sharpe_full"])).sum()
    print()
    print(f"k=20 canonical Sharpe (full-OOS): {k20_row['sharpe_full']:+.3f}")
    print(f"  -> falls inside the 90% bootstrap CI of {overlap_count}/{len(df)} other k values "
          f"({100*overlap_count/len(df):.0f}%)")
    if overlap_count == len(df):
        print("  -> k=20 is statistically indistinguishable from EVERY k in 10..20")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
