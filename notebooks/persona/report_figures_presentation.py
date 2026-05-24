"""report_figures_presentation.py - presentation visual aids (Person A lane).

Fills the high-value gaps in the figure deck not already covered by
report_figures.py / report_figures_audit.py / report_figures_broad.py and not
owned by Person B (SHAP, model comparison, long/short decomp). All to
results/persona_figures/:

  pipeline_flowchart.png        - end-to-end system schema (the "map" slide)
  cumulative_vs_market.png      - canonical vs US market, log scale
  decile_sort_returns.png       - avg next-month return by prediction decile
  feature_correlation.png       - 14-feature correlation heatmap
  monthly_return_heatmap.png    - calendar grid of net monthly returns
  gross_vs_net_equity.png       - transaction-cost impact on the equity curve
  sector_exposure.png           - net exposure per GICS sector over time

    python -m notebooks.persona.report_figures_presentation
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from notebooks.persona.verify_phase23_headline import fetch_ff5

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "results" / "persona_figures"
CANON = ROOT / "results" / "24_canonical_with_chmom"
RETURNS = ROOT / "data" / "processed" / "returns_broad_sharadar_2002_2024.parquet"
FEATURES = ROOT / "data" / "processed" / "features_broad_sharadar_with_chmom_maxret.parquet"
FEATS_13 = ["mom", "rev", "mvol", "ivol", "log_mktcap", "dvol", "bm", "ep",
            "roe", "roa", "de", "asset_growth", "accruals", "chmom"]
C_BLUE, C_AMBER, C_RED, C_GREEN = "#2563eb", "#f59e0b", "#ef4444", "#16a34a"
DPI = 135


def _xgb():
    return pickle.load(open(CANON / "per_model_results.pkl", "rb"))["XGBoost"]


def fig_pipeline_flowchart():
    boxes = [
        ("Sharadar\nSF1·SEP·DAILY·TICKERS\n(survivorship-free)", C_BLUE),
        ("Broad PIT universe\n~4,400/mo (survivorship-free)\n+ 14 features", C_BLUE),
        ("XGBoost\ncross-sectional\nreturn forecast", C_AMBER),
        ("Sector-relative\nrank → top/bottom\nk=20 per GICS", C_AMBER),
        ("Dollar-neutral\nlong–short book\n(10 bps/side)", C_GREEN),
        ("Walk-forward\nbacktest (PIT)\n+ FF5/UMD, DSR", C_GREEN),
    ]
    fig, ax = plt.subplots(figsize=(15, 3.2)); ax.axis("off")
    n = len(boxes); w = 0.142; gap = (1 - n * w) / (n - 1)
    for i, (txt, c) in enumerate(boxes):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.3), w, 0.42, boxstyle="round,pad=0.012",
                                    fc=c, ec="none", alpha=0.92, transform=ax.transAxes))
        ax.text(x + w / 2, 0.51, txt, ha="center", va="center", fontsize=8.5,
                color="white", fontweight="bold", transform=ax.transAxes)
        if i < n - 1:
            ax.annotate("", xy=(x + w + gap * 0.95, 0.51), xytext=(x + w, 0.51),
                        xycoords=ax.transAxes, textcoords=ax.transAxes,
                        arrowprops=dict(arrowstyle="-|>", color="#374151", lw=2.2))
    ax.set_title("End-to-end pipeline — data → forecast → portfolio → evaluation",
                 fontsize=13, fontweight="bold", y=0.92)
    fig.savefig(FIG / "pipeline_flowchart.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig); print("  pipeline_flowchart.png")


def fig_cumulative_vs_market(net):
    ff = fetch_ff5()
    mkt = (ff["Mkt-RF"] + ff["RF"])  # total US market monthly return
    n = net.copy(); n.index = n.index.to_period("M")
    mkt.index = mkt.index.to_period("M")
    common = n.index.intersection(mkt.index)
    n, m = n.loc[common], mkt.loc[common]
    idx = [p.to_timestamp("M") for p in common]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(idx, (1 + n).cumprod(), color=C_BLUE, lw=2.2, label="ML long–short (net)")
    ax.plot(idx, (1 + m).cumprod(), color="#6b7280", lw=1.8, ls="--", label="US market")
    ax.set_yscale("log"); ax.set_ylabel("growth of $1 (log)")
    ax.set_title("Cumulative net return — ML strategy vs US market", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10); ax.grid(alpha=0.25, which="both")
    fig.savefig(FIG / "cumulative_vs_market.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig); print("  cumulative_vs_market.png")


def fig_decile_sort(preds, ret):
    dates = preds.index.get_level_values("date").unique().sort_values()
    buckets = {d: [] for d in range(1, 11)}
    for t in dates:
        nxt = ret.index[ret.index > t]
        if len(nxt) == 0:
            continue
        p = preds.loc[t].dropna()
        r = ret.loc[nxt[0]]
        common = p.index.intersection(r.dropna().index)
        if len(common) < 100:
            continue
        dec = pd.qcut(p.loc[common].rank(method="first"), 10, labels=False) + 1
        rc = r.loc[common]
        for dd in range(1, 11):
            buckets[dd].extend(rc[dec.values == dd].values)
    means = [np.mean(buckets[d]) * 100 for d in range(1, 11)]
    spread = means[-1] - means[0]
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = [C_RED if v < 0 else C_GREEN for v in means]
    ax.bar(range(1, 11), means, color=colors, edgecolor="white")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("prediction decile (1 = most bearish → 10 = most bullish)")
    ax.set_ylabel("avg next-month return (%)")
    ax.set_title(f"Cross-sectional signal — avg return by prediction decile "
                 f"(D10−D1 spread = {spread:+.2f}%/mo)", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.2, axis="y")
    fig.savefig(FIG / "decile_sort_returns.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig); print(f"  decile_sort_returns.png (spread {spread:+.2f}%/mo)")


def fig_feature_corr(features):
    c = features[FEATS_13].corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(c.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATS_13))); ax.set_xticklabels(FEATS_13, rotation=90, fontsize=8)
    ax.set_yticks(range(len(FEATS_13))); ax.set_yticklabels(FEATS_13, fontsize=8)
    for i in range(len(FEATS_13)):
        for j in range(len(FEATS_13)):
            v = c.values[i, j]
            if abs(v) > 0.15 and i != j:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(v) > 0.5 else "black")
    ax.set_title("Feature correlation (sector-relative ranks) — low off-diagonal = little redundancy",
                 fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(FIG / "feature_correlation.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig); print("  feature_correlation.png")


def fig_monthly_heatmap(net):
    df = pd.DataFrame({"y": net.index.year, "m": net.index.month, "r": net.values * 100})
    grid = df.pivot_table(index="y", columns="m", values="r")
    fig, ax = plt.subplots(figsize=(12, 5))
    vmax = np.nanmax(np.abs(grid.values))
    im = ax.imshow(grid.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(12)); ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
    ax.set_yticks(range(len(grid.index))); ax.set_yticklabels(grid.index, fontsize=8)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5)
    ax.set_title("Net monthly return (%) — calendar heatmap", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="% / month")
    fig.savefig(FIG / "monthly_return_heatmap.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig); print("  monthly_return_heatmap.png")


def fig_gross_vs_net(res):
    gross = res.gross_returns.dropna(); net = res.portfolio_returns.dropna()
    idx = gross.index.intersection(net.index)
    gross, net = gross.loc[idx], net.loc[idx]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(idx, (1 + gross).cumprod(), color=C_GREEN, lw=2.0, label="gross")
    ax.plot(idx, (1 + net).cumprod(), color=C_BLUE, lw=2.0, label="net (10 bps/side)")
    ax.set_yscale("log"); ax.set_ylabel("growth of $1 (log)")
    drag = (gross.mean() - net.mean()) * 12 * 100
    ax.set_title(f"Transaction-cost impact — gross vs net equity "
                 f"(cost drag ≈ {drag:.1f}%/yr at 10 bps, ~175% turnover)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10); ax.grid(alpha=0.25, which="both")
    fig.savefig(FIG / "gross_vs_net_equity.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig); print("  gross_vs_net_equity.png")


def fig_sector_exposure(res, features):
    w = res.weights
    if not isinstance(w, pd.DataFrame):
        print("  (weights not a frame; skipping sector_exposure)"); return
    sec = features.reset_index().groupby("ticker")["sector"].first()
    sectors = sorted(sec.dropna().unique())
    net_exp = pd.DataFrame(index=w.index, columns=sectors, dtype=float)
    secmap = sec.to_dict()
    col_sec = pd.Series({c: secmap.get(c, "Unknown") for c in w.columns})
    for s in sectors:
        cols = col_sec[col_sec == s].index
        net_exp[s] = w[cols.intersection(w.columns)].sum(axis=1)
    fig, ax = plt.subplots(figsize=(12, 5))
    for s in sectors:
        ax.plot(net_exp.index, net_exp[s], lw=1.0, alpha=0.8, label=s)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("net exposure (Σ weights in sector)")
    ax.set_title("Net exposure per GICS sector — hovers near 0 (sector-neutral construction)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, ncol=4, loc="upper center")
    ax.grid(alpha=0.2)
    fig.savefig(FIG / "sector_exposure.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig); print(f"  sector_exposure.png (mean |net|/sector = {net_exp.abs().mean().mean():.3f})")


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    res = _xgb()
    net = res.portfolio_returns.dropna()
    ret = pd.read_parquet(RETURNS)
    feats = pd.read_parquet(FEATURES)
    preds = pd.read_parquet(CANON / "predictions.parquet")["XGBoost"]

    fig_pipeline_flowchart()
    fig_cumulative_vs_market(net)
    fig_decile_sort(preds, ret)
    fig_feature_corr(feats)
    fig_monthly_heatmap(net)
    fig_gross_vs_net(res)
    fig_sector_exposure(res, feats)
    print(f"\nDone -> {FIG}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
