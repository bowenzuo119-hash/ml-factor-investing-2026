"""report_figures_extra.py - the remaining optional presentation visuals.

  predicted_vs_realized.png   - calibration: binned mean realized return vs
                                prediction decile-bin (+ pooled rank IC)
  risk_return_scatter.png     - sigma-return plane: strategy, market, and the
                                10 prediction-decile portfolios, with Sharpe
                                iso-lines
  return_histogram.png        - strategy vs market monthly-return distribution
                                (mean/vol/skew/kurtosis annotated)
  scorecard.png               - one-glance metrics table, strategy vs market

    python -m notebooks.persona.report_figures_extra
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.metrics import sharpe_ratio, max_drawdown, annualised_return
from notebooks.persona.verify_phase23_headline import fetch_ff5

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "results" / "persona_figures"
CANON = ROOT / "results" / "24_canonical_with_chmom"
RETURNS = ROOT / "data" / "processed" / "returns_broad_sharadar_2002_2024.parquet"
C_BLUE, C_GREY, C_GREEN, C_RED = "#2563eb", "#6b7280", "#16a34a", "#ef4444"
DPI = 135


def _pass(preds, ret, n_bins=20):
    """One pass: pooled (pred, realized) for calibration + monthly decile
    portfolio returns for the risk-return plot."""
    dates = preds.index.get_level_values("date").unique().sort_values()
    pool_p, pool_r = [], []
    dec_rows = {}
    for t in dates:
        nxt = ret.index[ret.index > t]
        if len(nxt) == 0:
            continue
        p = preds.loc[t].dropna()
        r = ret.loc[nxt[0]]
        common = p.index.intersection(r.dropna().index)
        if len(common) < 100:
            continue
        pc, rc = p.loc[common], r.loc[common]
        pool_p.append(pc.values); pool_r.append(rc.values)
        dec = pd.qcut(pc.rank(method="first"), 10, labels=False) + 1
        dec_rows[nxt[0]] = {d: rc[dec.values == d].mean() for d in range(1, 11)}
    pool_p = np.concatenate(pool_p); pool_r = np.concatenate(pool_r)
    dec_df = pd.DataFrame(dec_rows).T.sort_index()
    return pool_p, pool_r, dec_df


def fig_predicted_vs_realized(pool_p, pool_r):
    ic = stats.spearmanr(pool_p, pool_r).correlation
    bins = pd.qcut(pd.Series(pool_p).rank(method="first"), 20, labels=False)
    df = pd.DataFrame({"p": pool_p, "r": pool_r, "b": bins})
    g = df.groupby("b").agg(pm=("p", "mean"), rm=("r", "mean"), rs=("r", "sem"))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(g["pm"], g["rm"] * 100, yerr=g["rs"] * 100, fmt="o", color=C_BLUE,
                ms=5, capsize=2, lw=1)
    sl, ic0 = np.polyfit(g["pm"], g["rm"] * 100, 1)
    xs = np.linspace(g["pm"].min(), g["pm"].max(), 50)
    ax.plot(xs, sl * xs + ic0, color=C_RED, lw=1.5, ls="--", label=f"slope {sl:.1f}")
    ax.axhline(0, color="black", lw=0.5); ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("model prediction (sector-relative, 20 bins)")
    ax.set_ylabel("avg realized next-month return (%)")
    ax.set_title(f"Calibration — realized return rises monotonically with prediction "
                 f"(pooled rank IC = {ic:+.3f})", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    fig.savefig(FIG / "predicted_vs_realized.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig); print(f"  predicted_vs_realized.png (IC {ic:+.3f})")


def fig_risk_return(net, mkt, dec_df):
    def ar_av(s):
        return annualised_return(s) * 100, s.std(ddof=1) * np.sqrt(12) * 100
    pts = []
    for d in range(1, 11):
        a, v = ar_av(dec_df[d].dropna())
        pts.append((f"D{d}", v, a, "#c7d2fe" if d < 10 else C_GREEN))
    sa, sv = ar_av(net); ma, mv = ar_av(mkt)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    # Sharpe iso-lines
    xmax = max([p[1] for p in pts] + [sv, mv]) * 1.15
    for sh in (0.5, 1.0, 1.5):
        xs = np.linspace(0, xmax, 50)
        ax.plot(xs, sh * xs, color="#d1d5db", lw=0.8, ls=":")
        ax.text(xmax * 0.98, sh * xmax * 0.98, f"SR {sh:.1f}", color="#9ca3af", fontsize=7)
    for lab, v, a, c in pts:
        ax.scatter(v, a, c=c, s=55, edgecolor="white", zorder=3)
        ax.annotate(lab, (v, a), fontsize=6.5, ha="center", va="center")
    ax.scatter(sv, sa, c=C_BLUE, s=180, marker="*", edgecolor="white", zorder=4,
               label=f"ML strategy (SR {sa/sv:.2f})")
    ax.scatter(mv, ma, c=C_GREY, s=120, marker="D", edgecolor="white", zorder=4,
               label=f"US market (SR {ma/mv:.2f})")
    ax.set_xlabel("annualised volatility (%)"); ax.set_ylabel("annualised return (%)")
    ax.set_title("Risk–return — strategy, market, and decile portfolios",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, xmax); ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=0.2)
    fig.savefig(FIG / "risk_return_scatter.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig); print("  risk_return_scatter.png")


def fig_histogram(net, mkt):
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(min(net.min(), mkt.min()), max(net.max(), mkt.max()), 40)
    ax.hist(mkt * 100, bins=bins * 100, alpha=0.5, color=C_GREY, label="US market", density=True)
    ax.hist(net * 100, bins=bins * 100, alpha=0.5, color=C_BLUE, label="ML strategy", density=True)
    ax.axvline(net.mean() * 100, color=C_BLUE, ls="--", lw=1)
    ax.axvline(mkt.mean() * 100, color=C_GREY, ls="--", lw=1)
    ax.set_xlabel("monthly return (%)"); ax.set_ylabel("density")
    txt = (f"strategy: μ {net.mean()*100:+.2f}%  σ {net.std()*100:.2f}%  "
           f"skew {stats.skew(net):+.2f}  kurt {stats.kurtosis(net):+.2f}")
    ax.set_title("Monthly return distribution — strategy vs market", fontsize=12, fontweight="bold")
    ax.text(0.02, 0.97, txt, transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round", fc="white", ec="#e5e7eb"))
    ax.legend(fontsize=10); ax.grid(alpha=0.2)
    fig.savefig(FIG / "return_histogram.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig); print("  return_histogram.png")


def _metrics(s, turnover=None):
    ann = annualised_return(s); vol = s.std(ddof=1) * np.sqrt(12)
    dn = s[s < 0].std(ddof=1) * np.sqrt(12)
    dd = max_drawdown(s)
    return {
        "Ann. return": f"{ann:+.1%}", "Ann. vol": f"{vol:.1%}",
        "Sharpe": f"{sharpe_ratio(s):+.2f}",
        "Sortino": f"{s.mean()*12/dn:+.2f}" if dn > 0 else "n/a",
        "Max drawdown": f"{dd:.1%}",
        "Calmar": f"{ann/abs(dd):+.2f}" if dd else "n/a",
        "Hit rate": f"{(s > 0).mean():.0%}",
        "Avg turnover": f"{turnover:.0%}" if turnover is not None else "—",
    }


def fig_scorecard(net, gross, mkt, turnover):
    sm = _metrics(net, turnover); mm = _metrics(mkt)
    keys = list(sm.keys())
    rows = [[k, sm[k], mm[k]] for k in keys]
    rows.append(["Gross Sharpe", f"{sharpe_ratio(gross):+.2f}", "—"])
    fig, ax = plt.subplots(figsize=(8, 5.2)); ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=["Metric", "ML strategy", "US market"],
                   colWidths=[0.4, 0.3, 0.3], cellLoc="center", loc="center", bbox=[0, 0, 1, 0.9])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.7)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#e5e7eb")
        if r == 0:
            cell.set_facecolor("#1f2937"); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f9fafb")
        if c == 1 and r > 0:
            cell.set_text_props(fontweight="bold", color=C_BLUE)
    ax.text(0.5, 0.97, "Scorecard — net of 10 bps/side costs (2012–2024 OOS)",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=12, fontweight="bold")
    fig.savefig(FIG / "scorecard.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig); print("  scorecard.png")


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    res = pickle.load(open(CANON / "per_model_results.pkl", "rb"))["XGBoost"]
    net = res.portfolio_returns.dropna(); gross = res.gross_returns.dropna()
    turnover = float(res.turnover.mean()) if hasattr(res, "turnover") else None
    ret = pd.read_parquet(RETURNS)
    preds = pd.read_parquet(CANON / "predictions.parquet")["XGBoost"]
    ff = fetch_ff5(); mkt = ff["Mkt-RF"] + ff["RF"]
    # align strategy & market by month period (trading-day vs calendar month-ends)
    net_p = net.copy(); net_p.index = net.index.to_period("M")
    mkt_p = mkt.copy(); mkt_p.index = mkt.index.to_period("M")
    common = net_p.index.intersection(mkt_p.index)
    net_al = net_p.loc[common].copy(); net_al.index = common.to_timestamp("M")
    mkt = mkt_p.loc[common].copy(); mkt.index = common.to_timestamp("M")

    pool_p, pool_r, dec_df = _pass(preds, ret)
    fig_predicted_vs_realized(pool_p, pool_r)
    fig_risk_return(net_al, mkt, dec_df)
    fig_histogram(net_al, mkt)
    fig_scorecard(net, gross, mkt, turnover)
    print(f"\nDone -> {FIG}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
