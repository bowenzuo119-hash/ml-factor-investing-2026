"""report_figures_broad.py - data-lane figures for REPORT 2 (broad rebuild).

Three figures into results/persona_figures/ (alongside the existing ones):

  1. universe_coverage_broad.png        - strict-PIT S&P500 vs broad top-2000
                                          vs total Sharadar coverage, over time
  2. universe_survivorship_comparison.png - the audit visual: naive (no-PIT)
                                          vs strict PIT vs broad PIT
  3. q_filter_exclusions.png            - Q-suffix bankrupt tickers dropped per
                                          year (clusters in 2008 / 2023)

Needs the local raw Sharadar tables (gitignored). Matches the existing figure
style (matplotlib, shared palette).

    python -m notebooks.persona.report_figures_broad
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data_loader import (
    RAW_DIR, PROCESSED_DIR, load_sp500_membership, load_universe_at,
)

FIG_DIR = Path(__file__).resolve().parents[2] / "results" / "persona_figures"
RETURNS = PROCESSED_DIR / "returns_broad_sharadar_2002_2024.parquet"

C_SP500 = "#60a5fa"   # blue  - strict PIT S&P 500
C_BROAD = "#f59e0b"   # amber - broad top-2000 (our canonical)
C_TOTAL = "#a78bfa"   # purple- total Sharadar coverage
C_NAIVE = "#ef4444"   # red   - naive no-PIT (the leak)
DPI = 130


def compute_counts() -> pd.DataFrame:
    ret = pd.read_parquet(RETURNS)
    dates = ret.index[ret.index >= pd.Timestamp("2002-04-01")]
    naive = ret.notna().sum(axis=1).reindex(dates)  # any ticker with data at t
    rows = []
    for i, d in enumerate(dates):
        sp = len(load_sp500_membership(asof=d.strftime("%Y-%m-%d")))
        broad = len(load_universe_at(d, top_n_by_marketcap=2000))
        total = len(load_universe_at(d, top_n_by_marketcap=10**9))
        rows.append({"date": d, "sp500": sp, "broad": broad,
                     "total": total, "naive": int(naive.loc[d])})
        if i % 24 == 0:
            print(f"  {d.date()}: sp500={sp}, broad={broad}, total={total}, naive={rows[-1]['naive']}")
    return pd.DataFrame(rows).set_index("date")


def fig_universe_coverage(c: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(c.index, c["total"], color=C_TOTAL, lw=1.6,
            label="Total Sharadar coverage (US common stock, major exch.)")
    ax.plot(c.index, c["broad"], color=C_BROAD, lw=2.0,
            label="Broad PIT top-2000 by market cap (canonical)")
    ax.plot(c.index, c["sp500"], color=C_SP500, lw=2.0,
            label="Strict PIT S&P 500")
    ax.set_title("Investable universe over time - strict PIT vs broad vs total",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("# eligible tickers at rebalance")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.text(0.5, -0.02,
             "The broad survivorship-free universe (~2000/mo) sits between the "
             "~500-name S&P 500 and the full ~5-8k Sharadar coverage.",
             ha="center", fontsize=8.5, style="italic")
    fig.savefig(FIG_DIR / "universe_coverage_broad.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  wrote universe_coverage_broad.png")


def fig_survivorship(c: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(c.index, c["naive"], color=C_NAIVE, lw=2.0,
            label="Naive (no PIT) - any ticker with data at t  [the leak]")
    ax.plot(c.index, c["broad"], color=C_BROAD, lw=2.0,
            label="Broad PIT top-2000 by mcap (canonical)")
    ax.plot(c.index, c["sp500"], color=C_SP500, lw=2.0,
            label="Strict PIT S&P 500")
    ax.fill_between(c.index, c["broad"], c["naive"], color=C_NAIVE, alpha=0.08)
    ax.set_title("Survivorship correction - what the engine is allowed to trade",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("# eligible tickers at rebalance")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.text(0.5, -0.02,
             "Gap between naive and PIT = the survivorship leak we closed; "
             "gap between S&P-500 and broad PIT = the universe expansion that unlocked alpha.",
             ha="center", fontsize=8.5, style="italic")
    fig.savefig(FIG_DIR / "universe_survivorship_comparison.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  wrote universe_survivorship_comparison.png")


def fig_qfilter() -> None:
    tk = pd.read_parquet(RAW_DIR / "tickers.parquet",
                         columns=["table", "ticker", "isdelisted", "lastpricedate"])
    tk = tk[tk["table"] == "SEP"]
    q = tk[(tk["isdelisted"] == "Y")
           & tk["ticker"].str.upper().str.match(r".{3,}Q$")].copy()
    q["year"] = pd.to_datetime(q["lastpricedate"], errors="coerce").dt.year
    by_year = q[(q["year"] >= 2002) & (q["year"] <= 2024)]["year"].value_counts().sort_index()
    by_year = by_year.reindex(range(2002, 2025), fill_value=0)

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(by_year.index, by_year.values, color=C_SP500, edgecolor="white")
    for yr in (2008, 2023):
        if yr in by_year.index:
            ax.annotate(f"{yr}\n({int(by_year.loc[yr])})",
                        xy=(yr, by_year.loc[yr]), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=8,
                        fontweight="bold", color="#b91c1c")
    ax.set_title("Bankrupt-ticker exclusions per year (Q-suffix, delisted)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("# Q-suffix tickers dropped")
    ax.set_xlabel("year of last price")
    ax.grid(alpha=0.25, axis="y")
    fig.text(0.5, -0.02,
             "Bankruptcies cluster in crises (GFC 2008, regional-bank stress 2023), "
             "so the Q-filter has an economic basis rather than being arbitrary cleaning.",
             ha="center", fontsize=8.5, style="italic")
    fig.savefig(FIG_DIR / "q_filter_exclusions.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote q_filter_exclusions.png ({int(by_year.sum())} Q-tickers 2002-24)")


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("Computing per-month universe counts ...")
    counts = compute_counts()
    print("\nGenerating figures ...")
    fig_universe_coverage(counts)
    fig_survivorship(counts)
    fig_qfilter()
    print(f"\nDone -> {FIG_DIR}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
