"""Phase 6 sector-neutrality audit.

Answers five concrete questions about the canonical Phase-3c portfolio:

1. Sector composition of the actual long and short books, per rebalance date.
2. Concentration: Herfindahl of long-leg sector weights, over time, vs the
   ~0.09 equal-sector benchmark for 11 GICS sectors.
3. Net sector exposure: long-count minus short-count per sector per month,
   to see if any sector is systematically over-weighted.
4. Sector size distribution: how many investable stocks per sector per
   month, to flag thin-sector noise (Materials, Energy, Real Estate).
5. Sector-mapping source audit: for each ticker that appears in 2019-2024,
   was its sector resolved via current GICS, via the 2-digit-SIC fallback,
   or as Unknown?

Output goes to results/06_sector_audit/. No model changes; purely
diagnostic on the artefacts Phase 3c already produced.

Run with:
    .venv/bin/python -m notebooks.personb.06_sector_audit
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import (
    PROCESSED_DIR, RAW_DIR, SP500_CURRENT_FILE,
    download_sp500_universe, load_prices,
)
from src.factors import _SIC2_TO_SECTOR, get_sector, load_sector_map


RESULTS_DIR = (
    Path(__file__).resolve().parents[2] / "results" / "06_sector_audit"
)
PHASE_DIR = (
    Path(__file__).resolve().parents[2] / "results"
    / "10_layer3_sector_neutral"
)

TEST_START = pd.Timestamp("2019-01-01")
TEST_END = pd.Timestamp("2024-12-31")


def sector_mapping_with_source(tickers: list[str]) -> pd.DataFrame:
    """Return (ticker, sector, source) for every ticker in the universe.

    Mirrors the resolution order used inside `factors.build_feature_panel`
    but with the source tagged so we can audit it.
    """
    # Current GICS from fja05680/sp500
    current_map = load_sector_map()

    # SIC fallback: pull each ticker's latest CRSP sic_code
    crsp = load_prices(start="2005-01-01", end="2022-12-30")
    permno_to_ticker = (
        crsp.reset_index().sort_values("date")
        .groupby("permno")["ticker"].last()
    )
    sic_long = (
        crsp[["sic_code"]].reset_index()
        .assign(ticker=lambda d: d["permno"].map(permno_to_ticker))
        .dropna(subset=["ticker"])
        .sort_values("date")
        .groupby("ticker")["sic_code"].last()
    )

    rows = []
    for t in tickers:
        sec_gics = current_map.get(t.upper())
        sic = sic_long.get(t)
        if sec_gics:
            source = "GICS (fja05680/sp500)"
            sector = sec_gics
        elif sic is not None and not (
            isinstance(sic, float) and np.isnan(sic)
        ):
            sec_sic = get_sector(t, sic, {})  # force SIC path with empty GICS map
            if sec_sic == "Unknown":
                source = "Unknown"
                sector = "Unknown"
            else:
                source = f"SIC fallback ({int(float(sic)) // 100})"
                sector = sec_sic
        else:
            source = "Unknown"
            sector = "Unknown"
        rows.append({"ticker": t, "sector": sector, "source": source,
                     "sic_code": str(sic) if sic is not None else ""})
    return pd.DataFrame(rows)


def herfindahl(weights: pd.Series) -> float:
    """Herfindahl of a positive weight vector: sum(w_i^2) / (sum w_i)^2."""
    w = weights[weights > 0]
    if w.empty:
        return float("nan")
    s = w.sum()
    return float(((w / s) ** 2).sum())


def sector_herfindahl(weights: pd.Series, ticker_to_sector: dict) -> float:
    """Herfindahl of weights AGGREGATED to sector level.

    Per-stock Herfindahl of equal-weighted N positions is always ~1/N -- not
    informative about sector concentration. Aggregating to sector first gives
    the metric we actually care about: how unevenly the (signed) weight is
    distributed across the ~11 sectors.
    """
    w = weights[weights > 0]
    if w.empty:
        return float("nan")
    sectors = pd.Series(
        [ticker_to_sector.get(t, "Unknown") for t in w.index], index=w.index
    )
    sector_weights = w.groupby(sectors).sum()
    s = sector_weights.sum()
    return float(((sector_weights / s) ** 2).sum())


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("Phase 6 sector-neutrality audit")
    print("=" * 72)

    # 1. Load Phase 3c canonical portfolio ----------------
    print("\n[1/5] Loading Phase 3c XGBoost weights...")
    with open(PHASE_DIR / "per_model_results.pkl", "rb") as f:
        results = pickle.load(f)
    res = results["XGBoost"]
    # res.weights: wide DataFrame, rows = rebalance dates, cols = tickers
    weights = res.weights
    print(f"  weights shape: {weights.shape}")

    # Restrict to test window 2019-2024
    test_mask = (weights.index >= TEST_START) & (weights.index <= TEST_END)
    weights_test = weights.loc[test_mask]
    print(f"  test-window weights: {weights_test.shape}")

    # 2. Resolve sector for every ticker --------------------
    print("\n[2/5] Mapping every ticker -> sector (with source)...")
    all_tickers = sorted({t for t in weights_test.columns if t})
    sector_df = sector_mapping_with_source(all_tickers)
    print(f"  {len(sector_df)} unique tickers in test window")
    src_counts = sector_df.groupby("source")["ticker"].count().sort_values(ascending=False)
    print(f"\n  sector-source breakdown:")
    for src, n in src_counts.items():
        pct = 100 * n / len(sector_df)
        print(f"    {src:35s}  {n:4d}  ({pct:5.1f}%)")
    sector_df.to_parquet(RESULTS_DIR / "sector_mapping_audit.parquet")

    # Ticker -> sector dict for fast lookups
    t2s = dict(zip(sector_df["ticker"], sector_df["sector"]))

    # 3. Sector composition per rebalance ---------------------
    print("\n[3/5] Computing per-rebalance long/short sector counts...")
    all_sectors = sorted(sector_df["sector"].unique())
    long_counts = pd.DataFrame(0, index=weights_test.index,
                               columns=all_sectors, dtype=float)
    short_counts = pd.DataFrame(0, index=weights_test.index,
                                columns=all_sectors, dtype=float)
    long_concentration = pd.Series(np.nan, index=weights_test.index,
                                   name="herfindahl_long")
    short_concentration = pd.Series(np.nan, index=weights_test.index,
                                    name="herfindahl_short")

    for date, row in weights_test.iterrows():
        # Longs
        longs = row[row > 0]
        for tkr, w in longs.items():
            sec = t2s.get(tkr, "Unknown")
            long_counts.loc[date, sec] += 1
        # Shorts
        shorts = row[row < 0]
        for tkr, w in shorts.items():
            sec = t2s.get(tkr, "Unknown")
            short_counts.loc[date, sec] += 1
        # Sector-aggregated Herfindahl (this is what we actually care about)
        long_concentration.loc[date] = sector_herfindahl(longs, t2s)
        short_concentration.loc[date] = sector_herfindahl(-shorts, t2s)

    long_counts.to_parquet(RESULTS_DIR / "long_counts_by_sector.parquet")
    short_counts.to_parquet(RESULTS_DIR / "short_counts_by_sector.parquet")

    # Net per-sector exposure (long - short positions)
    net_counts = long_counts - short_counts
    net_counts.to_parquet(RESULTS_DIR / "net_counts_by_sector.parquet")

    print(f"\n  per-sector average long positions (test window mean):")
    avg_long = long_counts.mean().sort_values(ascending=False)
    avg_short = short_counts.mean().sort_values(ascending=False)
    avg_net = net_counts.mean().sort_values(ascending=False)
    for sec, n_long in avg_long.items():
        n_short = short_counts[sec].mean()
        n_net = n_long - n_short
        print(f"    {sec:25s}  long {n_long:4.1f}  short {n_short:4.1f}  "
              f"net {n_net:+5.1f}")

    eq_baseline = 1.0 / len(all_sectors)
    print(f"\n  long-leg Herfindahl, statistics:")
    print(f"    mean    = {long_concentration.mean():.4f}")
    print(f"    median  = {long_concentration.median():.4f}")
    print(f"    max     = {long_concentration.max():.4f}  "
          f"(on {long_concentration.idxmax().date()})")
    print(f"    equal-{len(all_sectors)}-sector baseline = {eq_baseline:.4f}")

    # 4. Sector size diagnostic ---------------------
    # How many stocks per sector are in the investable universe each month
    print("\n[4/5] Sector size over time (investable universe)...")
    # Use the predictions panel as the "investable universe"
    preds = pd.read_parquet(PHASE_DIR / "predictions.parquet")
    preds_test = preds[(preds.index.get_level_values("date") >= TEST_START)
                       & (preds.index.get_level_values("date") <= TEST_END)]
    universe_by_date = (
        preds_test.reset_index()[["date", "ticker"]]
        .drop_duplicates()
    )
    universe_by_date["sector"] = universe_by_date["ticker"].map(t2s).fillna("Unknown")
    sector_size = (
        universe_by_date.groupby(["date", "sector"])["ticker"].nunique()
        .unstack("sector").fillna(0).astype(int)
    )
    sector_size.to_parquet(RESULTS_DIR / "sector_size_over_time.parquet")
    print(f"\n  average # stocks per sector (test window):")
    avg_size = sector_size.mean().sort_values(ascending=False)
    for sec, n in avg_size.items():
        flag = " <- THIN" if n < 25 else ""
        print(f"    {sec:25s}  {n:5.1f}{flag}")

    # 5. Plots ----------------
    print("\n[5/5] Saving plots...")

    # 5a. Long-leg Herfindahl over time vs equal baseline
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(long_concentration.index, long_concentration.values,
            color="#1F3864", lw=1.6, label="Actual long-leg Herfindahl")
    ax.plot(short_concentration.index, short_concentration.values,
            color="#DC2626", lw=1.2, alpha=0.7, label="Actual short-leg Herfindahl")
    ax.axhline(eq_baseline, color="green", lw=1.5, linestyle="--",
               label=f"Equal across {len(all_sectors)} sectors ({eq_baseline:.3f})")
    ax.set_title("Sector concentration over time (Herfindahl index of leg weights)\n"
                 "Higher = more concentrated in fewer sectors",
                 fontsize=11, weight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Herfindahl")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "herfindahl_over_time.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)

    # 5b. Net sector exposure heatmap (sector x date)
    # Re-order sectors by average size so largest is on top
    sector_order = avg_size.index.tolist()
    net_for_plot = net_counts[sector_order].T  # rows = sector, cols = date
    fig, ax = plt.subplots(figsize=(12, 5.5))
    vmax = float(np.nanmax(np.abs(net_for_plot.to_numpy())))
    im = ax.imshow(net_for_plot.to_numpy(),
                   aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax,
                   extent=[0, net_for_plot.shape[1] - 1,
                           net_for_plot.shape[0] - 1, 0])
    ax.set_yticks(range(len(sector_order)))
    ax.set_yticklabels(sector_order)
    # Show some date ticks
    n_dates = net_for_plot.shape[1]
    tick_idx = np.linspace(0, n_dates - 1, min(8, n_dates)).astype(int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(
        [str(net_for_plot.columns[i].date()) for i in tick_idx],
        rotation=30, ha="right",
    )
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("Net positions (longs - shorts) per sector")
    ax.set_title("Net sector exposure over time\n"
                 "Red = net long that sector, Blue = net short, White = balanced",
                 fontsize=11, weight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "net_sector_exposure_heatmap.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)

    # 5c. Sector size over time (line plot, one per sector)
    fig, ax = plt.subplots(figsize=(11, 5))
    for sec in sector_order:
        color = "#DC2626" if avg_size[sec] < 25 else "#1F3864"
        ax.plot(sector_size.index, sector_size[sec].values,
                color=color, lw=1.2, alpha=0.75)
        # Annotate at the right
        ax.text(sector_size.index[-1], sector_size[sec].iloc[-1],
                f" {sec}", fontsize=8, va="center",
                color=color)
    ax.axhline(25, color="grey", lw=0.8, linestyle="--",
               label="thin-sector threshold (25 names)")
    ax.set_title("Investable stocks per sector, test window 2019-2024\n"
                 "Red lines = thin sectors (avg < 25)",
                 fontsize=11, weight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("# stocks in universe")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "sector_size_over_time.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote 3 PNG plots + 4 parquet files to "
          f"{RESULTS_DIR.relative_to(Path.cwd())}/")

    # 6. Summary report -----------
    eq_baseline = 1.0 / len(all_sectors)
    mean_h = long_concentration.mean()
    excess = (mean_h - eq_baseline) / eq_baseline * 100

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(f"Long-leg Herfindahl avg = {mean_h:.4f}  vs  "
          f"equal-sector baseline {eq_baseline:.4f}")
    print(f"Excess concentration = {excess:+.1f}% above pure equal-weight")
    print(f"Worst month: {long_concentration.idxmax().date()} "
          f"(Herfindahl {long_concentration.max():.4f})")
    print()
    print(f"Thin sectors (avg < 25 names): "
          f"{', '.join([s for s in sector_order if avg_size[s] < 25])}")
    print()
    print(f"Sector-source breakdown (% of {len(sector_df)} tickers):")
    for src, n in src_counts.items():
        print(f"  {src:35s}  {100*n/len(sector_df):5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
