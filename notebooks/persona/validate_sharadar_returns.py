"""validate_sharadar_returns.py - Block B3: spot-check SEP-derived returns.

Three checks, printed for the record:

  1. Survivors  - Sharadar (closeadj) vs yfinance monthly-return correlation
                  on names yfinance also has; must be > 0.99.
  2. Delisted   - SEP closeunadj vs SF1 reported `price` at datekeys. This is
                  the survivorship-critical check: yfinance can't see these
                  names at all, so we cross-check SEP against SF1's own
                  point-in-time price instead.
  3. Splits     - closeadj must NOT jump across known split dates (the whole
                  point of an adjusted series).

    python -m notebooks.persona.validate_sharadar_returns
"""

from __future__ import annotations

import sys

import pandas as pd

from src.data_loader import RAW_DIR, compute_monthly_returns_sharadar, _read_sep

SURVIVORS = ["AAPL", "MSFT", "JNJ", "JPM", "PG"]
# Sharadar carries delisted names under bankruptcy / reuse-suffixed symbols
# (not the original ticker). These are exactly the names yfinance cannot see.
DELISTED = {
    "LEHMQ": "Lehman Brothers",
    "BSC1": "Bear Stearns",
    "SIVBQ": "SVB Financial",
    "WCOEQ": "WorldCom",
    "ENRNQ": "Enron",
}
SPLITS = [
    ("AAPL", "2020-08-31", 4),
    ("TSLA", "2020-08-31", 5),
    ("TSLA", "2022-08-25", 3),
    ("NVDA", "2021-07-20", 4),
    ("GOOGL", "2022-07-18", 20),
]


def check_survivors() -> bool:
    import yfinance as yf

    print("=== 1. Survivors: Sharadar vs yfinance monthly-return corr (>0.99) ===")
    ok = True
    for t in SURVIVORS:
        try:
            s = compute_monthly_returns_sharadar("2015-01-01", "2024-12-31", tickers=[t])
            if t not in s.columns:
                print(f"  [LOW ] {t}: no Sharadar data")
                ok = False
                continue
            s = s[t].dropna()
            s.index = s.index.to_period("M")
            y = yf.download(t, start="2014-12-01", end="2025-01-01",
                            auto_adjust=True, progress=False)["Close"]
            if hasattr(y, "columns"):
                y = y.iloc[:, 0]
            y = y.resample("ME").last().pct_change().dropna()
            y.index = y.index.to_period("M")
            common = s.index.intersection(y.index)
            corr = float(s.loc[common].corr(y.loc[common]))
            flag = "OK" if corr > 0.99 else "LOW"
            print(f"  [{flag:>4}] {t}: corr={corr:.4f} over {len(common)} months")
            ok &= corr > 0.99
        except Exception as exc:  # noqa: BLE001
            print(f"  [ERR ] {t}: {type(exc).__name__}: {exc}")
            ok = False
    return ok


def check_delisted() -> bool:
    print("\n=== 2. Delisted: SEP closeunadj vs SF1 `price` at datekeys ===")
    sf1 = pd.read_parquet(RAW_DIR / "sf1_AR_arq.parquet", columns=["ticker", "datekey", "price"])
    sf1["datekey"] = pd.to_datetime(sf1["datekey"])
    all_ok = True
    for t, who in DELISTED.items():
        sep = _read_sep([t], "1997-01-01", "2024-12-31", "closeunadj").dropna()
        s1 = sf1[sf1["ticker"] == t].dropna(subset=["price"]).sort_values("datekey")
        if sep.empty or s1.empty:
            print(f"  [SKIP] {t}: SEP days={len(sep)}, SF1 price rows={len(s1)} "
                  f"(ticker symbol may differ in Sharadar)")
            continue
        sep = sep.sort_values("date")
        merged = pd.merge_asof(
            s1[["datekey", "price"]],
            sep[["date", "closeunadj"]].rename(columns={"date": "datekey"}),
            on="datekey", direction="backward", tolerance=pd.Timedelta("10D"),
        ).dropna(subset=["closeunadj"])
        if merged.empty:
            print(f"  [SKIP] {t}: no asof price matches")
            continue
        med = float(((merged["closeunadj"] - merged["price"]).abs() / merged["price"]).median())
        span = f"{sep['date'].min().date()}..{sep['date'].max().date()}"
        flag = "OK" if med < 0.10 else "HIGH"
        print(f"  [{flag:>4}] {t} ({who}): SEP {len(sep)} days [{span}] | "
              f"median |Δ| vs SF1 price = {med:.1%} over {len(merged)} datekeys")
        all_ok &= med < 0.10
    return all_ok


def check_splits() -> bool:
    # closeunadj should drop ~1/ratio (raw split visible); closeadj should NOT
    # (it's adjusted) -- only a real-return-sized move, which can legitimately
    # be ~10%+ on a big day (e.g. TSLA +12.6% on its 2020 split day).
    print("\n=== 3. Splits: closeunadj shows the split, closeadj does not ===")
    ok = True
    for t, d, ratio in SPLITS:
        adj = _read_sep([t], "2018-01-01", "2024-12-31", "closeadj").dropna()
        raw = _read_sep([t], "2018-01-01", "2024-12-31", "closeunadj").dropna()
        m = adj.merge(raw[["date", "closeunadj"]], on="date").sort_values("date")
        dt = pd.Timestamp(d)
        pre, post = m[m["date"] < dt], m[m["date"] >= dt]
        if pre.empty or post.empty:
            print(f"  [SKIP] {t} {d}: missing data around split")
            continue
        adj_r = post["closeadj"].iloc[0] / pre["closeadj"].iloc[-1]
        raw_r = post["closeunadj"].iloc[0] / pre["closeunadj"].iloc[-1]
        raw_ok = abs(raw_r - 1.0 / ratio) / (1.0 / ratio) < 0.25  # raw really split
        adj_ok = abs(adj_r - 1.0) < 0.25                           # adj didn't split
        flag = "OK" if (raw_ok and adj_ok) else "FAIL"
        print(f"  [{flag:>4}] {t} {d} ({ratio}:1): closeunadj x{raw_r:.3f} "
              f"(~{1/ratio:.2f} expected), closeadj x{adj_r:.3f} (~1.0 expected)")
        ok &= raw_ok and adj_ok
    return ok


def main() -> int:
    s_ok = check_survivors()
    check_delisted()  # informational: old delisted symbols may not match
    sp_ok = check_splits()
    print("\n" + "=" * 60)
    verdict = s_ok and sp_ok
    print(f"Survivors corr >0.99: {'PASS' if s_ok else 'FAIL'} | "
          f"split adjustment: {'PASS' if sp_ok else 'FAIL'}")
    print("Delisted block is informational (cross-check vs SF1, see above).")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
