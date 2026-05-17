"""
Week 1 — Person C: Regime Feature Dataset Builder
===================================================
Run this on YOUR LAPTOP (not Google Colab — you need persistent file output).

What it pulls:
  1. S&P 500 daily prices (^GSPC via yfinance)    → realized vol + 3m return
  2. VIX index (^VIX via yfinance)
  3. 10Y & 2Y Treasury yields (FRED direct CSV)   → yield curve slope
  4. BAA & AAA corporate yields (FRED direct CSV) → credit spread

All 6 features are lagged 1 trading day before saving (no look-ahead bias).

Outputs:
  regime_features_daily_2005_2024.csv    <- daily, lagged 1 day
  regime_features_monthly_2005_2024.csv  <- month-end snapshot (Week 2 input)

SETUP (run once in your terminal):
  pip install pandas numpy yfinance requests

RUN:
  python week1_regime_data.py
"""

import warnings
warnings.filterwarnings("ignore")
import sys
import io
import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf
except ImportError:
    sys.exit("ERROR: yfinance not installed. Run:  pip install yfinance")


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
REPORT_DIR = BASE_DIR / "report"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
START_PULL   = "2004-01-01"
END_PULL     = "2024-12-31"
REPORT_START = "2005-01-01"

VOL_SHORT    = 21    # trading days ~ 1 month
VOL_LONG     = 63    # trading days ~ 3 months
RETURN_WIN   = 63

OUT_DAILY    = DATA_DIR / "regime_features_daily_2005_2024.csv"
OUT_MONTHLY  = DATA_DIR / "regime_features_monthly_2005_2024.csv"

FRED_SERIES  = {
    "GS10" : "treasury_10y",
    "GS2"  : "treasury_2y",
    "DBAA" : "yield_baa",
    "DAAA" : "yield_aaa",
}

FEATURE_COLS = [
    "rv_21d", "rv_63d", "vix",
    "yield_curve_slope", "credit_spread", "sp500_ret_3m",
]

SEP = "=" * 62

def banner(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

# ── STEP 1: S&P 500 ───────────────────────────────────────────────────────────
banner("STEP 1 / 4 — S&P 500 Price History (^GSPC)")

raw = yf.download("^GSPC", start=START_PULL, end=END_PULL,
                  progress=False, auto_adjust=True)
if raw.empty:
    sys.exit("ERROR: ^GSPC download failed. Check your internet connection.")

close = raw["Close"]
if isinstance(close, pd.DataFrame):
    close = close.iloc[:, 0]
sp500 = close.squeeze().dropna()
sp500.name = "sp500_close"
print(f"  ✓ {len(sp500)} rows | {sp500.index[0].date()} → {sp500.index[-1].date()}")

log_ret = np.log(sp500 / sp500.shift(1))
rv21    = (log_ret.rolling(VOL_SHORT).std() * np.sqrt(252)).rename("rv_21d")
rv63    = (log_ret.rolling(VOL_LONG).std()  * np.sqrt(252)).rename("rv_63d")
ret_3m  = sp500.pct_change(RETURN_WIN).rename("sp500_ret_3m")
print("  ✓ Realized vol (21d, 63d) and 3-month return computed")

# ── STEP 2: VIX ───────────────────────────────────────────────────────────────
banner("STEP 2 / 4 — VIX Index (^VIX)")

raw_vix = yf.download("^VIX", start=START_PULL, end=END_PULL,
                      progress=False, auto_adjust=True)
if raw_vix.empty:
    print("  ⚠ VIX download failed — column will be NaN")
    vix = pd.Series(np.nan, index=sp500.index, name="vix")
else:
    v = raw_vix["Close"]
    if isinstance(v, pd.DataFrame):
        v = v.iloc[:, 0]
    vix = v.squeeze().dropna().rename("vix")
    print(f"  ✓ {len(vix)} rows | {vix.index[0].date()} → {vix.index[-1].date()}")

# ── STEP 3: FRED yields ───────────────────────────────────────────────────────
banner("STEP 3 / 4 — FRED Macro Data (no API key needed)")

fred = {}
for sid, name in FRED_SERIES.items():
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        s = pd.read_csv(io.StringIO(r.text), index_col=0,
                        parse_dates=True, na_values=".")
        s.columns = [name]
        fred[name] = s[name].dropna()
        print(f"  ✓ {sid:<5} ({name}): {len(fred[name])} obs")
    except Exception as e:
        print(f"  ✗ {sid} failed: {e}")
        fred[name] = pd.Series(dtype=float, name=name)

# Yield curve slope
if fred.get("treasury_10y", pd.Series()).size and fred.get("treasury_2y", pd.Series()).size:
    slope = (fred["treasury_10y"] - fred["treasury_2y"]).rename("yield_curve_slope")
    print("  ✓ yield_curve_slope = 10Y − 2Y")
else:
    slope = pd.Series(dtype=float, name="yield_curve_slope")
    print("  ✗ yield_curve_slope unavailable")

# Credit spread
if fred.get("yield_baa", pd.Series()).size and fred.get("yield_aaa", pd.Series()).size:
    credit = (fred["yield_baa"] - fred["yield_aaa"]).rename("credit_spread")
    print("  ✓ credit_spread = BAA − AAA")
else:
    credit = pd.Series(dtype=float, name="credit_spread")
    print("  ✗ credit_spread unavailable")

# ── STEP 4: Combine, lag, save ────────────────────────────────────────────────
banner("STEP 4 / 4 — Combine, 1-Day Lag, Save")

idx = sp500.index   # master trading-day index

df = pd.DataFrame(index=idx)
df["rv_21d"]       = rv21
df["rv_63d"]       = rv63
df["sp500_ret_3m"] = ret_3m
df["vix"]          = vix.reindex(idx, method="ffill")
df["yield_curve_slope"] = (
    slope.reindex(idx, method="ffill") if slope.size else np.nan
)
df["credit_spread"] = (
    credit.reindex(idx, method="ffill") if credit.size else np.nan
)
df["sp500_close"] = sp500

print(f"  Raw combined shape: {df.shape}")

# ── THE KEY STEP: lag features 1 trading day ──────────────────────────────────
# On any given day we only see data that was available YESTERDAY.
# This is what prevents look-ahead bias.
df_lagged = df.copy()
df_lagged[FEATURE_COLS] = df[FEATURE_COLS].shift(1)

# Trim to report window and drop all-NaN rows
df_lagged = df_lagged.loc[REPORT_START:].dropna(subset=FEATURE_COLS, how="all")
print(f"  After lag + trim to {REPORT_START}: {df_lagged.shape}")

# Save daily
df_lagged.index.name = "date"
df_lagged.to_csv(OUT_DAILY)
print(f"\n  ✅  {OUT_DAILY}")

# Save monthly (last trading day of each calendar month)
df_monthly = df_lagged.resample("ME").last().dropna(subset=FEATURE_COLS, how="all")
df_monthly.index.name = "month_end"
df_monthly.to_csv(OUT_MONTHLY)
print(f"  ✅  {OUT_MONTHLY}")

# ── Summary ───────────────────────────────────────────────────────────────────
banner("SUMMARY — Monthly Dataset Statistics")
print(df_monthly[FEATURE_COLS].describe().round(4).to_string())

banner("MISSING DATA CHECK")
total, all_ok = len(df_monthly), True
for col in FEATURE_COLS:
    n = df_monthly[col].isnull().sum()
    pct = 100 * n / total
    icon = "✓" if n == 0 else "⚠"
    all_ok = all_ok and (n == 0)
    print(f"  {icon}  {col:<25}  {n:>4} missing  ({pct:.1f}%)")

msg = "All clean — ready for Week 2." if all_ok else "⚠ Some missing data. Check connectivity."
print(f"\n  {msg}")

banner("WEEK 1 DELIVERABLE — COMPLETE")
print(f"""
Files produced (in same folder as this script):
  {OUT_DAILY}
  {OUT_MONTHLY}

Feature columns (ALL lagged 1 trading day — no look-ahead bias):
  rv_21d              Annualised realised volatility, trailing 21 days
  rv_63d              Annualised realised volatility, trailing 63 days
  vix                 CBOE VIX index level
  yield_curve_slope   10Y minus 2Y US Treasury yield (from FRED)
  credit_spread       Moody's BAA minus AAA corporate yield (from FRED)
  sp500_ret_3m        Trailing 3-month S&P 500 simple return

Reference only (do NOT use as a model feature):
  sp500_close         Raw S&P 500 closing price

Bring regime_features_monthly_2005_2024.csv to the Week 1 Friday sync.
""")
