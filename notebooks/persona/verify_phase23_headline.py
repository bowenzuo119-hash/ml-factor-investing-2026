"""verify_phase23_headline.py - independent audit of the +1.05 broad canonical.

Person A rigor check on Person B's Phase 23g headline before it locks as the
report result. Loads B's saved XGBoost canonical net returns + weights and
INDEPENDENTLY (own FF download, own Newey-West HAC):

  1. confirms net Sharpe ~1.05
  2. checks dollar-neutrality: net weight per rebalance ~0?
  3. FF3/FF5 regression -> alpha (annualised), t(alpha) w/ Newey-West, market beta
  4. decomposes mean monthly return into beta*E[Mkt-RF] vs alpha
     (how much of the +34%/yr is just leveraged bull-market beta)

    python -m notebooks.persona.verify_phase23_headline
"""

from __future__ import annotations

import io
import pickle
import urllib.request
import zipfile

import numpy as np
import pandas as pd

RESULT = "results/23g_canonical_qfiltered_orig_tune/per_model_results.pkl"
FF5_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "F-F_Research_Data_5_Factors_2x3_CSV.zip")


def load_xgb():
    with open(RESULT, "rb") as f:
        r = pickle.load(f)
    res = r["XGBoost"]
    return res.portfolio_returns.dropna(), res.weights


def fetch_ff5() -> pd.DataFrame:
    raw = urllib.request.urlopen(FF5_URL, timeout=60).read()
    z = zipfile.ZipFile(io.BytesIO(raw))
    txt = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    hi = next(i for i, l in enumerate(txt) if "Mkt-RF" in l)
    cols = [c.strip() for c in txt[hi].split(",")][1:]
    rows = []
    for l in txt[hi + 1:]:
        p = [x.strip() for x in l.split(",")]
        if len(p) < 6 or not p[0].isdigit() or len(p[0]) != 6:
            break
        rows.append(p)
    df = pd.DataFrame(rows, columns=["ym"] + cols)
    df["date"] = pd.to_datetime(df["ym"], format="%Y%m") + pd.offsets.MonthEnd(0)
    for c in cols:
        df[c] = df[c].astype(float) / 100.0
    return df.set_index("date")[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]]


def nw_ols(y: np.ndarray, X: np.ndarray, lags: int = 6):
    """OLS with Newey-West HAC SEs. X includes the intercept column."""
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    Xe = X * resid[:, None]
    S = Xe.T @ Xe
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = Xe[L:].T @ Xe[:-L]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    return beta, se, beta / np.where(se > 0, se, np.nan)


def sharpe(r: pd.Series) -> float:
    return float(r.mean() / r.std(ddof=1) * np.sqrt(12))


def main() -> int:
    ret, weights = load_xgb()
    print("=" * 74)
    print("INDEPENDENT AUDIT - Phase 23g XGBoost broad canonical")
    print("=" * 74)
    print(f"net monthly returns: {len(ret)} months, "
          f"{ret.index.min().date()}..{ret.index.max().date()}")
    print(f"  net Sharpe (full): {sharpe(ret):+.3f}  "
          f"(ann ret {ret.mean()*12:+.1%}, ann vol {ret.std(ddof=1)*np.sqrt(12):.1%})")

    # 2. dollar-neutrality from weights
    print("\n[2] Dollar-neutrality (net = sum of weights per rebalance):")
    if isinstance(weights, pd.DataFrame):
        net = weights.sum(axis=1)
        gross = weights.abs().sum(axis=1)
        print(f"  net exposure  : mean {net.mean():+.4f}, "
              f"|max| {net.abs().max():.4f}  (≈0 => dollar-neutral)")
        print(f"  gross exposure: mean {gross.mean():.3f} "
              f"(2.0 = fully invested both legs x leverage 1.0)")
    else:
        print(f"  weights type {type(weights)} - skipping")

    # 3 + 4. FF regressions
    print("\n[3] Fama-French regressions (own Newey-West HAC, 6 lags):")
    try:
        ff = fetch_ff5()
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] FF download failed ({e}); skipping FF block")
        return 0

    r = ret.copy()
    r.index = r.index.to_period("M")
    ff.index = ff.index.to_period("M")
    common = r.index.intersection(ff.index)
    r = r.loc[common]
    f = ff.loc[common]
    y = (r.values - f["RF"].values)  # excess return
    mkt = f["Mkt-RF"].values

    for name, facs in [("FF3", ["Mkt-RF", "SMB", "HML"]),
                       ("FF5", ["Mkt-RF", "SMB", "HML", "RMW", "CMA"])]:
        X = np.column_stack([np.ones(len(y))] + [f[c].values for c in facs])
        beta, se, t = nw_ols(y, X)
        names = ["alpha"] + facs
        print(f"\n  {name}:  (n={len(y)})")
        for nm, b, tt in zip(names, beta, t):
            extra = f"  ann={b*12:+.1%}" if nm == "alpha" else ""
            print(f"    {nm:8s} coef {b:+.4f}  t {tt:+.2f}{extra}")

    # 4. decomposition (use FF5 market beta)
    Xm = np.column_stack([np.ones(len(y)), mkt])
    b_mkt, _, _ = nw_ols(y, Xm)
    alpha_capm, beta_capm = b_mkt[0], b_mkt[1]
    mean_excess = y.mean()
    beta_part = beta_capm * mkt.mean()
    print("\n[4] Return decomposition (CAPM, monthly excess):")
    print(f"  mean excess return : {mean_excess*100:+.3f}%/mo  ({mean_excess*12:+.1%}/yr)")
    print(f"  market beta        : {beta_capm:+.2f}")
    print(f"  beta * E[Mkt-RF]   : {beta_part*100:+.3f}%/mo  ({beta_part*12:+.1%}/yr) "
          f"= {beta_part/mean_excess*100:.0f}% of the return")
    print(f"  CAPM alpha         : {alpha_capm*100:+.3f}%/mo  ({alpha_capm*12:+.1%}/yr) "
          f"= {alpha_capm/mean_excess*100:.0f}% of the return")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
