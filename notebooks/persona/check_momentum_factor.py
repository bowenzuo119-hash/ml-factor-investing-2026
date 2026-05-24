"""check_momentum_factor.py - does the alpha survive a MOMENTUM control?

The canonical's signal is momentum-dominated (SHAP), but the factor regression
only uses FF5 (Mkt, SMB, HML, RMW, CMA) -- which has NO momentum factor. A
momentum-heavy strategy's "FF5 alpha" can largely be the momentum (UMD)
premium. This regresses the Phase 24-RT canonical on FF5 and on FF5 + UMD
(Carhart-style 6-factor) to see how much alpha survives a momentum control.

    python -m notebooks.persona.check_momentum_factor
"""

from __future__ import annotations

import io
import pickle
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from notebooks.persona.verify_phase23_headline import fetch_ff5, nw_ols

ROOT = Path(__file__).resolve().parents[2]
CANON = ROOT / "results" / "24_canonical_with_chmom" / "per_model_results.pkl"
MOM_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "F-F_Momentum_Factor_CSV.zip")


def fetch_mom() -> pd.Series:
    raw = urllib.request.urlopen(MOM_URL, timeout=60).read()
    z = zipfile.ZipFile(io.BytesIO(raw))
    txt = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    hi = next(i for i, l in enumerate(txt) if "Mom" in l and "," in l)
    rows = []
    for l in txt[hi + 1:]:
        p = [x.strip() for x in l.split(",")]
        if len(p) < 2 or not p[0].isdigit() or len(p[0]) != 6:
            break
        rows.append((p[0], float(p[1]) / 100.0))
    s = pd.DataFrame(rows, columns=["ym", "UMD"])
    s["date"] = pd.to_datetime(s["ym"], format="%Y%m") + pd.offsets.MonthEnd(0)
    return s.set_index("date")["UMD"]


def main() -> int:
    net = pickle.load(open(CANON, "rb"))["XGBoost"].portfolio_returns.dropna()
    ff = fetch_ff5()
    umd = fetch_mom()
    for d in (net, ):
        pass
    net.index = net.index.to_period("M")
    ff.index = ff.index.to_period("M")
    umd.index = umd.index.to_period("M")
    common = net.index.intersection(ff.index).intersection(umd.index)
    y = net.loc[common].values - ff.loc[common, "RF"].values
    f = ff.loc[common]
    mom = umd.loc[common].values

    print("=" * 64)
    print("MOMENTUM CONTROL - Phase 24-RT canonical, FF5 vs FF5+UMD")
    print("=" * 64)
    specs = [
        ("FF5", ["Mkt-RF", "SMB", "HML", "RMW", "CMA"], None),
        ("FF5 + UMD (Carhart-style 6F)", ["Mkt-RF", "SMB", "HML", "RMW", "CMA"], mom),
    ]
    for name, facs, extra in specs:
        cols = [np.ones(len(y))] + [f[c].values for c in facs]
        labels = ["alpha"] + facs
        if extra is not None:
            cols.append(extra); labels.append("UMD")
        X = np.column_stack(cols)
        beta, se, t = nw_ols(y, X)
        print(f"\n  {name}  (n={len(y)})")
        for nm, b, tt in zip(labels, beta, t):
            ann = f"  ann={b*12:+.1%}" if nm == "alpha" else ""
            star = "  <--" if nm in ("alpha", "UMD") else ""
            print(f"    {nm:8s} coef {b:+.4f}  t {tt:+.2f}{ann}{star}")

    # explicit before/after alpha
    X5 = np.column_stack([np.ones(len(y))] + [f[c].values for c in ["Mkt-RF","SMB","HML","RMW","CMA"]])
    a5 = nw_ols(y, X5)[0][0]
    X6 = np.column_stack([np.ones(len(y))] + [f[c].values for c in ["Mkt-RF","SMB","HML","RMW","CMA"]] + [mom])
    b6, _, t6 = nw_ols(y, X6)
    print(f"\n  Readout: FF5 alpha {a5*12:+.1%}/yr  ->  FF5+UMD alpha {b6[0]*12:+.1%}/yr "
          f"(t={t6[0]:+.2f})")
    print(f"  UMD loading {b6[-1]:+.2f} (t={t6[-1]:+.2f}) -- positive = momentum-tilted")
    drop = (a5 - b6[0]) / a5 * 100 if a5 else 0
    print(f"  alpha absorbed by momentum: {drop:.0f}%  "
          f"-> {'mostly momentum premium' if drop > 50 else 'survives momentum control'}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
