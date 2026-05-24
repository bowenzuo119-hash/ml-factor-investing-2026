"""cost_sensitivity_phase23.py - how high can costs go before the alpha dies?

The Phase 23g canonical was run at 10 bps/side. With a heavy small-cap tilt
(SMB beta +1.1) and ~175% monthly turnover, 10 bps is the optimistic end for
the smaller names. Cost is linear in bps, so realised cost at X bps is exactly
(X/10) x the realised cost at 10 bps -- no turnover re-alignment needed:

    net(X) = gross - (X/10) * (gross - net@10bps)

Reports net Sharpe and FF5 alpha (own Newey-West) across a cost grid, so we
can state the break-even cost at which the alpha loses significance.

    python -m notebooks.persona.cost_sensitivity_phase23
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from notebooks.persona.verify_phase23_headline import (
    RESULT, fetch_ff5, nw_ols, sharpe,
)

BPS_GRID = [10, 20, 30, 50, 75, 100]


def main() -> int:
    import sys
    from pathlib import Path
    # optional arg: a result dir or per_model_results.pkl to cost-test (defaults
    # to 23g). e.g. `... cost_sensitivity_phase23 results/24b_canonical_all_gkx`
    result_path = RESULT
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        result_path = p / "per_model_results.pkl" if p.is_dir() else p
    print(f"cost grid on: {result_path}")
    res = pickle.load(open(result_path, "rb"))["XGBoost"]
    gross = res.gross_returns.dropna()
    net10 = res.portfolio_returns.dropna()
    idx = gross.index.intersection(net10.index)
    gross, net10 = gross.loc[idx], net10.loc[idx]
    cost10 = gross - net10  # realised cost series at 10 bps

    print("=" * 70)
    print("COST SENSITIVITY - Phase 23g XGBoost broad canonical")
    print("=" * 70)
    print(f"realised cost @10bps: {cost10.mean()*1e4:.1f} bps/mo "
          f"(~{cost10.mean()*12*100:.1f}%/yr drag); n={len(idx)} months")

    ff = fetch_ff5()
    ffp = ff.copy()
    ffp.index = ffp.index.to_period("M")

    rows = []
    for bps in BPS_GRID:
        net = gross - (bps / 10.0) * cost10
        r = net.copy()
        r.index = r.index.to_period("M")
        common = r.index.intersection(ffp.index)
        r, f = r.loc[common], ffp.loc[common]
        y = r.values - f["RF"].values
        X = np.column_stack([np.ones(len(y))] + [f[c].values
                             for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]])
        beta, se, t = nw_ols(y, X)
        rows.append({
            "bps_per_side": bps,
            "ann_ret": net.mean() * 12,
            "sharpe_net": sharpe(net),
            "ff5_alpha_ann": beta[0] * 12,
            "ff5_alpha_t": t[0],
        })

    df = pd.DataFrame(rows)
    print()
    print(f"{'bps/side':>9} {'ann_ret':>9} {'Sharpe':>8} {'FF5 a/yr':>10} {'t(a)':>7}")
    print("-" * 50)
    for _, x in df.iterrows():
        print(f"{int(x.bps_per_side):>9} {x.ann_ret:>+8.1%} {x.sharpe_net:>+8.2f} "
              f"{x.ff5_alpha_ann:>+9.1%} {x.ff5_alpha_t:>+7.2f}")

    # break-even: highest bps where t(alpha) still > 2
    sig = df[df["ff5_alpha_t"] > 2.0]
    print()
    if len(sig) == len(df):
        print(f"  Alpha stays significant (t>2) through {BPS_GRID[-1]} bps/side -- robust to costs.")
    elif len(sig) == 0:
        print("  Alpha already insignificant at 10 bps -- fragile.")
    else:
        print(f"  Alpha significant (t>2) up to ~{int(sig['bps_per_side'].max())} bps/side; "
              f"loses significance beyond that.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
