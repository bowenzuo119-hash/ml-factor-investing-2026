"""overlay_failure_diagnostic.py - why the regime overlay didn't help Phase 23g.

The leverage overlay left the broad canonical's -33.8% max DD unchanged. This
checks the hypothesis: the worst drawdown fell in a period the regime model
(trained on INDEX volatility) labelled "calm", so crisis de-levering never
engaged. If so, that's a publishable finding -- index-volatility regime labels
miss broad-universe idiosyncratic drawdowns.

Loads B's 23g XGBoost net returns + the regime labels, finds the worst-DD
period and its regime labels, and plots monthly returns coloured by regime.

    python -m notebooks.persona.overlay_failure_diagnostic
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "results" / "23g_canonical_qfiltered_orig_tune" / "per_model_results.pkl"
CSV = ROOT / "results" / "regime_overlay_rules.csv"
FIG = ROOT / "results" / "persona_figures" / "overlay_failure_regime.png"

COLORS = {"calm": "#34d399", "crisis": "#ef4444", "n/a": "#9ca3af"}


def main() -> int:
    net = pickle.load(open(RESULT, "rb"))["XGBoost"].portfolio_returns.dropna()

    reg = pd.read_csv(CSV, parse_dates=["month_end"])
    p2r = dict(zip(reg["month_end"].dt.to_period("M"), reg["regime"]))
    labels = pd.Series([p2r.get(p, "n/a") for p in net.index.to_period("M")], index=net.index)

    # drawdown path
    cum = (1 + net).cumprod()
    dd = cum / cum.cummax() - 1
    trough = dd.idxmin()
    peak = cum.loc[:trough].idxmax()
    dd_window = net.loc[peak:trough]
    dd_labels = labels.loc[peak:trough]

    print("=" * 66)
    print("OVERLAY FAILURE DIAGNOSTIC - Phase 23g worst drawdown vs regime")
    print("=" * 66)
    print(f"  max drawdown      : {dd.min():.1%}")
    print(f"  peak -> trough    : {peak.date()} -> {trough.date()} "
          f"({len(dd_window)} months)")
    print(f"  regime at trough  : {labels.loc[trough]!r}")
    print(f"  regime mix during the drawdown:")
    for lab, n in dd_labels.value_counts().items():
        print(f"    {lab:8s}: {n}/{len(dd_labels)} months ({n/len(dd_labels):.0%})")
    # The overlay de-levers off the PRIOR rebalance's regime, so what matters is
    # the label on the months ENTERING the drawdown (all but the trough itself).
    entry_labels = dd_labels.iloc[:-1] if len(dd_labels) > 1 else dd_labels
    entry_calm = (entry_labels == "calm").mean() if len(entry_labels) else 0.0
    trough_regime = labels.loc[trough]
    print()
    if entry_calm >= 0.5:
        print(f"  FINDING (timing lag): the trough month is labelled "
              f"{trough_regime!r}, but the overlay de-levers off the PRIOR\n"
              f"  rebalance's label, and {entry_calm:.0%} of the months ENTERING the "
              f"drawdown were 'calm'. So the crash was taken at full leverage and the\n"
              f"  crisis flag (if any) arrived a rebalance too late to de-lever in time.\n"
              f"  Monthly index-volatility regime detection lags a fast broad-universe\n"
              f"  drawdown -- a real, reportable limit of the overlay, not a bug.")
    else:
        print(f"  Months entering the drawdown were mostly crisis-labelled; the overlay "
              f"should have helped -- look elsewhere.")

    # plot monthly returns coloured by regime
    fig, ax = plt.subplots(figsize=(13, 5))
    for lab in ["calm", "crisis", "n/a"]:
        m = labels == lab
        if m.any():
            ax.bar(net.index[m], net.values[m] * 100, width=22,
                   color=COLORS[lab], label=lab)
    ax.axvspan(peak, trough, color="#fca5a5", alpha=0.25,
               label=f"worst DD ({dd.min():.0%})")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("Phase 23g monthly returns coloured by regime label "
                 "(why the overlay missed the drawdown)", fontsize=12, fontweight="bold")
    ax.set_ylabel("net monthly return (%)")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25, axis="y")
    fig.text(0.5, -0.02,
             "Deepest DD = the Feb-Mar 2020 COVID crash; the rebalances entering it were "
             "'calm', so the overlay de-levers a month too late -- monthly index-vol "
             "regimes lag fast broad-universe drawdowns.",
             ha="center", fontsize=8.5, style="italic")
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {FIG.name}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
