# regime.py - Market regime detection (HMM and related models)
from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import pandas as pd


class RegimeParams(TypedDict, total=False):
    leverage: float
    long_quantile: float
    short_quantile: float
    k_per_sector: int


DEFAULT_REGIME_PARAM_MAP: dict[str, RegimeParams] = {
    "calm": {
        "leverage": 1.00,
        "k_per_sector": 5,
        "long_quantile": 0.10,
        "short_quantile": 0.10,
    },
    "normal": {
        "leverage": 0.70,
        "k_per_sector": 3,
        "long_quantile": 0.07,
        "short_quantile": 0.07,
    },
    "crisis": {
        "leverage": 0.40,
        "k_per_sector": 2,
        "long_quantile": 0.04,
        "short_quantile": 0.04,
    },
}


def load_regime_overlay_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["month_end"])
    # Required: month_end + regime + leverage. Optional: k_per_sector,
    # long_quantile, short_quantile -- the production overlay is now
    # leverage-only (the breadth/quantile lever was tested in
    # `regime_ablation_check.py` and rejected: it hurt drawdown without
    # adding alpha). See DECISIONS.md 2026-05-24.
    required = {"month_end", "regime", "leverage"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in regime overlay csv: {sorted(missing)}")

    return df.sort_values("month_end").reset_index(drop=True)


def make_regime_dict(path: str | Path) -> dict[pd.Timestamp, RegimeParams]:
    df = load_regime_overlay_csv(path)
    regime_dict: dict[pd.Timestamp, RegimeParams] = {}

    for _, row in df.iterrows():
        ts = pd.Timestamp(row["month_end"])
        params: RegimeParams = {"leverage": float(row["leverage"])}
        # Optional breadth/quantile columns -- the leverage-only canonical
        # does not populate them, but legacy CSVs may.
        for opt_int in ("k_per_sector",):
            if opt_int in df.columns and pd.notna(row.get(opt_int)):
                params[opt_int] = int(row[opt_int])
        for opt_float in ("long_quantile", "short_quantile"):
            if opt_float in df.columns and pd.notna(row.get(opt_float)):
                params[opt_float] = float(row[opt_float])
        regime_dict[ts] = params

    return regime_dict


def make_regime_fn(path: str | Path):
    """Build the date -> RegimeParams function the backtest engine consumes.

    Lookup is by **month period**, not exact timestamp. The overlay CSV is
    keyed by calendar month-ends (e.g. 2015-01-31), but the backtest
    rebalances on trading-day month-ends (2015-01-30 when the 31st is a
    weekend/holiday). An exact-date lookup silently missed ~30% of months
    and ran them at neutral params (leverage 1.0, no sector cap), so the
    regime overlay was a partial no-op. Matching on (year, month) makes
    2015-01-30 and 2015-01-31 both resolve to the January-2015 rule.
    """
    regime_dict = make_regime_dict(path)
    by_period: dict[pd.Period, RegimeParams] = {
        pd.Period(ts, freq="M"): params for ts, params in regime_dict.items()
    }

    def regime_fn(ts: pd.Timestamp) -> RegimeParams:
        return by_period.get(pd.Period(pd.Timestamp(ts), freq="M"), {})

    return regime_fn