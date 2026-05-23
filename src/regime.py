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
    required = {
        "month_end",
        "regime",
        "leverage",
        "k_per_sector",
        "long_quantile",
        "short_quantile",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in regime overlay csv: {sorted(missing)}")

    return df.sort_values("month_end").reset_index(drop=True)


def make_regime_dict(path: str | Path) -> dict[pd.Timestamp, RegimeParams]:
    df = load_regime_overlay_csv(path)
    regime_dict: dict[pd.Timestamp, RegimeParams] = {}

    for _, row in df.iterrows():
        ts = pd.Timestamp(row["month_end"])
        regime_dict[ts] = {
            "leverage": float(row["leverage"]),
            "k_per_sector": int(row["k_per_sector"]),
            "long_quantile": float(row["long_quantile"]),
            "short_quantile": float(row["short_quantile"]),
        }

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