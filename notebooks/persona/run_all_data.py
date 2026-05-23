"""run_all_data.py - One-command rebuild of every data cache the project needs.

Run with:
    python -m notebooks.persona.run_all_data

This is the reproducibility entry point. The raw data files and parquet
caches under data/ are gitignored (we version the code that produces them,
not the data), so a fresh clone has no caches. This script regenerates all
of them in dependency order, skipping gracefully when a prerequisite is
missing rather than crashing.

Prerequisites (each step says so and skips if absent):
  * CRSP monthly prices  -> needs data/raw/CRSPData_1925_2022.csv (vendor
    file, not downloadable; shared by the course TA). Everything that
    splices CRSP (returns panel) is skipped if it's missing.
  * Sharadar fundamentals -> needs NASDAQ_DATA_LINK_API_KEY in .env
    (paid subscription). Skipped if absent.
  * Everything else (S&P 500 membership, FRED macro, yfinance prices /
    dollar volume, the regime pipeline) pulls from free public sources.

Outputs (all under data/ or results/, all gitignored):
  data/raw/sp500_*.csv                       S&P 500 membership
  data/processed/crsp_monthly.parquet        CRSP cache
  data/processed/macro_daily.parquet         FRED macro
  data/processed/yfinance_monthly.parquet    yfinance OHLCV
  data/processed/yfinance_dollar_volume_monthly.parquet
  data/processed/sharadar_sf1_{ARQ,ART}.parquet
  data/processed/returns_spliced_2003_2025.parquet
  results/regime_overlay_rules.csv           Person C's regime overlay
"""

from __future__ import annotations

import subprocess
import sys
import traceback
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]


def _run(label: str, fn) -> tuple[str, str]:
    """Run one build step, return (label, status). Never raises."""
    print("\n" + "=" * 70)
    print(f"STEP: {label}")
    print("=" * 70)
    try:
        status = fn()
        return (label, status or "BUILT")
    except FileNotFoundError as exc:
        print(f"  SKIPPED — missing prerequisite: {exc}")
        return (label, "SKIPPED")
    except Exception as exc:  # noqa: BLE001 - rebuild log: report and continue
        print(f"  FAILED — {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return (label, "FAILED")


def main() -> int:
    from src.data_loader import (
        download_sp500_universe,
        load_macro,
        load_prices,
    )

    results: list[tuple[str, str]] = []

    # 1. S&P 500 membership (free GitHub source)
    results.append(_run(
        "S&P 500 membership (fja05680)",
        lambda: (download_sp500_universe(), "BUILT")[1],
    ))

    # 2. FRED macro (free, no key)
    results.append(_run(
        "FRED macro bundle",
        lambda: (load_macro(start="2003-01-01", end="2025-12-31"), "BUILT")[1],
    ))

    # 3. CRSP monthly cache (needs the vendor raw CSV)
    def _crsp() -> str:
        load_prices(start="2003-01-01", end="2022-12-31")  # triggers cache build
        return "BUILT"
    results.append(_run("CRSP monthly prices cache", _crsp))

    # 4. yfinance dollar volume (free, yfinance daily)
    results.append(_run(
        "yfinance dollar volume (Feature 4)",
        lambda: _run_module("notebooks.persona.pull_dollar_volume"),
    ))

    # 5. Sharadar fundamentals (needs API key)
    results.append(_run(
        "Sharadar SF1 fundamentals",
        lambda: _run_module("notebooks.persona.pull_fundamentals"),
    ))

    # 6. Spliced returns panel (needs CRSP cache + yfinance)
    results.append(_run(
        "Spliced returns panel 2003-2025",
        lambda: _run_module("notebooks.persona.freeze_returns_panel"),
    ))

    # 7. Person C's regime overlay pipeline (free: yfinance + FRED)
    for wk in ("week1_regime_data", "week2_regime_models", "week3_regime_finalise"):
        results.append(_run(
            f"Regime pipeline: {wk}",
            lambda wk=wk: _run_script(BASE_DIR / "notebooks" / "personc" / f"{wk}.py"),
        ))

    # --- Summary ---
    print("\n" + "=" * 70)
    print("REBUILD SUMMARY")
    print("=" * 70)
    for label, status in results:
        print(f"  [{status:>7}]  {label}")
    n_ok = sum(1 for _, s in results if s == "BUILT")
    n_skip = sum(1 for _, s in results if s == "SKIPPED")
    n_fail = sum(1 for _, s in results if s == "FAILED")
    print(f"\n{n_ok} built, {n_skip} skipped, {n_fail} failed.")
    if n_fail:
        print("Some steps FAILED — see tracebacks above.")
        return 1
    if n_skip:
        print("Some steps SKIPPED — provide the CRSP raw CSV / Nasdaq key to "
              "build them, then re-run.")
    return 0


def _run_module(module: str) -> str:
    """Run a `python -m module` step in a subprocess; return BUILT/SKIPPED."""
    proc = subprocess.run(
        [sys.executable, "-m", module],
        cwd=BASE_DIR, capture_output=True, text=True,
    )
    sys.stdout.write(proc.stdout[-2000:])  # tail of the child's log
    if proc.returncode != 0:
        # A clean "no key / no data" skip vs a real failure.
        tail = (proc.stdout + proc.stderr).lower()
        if "not set" in tail or "skipping" in tail or "no .env" in tail:
            print("  (child reported a missing prerequisite)")
            return "SKIPPED"
        raise RuntimeError(f"{module} exited {proc.returncode}")
    return "BUILT"


def _run_script(path: Path) -> str:
    """Run a standalone script in a subprocess; return BUILT/FAILED."""
    proc = subprocess.run(
        [sys.executable, str(path)],
        cwd=BASE_DIR, capture_output=True, text=True,
    )
    sys.stdout.write(proc.stdout[-1500:])
    if proc.returncode != 0:
        raise RuntimeError(f"{path.name} exited {proc.returncode}")
    return "BUILT"


if __name__ == "__main__":
    sys.exit(main())
