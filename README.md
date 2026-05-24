# ml-factor-investing-2026

Systematic equity factor investing on the S&P 500: a long–short cross-sectional ML strategy with a Gaussian-mixture regime overlay that modulates leverage, evaluated under a strict walk-forward backtest with transaction costs. Methodology inspired by Gu, Kelly & Xiu (2020), *Empirical Asset Pricing via Machine Learning*.

This is a 5-week course project for a 3-person team.

## Goal

1. Build a cross-sectional return-forecasting model (linear baseline + XGBoost + a small NN).
2. Translate forecasts into a top-decile-long / bottom-decile-short, monthly-rebalanced portfolio.
3. Detect market regimes with a Gaussian mixture on macro features and use the regime to scale the gross leverage of the portfolio.
4. Backtest the whole pipeline walk-forward, with transaction costs, and produce a tear sheet (Sharpe, drawdown, turnover, IC) + a written report.

## Repo layout

```
src/                # Reusable code. No notebooks here.
  data_loader.py    # yfinance + FRED I/O, parquet caching, point-in-time universe
  factors.py        # Cross-sectional feature construction (momentum, value, size, ...)
  models.py         # Lasso / XGBoost / NN wrappers, common fit/predict interface
  backtest.py       # Walk-forward engine (the integration point - read the docstring)
  regime.py         # GMM regime detection + leverage_fn factory
  metrics.py        # Sharpe, IC, turnover, max drawdown, ...
notebooks/          # Exploration only. Anything reusable graduates into src/.
data/               # Parquet cache. Gitignored. See data/README.md.
results/            # Backtest outputs, plots, pickled models. Gitignored.
report/             # Final write-up.
DECISIONS.md        # Running log of every non-trivial design choice.
```

## Team and ownership

| Person | Workstream | Owns |
|--------|------------|------|
| A (bowenzuo119-hash) | Data & infrastructure | `data_loader.py`, `backtest.py`, `metrics.py`, CI |
| B (Nicolas) | Alpha model | `factors.py`, `models.py` |
| C (Andrea) | Risk & regime | `regime.py` and the macro side of `data_loader` |

Branch convention: `persona-*` (A), `personb-models` (Nicolas), `personc-regime` (Andrea). Never push to `main` directly; open a PR and get at least one reviewer.

## Setup

Requires Python 3.11+.

```bash
git clone https://github.com/bowenzuo119-hash/ml-factor-investing-2026.git
cd ml-factor-investing-2026
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Verify the install:

```python
import xgboost, hmmlearn, yfinance, torch
print(torch.cuda.is_available())  # CPU is fine for everything except the NN baseline
```

If you hit a `pandas_datareader` import error, make sure pandas is < 3.0 (see DECISIONS.md, entry 2026-04-23).

**macOS only — libomp deadlock during NN / XGBoost training.** On macOS, having both PyTorch's and XGBoost's bundled `libomp` loaded in one process can deadlock (the walk-forward NN baseline hangs with no error). Export this before running any training:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

Add it to your shell profile, or set it at the top of a notebook with `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` **before** importing torch/xgboost. Linux/Windows are unaffected.

## Rebuilding the data

All files under `data/` and the `*.parquet` caches in `results/` are gitignored — a fresh clone has no data. Rebuild every cache in one command:

```bash
python -m notebooks.persona.run_all_data
```

It pulls S&P 500 membership (GitHub), FRED macro, yfinance prices + dollar volume, and Person C's regime overlay from free public sources, and builds the CRSP cache + Sharadar fundamentals **if** their prerequisites are present:

* CRSP monthly prices need the vendor file `data/raw/CRSPData_1925_2022.csv` (not downloadable; shared by the course TA).
* Sharadar fundamentals need `NASDAQ_DATA_LINK_API_KEY` in `.env` (copy `.env.example`).

Steps whose prerequisite is missing are **skipped, not fatal**. Person A's methodology figures (sanity gate, universe coverage, walk-forward scheme, splice timeline) regenerate with `python -m notebooks.persona.report_figures`.

## Reproducing the broad-universe canonical

The canonical strategy runs on the **survivorship-free Sharadar broad universe** (top-2000 by market cap, PIT). Prerequisite: a Sharadar premium subscription (`NASDAQ_DATA_LINK_API_KEY` in `.env`). The subscription expires **2026-06-22**, but the raw tables are cached locally under `data/raw/sharadar/` (gitignored) once pulled, so the pipeline runs offline afterwards.

```bash
# 1. one-time bulk pull of the 8 Sharadar tables + verify
python -m notebooks.persona.pull_all_sharadar
python -m notebooks.persona.verify_sharadar_pulls

# 2. freeze the broad panels (returns + features); B3 returns validation
python -m notebooks.persona.freeze_broad_panel_sharadar
python -m notebooks.persona.freeze_broad_features_sharadar
python -m notebooks.persona.validate_sharadar_returns
python -m notebooks.personb.compute_chmom_maxret_features   # +chmom, +maxret
python -m notebooks.persona.add_mom6m_mom36m                # +mom36m (mom6m dropped, redundant)

# 3. canonical walk-forward (XGBoost, 14 features, Q-filter, PIT, k=20)
python -m notebooks.personb.24_canonical_with_chmom          # Phase 24-RT (FINAL canonical)
# python -m notebooks.personb.23g_canonical_qfiltered_orig_tune  # 13-feature predecessor / baseline

# 4. audit + cost robustness of the headline
python -m notebooks.persona.verify_phase23_headline      # FF5 alpha, dollar-neutrality
python -m notebooks.persona.cost_sensitivity_phase23     # alpha vs bps/side grid

# 5. methodology checks
python -m notebooks.persona.out_of_time_test             # static 2002-18 -> 2019-24
python -m notebooks.persona.regime_overlay_ablation_broad  # overlay with/without
python -m notebooks.persona.overlay_failure_diagnostic     # COVID-timing diagnostic
```

The **canonical is the 14-feature Phase 24-RT** run (13 base features + GKX `chmom`, retuned). `chmom` adds a small lift over the 13-feature 23g predecessor; the *further* GKX features `maxret` + `mom36m` (a 16-feature "24b" variant) were tested and **did not improve** the headline (Sharpe fell to ~0.97), so they remain in the panel for sensitivity reference but are excluded from the canonical `INCLUDE_FEATURES`.

## Reproducibility rules

* Every model gets `random_state=42`. Every sampling step gets a seed. No exceptions.
* Raw data files live in `data/` and are gitignored; the **code** that produces them is the source of truth.
* Significant choices go in `DECISIONS.md` the day they are made.

## Further reading

* Gu, Kelly & Xiu (2020), *Empirical Asset Pricing via Machine Learning*, RFS.
* Lopez de Prado (2018), *Advances in Financial Machine Learning*, ch. 11 (Backtesting).
* Nystrup, Madsen & Lindstrom (2018), *Dynamic Allocation or Diversification: A Regime-Based Approach*.
