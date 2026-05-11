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
| B | Risk & regime | `regime.py` and the macro side of `data_loader` |
| C | Alpha model | `factors.py`, `models.py` |

Branch convention: `persona-data-pipeline` (A), `personb-regime` or current `personc-regime` (Nicolas), `personc-models` (Andrea). Never push to `main` directly; open a PR and get at least one reviewer.

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

## Reproducibility rules

* Every model gets `random_state=42`. Every sampling step gets a seed. No exceptions.
* Raw data files live in `data/` and are gitignored; the **code** that produces them is the source of truth.
* Significant choices go in `DECISIONS.md` the day they are made.

## Further reading

* Gu, Kelly & Xiu (2020), *Empirical Asset Pricing via Machine Learning*, RFS.
* Lopez de Prado (2018), *Advances in Financial Machine Learning*, ch. 11 (Backtesting).
* Nystrup, Madsen & Lindstrom (2018), *Dynamic Allocation or Diversification: A Regime-Based Approach*.
