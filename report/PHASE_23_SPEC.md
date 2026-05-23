# Phase 23 — Broader-Universe Canonical (Sharadar rebuild)

*Person B's spec for the post-PIT-audit rebuild. Person A (Bowen) owns
the data engineering; Person B (Nicolas) owns the modeling and analysis.
Created 2026-05-23; first draft.*

## Goal

Replace Phase 22 (S&P-500-only, no factor-adjusted alpha) with a
GKX-comparable broader-universe canonical that can recover real
cross-sectional alpha if it exists. Test the academic claim that ML
factor strategies' alpha comes from the small/mid-cap tail by giving the
model access to that tail.

## Data dependencies (Bowen ships first)

Phase 23 cannot start until ALL of these exist:

```
data/raw/sharadar/sf1_AR_arq.parquet       # AS-REPORTED quarterly fundamentals
data/raw/sharadar/sf1_AR_art.parquet       # AS-REPORTED TTM fundamentals
data/raw/sharadar/sf1_MR_arq.parquet       # RESTATED (for sensitivity check)
data/raw/sharadar/daily.parquet            # daily marketcap, ratios per ticker
data/raw/sharadar/tickers.parquet          # 17,689 tickers + delisted flag + permaticker
data/raw/sharadar/sp500.parquet            # PIT S&P 500 membership history
data/raw/sharadar/actions.parquet          # dividends + splits + M&A + delistings
```

And these new functions in `src/data_loader.py`:

```python
def load_universe_at(
    asof: pd.Timestamp,
    min_marketcap: float = 1e9,
    require_common_stock: bool = True,
    exchanges: tuple[str, ...] = ("NYSE", "NASDAQ", "ARCA", "BATS"),
) -> pd.DataFrame: ...
# Returns: DataFrame[ticker, permaticker, sector, marketcap]
# Used as: universe_fn = lambda d: set(load_universe_at(d)['ticker'])

def compute_monthly_returns_sharadar(
    start: str, end: str,
    tickers: list[str] | None = None,
) -> pd.DataFrame: ...
# Returns: wide-format DataFrame indexed by month-end, columns = tickers,
# values = monthly total returns (split + dividend adjusted)
```

And:
```
data/processed/returns_broad_sharadar_2002_2024.parquet
```
shape ≈ 2500 columns × 273 month-end rows.

## Phase 23 driver structure (`notebooks/personb/23_canonical_broad_sharadar.py`)

```python
from src.backtest import run_walk_forward_backtest
from src.data_loader import load_universe_at
from src.factors import build_feature_panel
from src.models import LassoModel, NNModel, XGBoostModel

START = "2002-04-01"
END = "2024-12-31"
TRAIN_WINDOW = 120
TEST_WINDOW = 12
TRANSACTION_COST_BPS = 10.0
K_PER_SECTOR = 5
TARGET_KIND = "sector_relative"

# Broader-universe panel from Bowen
PANEL_FILE = "data/processed/returns_broad_sharadar_2002_2024.parquet"

# Retuned hyperparameters from Phase 23a (run after panel lands).
# Placeholder values - replaced with Optuna results.
RETUNED_XGB_PARAMS = {...}
RETUNED_LASSO_ALPHA = ...
RETUNED_NN_PARAMS = {...}

# Universe filter: PIT, broader (mcap > $1B)
def universe_at(date):
    return set(load_universe_at(date, min_marketcap=1e9)['ticker'])

# Sector map: derive from features (SIC fallback handles delisted)
def make_sector_map(features):
    return features.reset_index().groupby('ticker')['sector'].first().to_dict()

# Three models with retuned params
model_factories = [
    ("Lasso",   lambda: LassoModel(alpha=RETUNED_LASSO_ALPHA, target_kind=TARGET_KIND)),
    ("XGBoost", lambda: XGBoostModel(target_kind=TARGET_KIND, **RETUNED_XGB_PARAMS)),
    ("NN",      lambda: NNModel(target_kind=TARGET_KIND, **RETUNED_NN_PARAMS)),
]

# Run walk-forward with broader universe
for name, factory in model_factories:
    res = run_walk_forward_backtest(
        returns=returns_wide,
        features=features,
        model=factory(),
        train_window=TRAIN_WINDOW,
        test_window=TEST_WINDOW,
        transaction_cost_bps=TRANSACTION_COST_BPS,
        regime_fn=lambda d: {"k_per_sector": K_PER_SECTOR},
        sector_map=make_sector_map(features),
        eligible_universe_fn=universe_at,   # broader PIT
        apply_pit_to_training=True,         # strict on broader universe
    )
```

## Phase 23a (Optuna retune, runs before driver)

`notebooks/personb/23a_retune_broad.py` — adapt Phase 19's structure.

Differences from Phase 19:
- Panel: `returns_broad_sharadar_2002_2024.parquet` (broader)
- Universe filter at train+val: `load_universe_at(d, min_marketcap=1e9)`
- Search XGBoost, Lasso, NN — same hyperparameter spaces as Phase 19
- 60 trials per model, R² on 2017-2018 val as objective
- Run NN with a watchdog or smaller batches to avoid the Phase 19 hang

Output: `results/23a_retune_broad/best_params.json` consumed by Phase 23.

## Phase 24 (regime overlay applied)

Same as Phase 23 but with `regime_fn = make_regime_fn('results/regime_overlay_rules.csv')`
(Bowen's overlay). Should run in parallel for the ablation table.

## Phase 25 (statistical robustness)

Re-run Phase 7's bootstrap + DSR + FF5 on Phase 23 results.
Repoint `PHASE_DIR` to `results/23_canonical_broad_sharadar/`.

Specifically check:
- Long-OOS FF5 alpha t-stat — IS the broader universe alpha significant?
- Mkt-RF beta — should be much smaller than Phase 22's +0.30 (we hope)
- HML / SMB loadings — broader universe should reduce these
- DSR with the trial list updated for 9-10 phases tested

## AR-vs-MR sensitivity (Phase 26 — methodology win)

Run Phase 23 twice — once using SF1 AR (as-reported, PIT-correct), once
using SF1 MR (restated). The Sharpe difference quantifies look-ahead.
Per Bowen's suggestion: half-page in methodology section showing the
project understands restatements.

```
Phase 23 (AR): Sharpe = X.XX, alpha = Y.YY%/yr (t=Z.ZZ)
Phase 26 (MR): Sharpe = X.XX, alpha = Y.YY%/yr (t=Z.ZZ)
Δ: <1% Sharpe difference => the model is genuinely PIT-correct
   >5% difference => look-ahead from restatements is material
```

## Expected outcomes (priors before running)

| Universe scale | Plausible long-OOS Sharpe range | Plausible FF5 alpha t-stat |
|---|---|---|
| Phase 22 (S&P 500 only, ~500-1000 names) | -0.3 to +0.4 | -1 to +0.5 (we got +0.31 / t=-0.4) |
| Phase 23 (broader, ~2000-3000 names) | +0.5 to +1.2 | +1.5 to +2.5 |
| GKX 2020 reference (~3000-6000 names) | +1.5 to +2.0 | +2.0 to +3.0 |

If Phase 23 lands in the predicted range, the project's headline becomes
"ML factor strategy on Russell-1500-equivalent universe with significant
FF5 alpha." If it lands closer to Phase 22 (no significant alpha), the
honest finding is "ML factor strategies don't have alpha at our universe
scale even after broadening."

Both outcomes are publishable. The first is a positive finding; the
second is the responsible negative finding that's better than reporting
the leaky +1.5.

## Timeline (counting from 2026-05-23)

- **Sat-Sun (24-25 May)**: Bowen bulk pull + new loaders + broader panel + sanity gate
- **Mon (26 May)**: Person B Phase 23a Optuna retune
- **Tue (27 May)**: Person B Phase 23/24/25 canonical runs + FF5 + DSR
- **Wed (28 May)**: AR-vs-MR sensitivity check, REPORT.md headline numbers, presentation plots regenerated
- **Thu-Fri (29-30 May)**: report polishing, methodology write-up of the audit + rebuild
- **Sat (31 May)**: final review and submission prep

Subscription expires **22 June 2026** — bulk pull MUST be done before then.

## Hand-off criteria (Person B starts when these are all true)

1. ✅ All 7 raw Sharadar parquets exist under `data/raw/sharadar/`
2. ✅ `src/data_loader.py::load_universe_at` implemented + smoke-tested
3. ✅ `src/data_loader.py::compute_monthly_returns_sharadar` implemented + validated vs yfinance (median per-ticker correlation > 0.99)
4. ✅ `data/processed/returns_broad_sharadar_2002_2024.parquet` exists with ~2000-3000 columns × 273 rows
5. ✅ `python -m src.sanity` passes Random/Oracle/Uniform on the new panel
6. ✅ Bowen commits + pushes with message like `persona: broad Sharadar panel + universe helper + sanity passing`

Once all six are true: Person B runs Phase 23a-26 and reports results to the team.
