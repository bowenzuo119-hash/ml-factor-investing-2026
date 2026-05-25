# Person B Deep Dive — Everything About the Alpha Model

> **Reader:** Nicolas (Person B) — for self-review, Q&A prep, and demonstrating mastery to the examiner.
>
> **Style:** every concept explained in plain English first, then made precise. Read top-to-bottom, or skim the headers.

---

## Part 1 — What I Built (The 10-Second Pitch)

I built the **alpha model** — the part of the project that decides which stocks to long and short each month.

It works like this:
1. Every month, I look at ~4,400 US stocks
2. I describe each one with **14 numbers** (its "features" — things like momentum, volatility, valuation)
3. I feed those 14 numbers into a **machine learning model** (XGBoost)
4. The model spits out one number per stock: "predicted excess return next month"
5. I sort the stocks by that prediction within each sector
6. I **buy the top 20 per sector** (longs) and **short-sell the bottom 20 per sector** (shorts)
7. I hold for one month, then repeat

That's the whole strategy in 7 steps. Everything below is just detail on each of those steps.

---

## Part 2 — The 14 Features (What They Are, Why They Matter)

### Why 14 features?

Because **Gu, Kelly & Xiu (2020)** — the academic paper our project replicates — used a similar feature set. We picked their **top 13 by importance**, then added one more (`chmom`) that's their #4-ranked feature. We tested adding two more (`maxret` and `mom36m`) but the model got worse, so we stopped at 14.

### The 14 features, grouped by type

**1. Price-trend features (5 of them):**

| Feature | Plain English | Why it matters |
|---|---|---|
| `mom` | 12-month return, skipping last month (Jegadeesh-Titman 1993) | "Stocks that have been going up tend to keep going up" |
| `rev` | Last month's return (sign-flipped) | "Stocks that just spiked tend to mean-revert" |
| `mvol` | 12-month volatility | "Volatile stocks behave differently than calm ones" |
| `ivol` | Idiosyncratic volatility (residual from CAPM regression) | "Volatility that's NOT explained by market beta" |
| `chmom` | Change in 6-month momentum (recent half minus older half) | "Acceleration of momentum — is the trend strengthening or weakening?" |

**2. Liquidity features (2):**

| Feature | Plain English |
|---|---|
| `log_mktcap` | Log of market cap | "How big is this company?" |
| `log_dvol` | Log of dollar trading volume | "How heavily traded is it?" |

**3. Value features (2):**

| Feature | Plain English |
|---|---|
| `bm` | Book-to-market (book value ÷ market value) | "Is the stock 'cheap' relative to its accounting value?" |
| `ep` | Earnings-to-price (earnings ÷ price) | "How much profit do you get per dollar of share price?" |

**4. Quality / investment features (5):**

| Feature | Plain English |
|---|---|
| `roe` | Return on equity | "How profitable is the company per dollar of shareholder capital?" |
| `roa` | Return on assets | "How profitable is the company per dollar of assets?" |
| `de` | Debt-to-equity | "How leveraged is the balance sheet?" |
| `asset_growth` | Year-over-year change in total assets | "Is the company aggressively investing?" |
| `accruals` | (Net income − operating cash flow) ÷ total assets | "Are earnings real cash or accounting magic?" |

### How each feature becomes a number

For each stock on each rebalance date, we compute these 14 features using only data observable at that moment (no look-ahead). Then for the model to compare stocks fairly:

- All features are **cross-sectionally ranked** within each date (so a value of "0.8" means "this stock's feature is in the 80th percentile that month relative to all other stocks")

This step is important — it normalises across time. A stock with B/M = 2.0 in 2020 vs 2.0 in 2024 might mean very different things; ranking by percentile each month removes that issue.

### What we DIDN'T include and why

- `maxret` (max daily return last month — GKX rank #5): tested in Phase 24b, made the model worse
- `mom36m` (long-term reversal — GKX rank #2): same — tested, worse
- Macro features (VIX, term spread, etc.): these go in **Person C's regime overlay**, not in stock-picking
- News sentiment / alt data: out of scope, would require new data sources

---

## Part 3 — The 3 Models (Lasso, XGBoost, NN)

### Why three models?

Because we need to **prove that XGBoost is the right choice** — not just assume it. The course rule is **always compare against a baseline.** So we built three models with the **same interface** (you can swap them in the backtest) and compared them rigorously.

### Model 1: Lasso (the linear baseline)

**What it is:** A linear regression with a penalty for having too many active features (L1 regularization). The model learns weights for each feature and predicts return as:

```
predicted_return = w_1 × momentum + w_2 × reversal + w_3 × volatility + ... + w_14 × accruals
```

**Why we include it:**
- Cheap, fast, interpretable
- A great sanity check: if the more complex models can't beat Lasso, they're not adding value
- The L1 penalty automatically picks the most useful features (shrinks weak ones to zero)

**Result:** Sharpe +0.71 (Lasso) vs +1.14 (XGBoost). Lasso is worse — but not terrible. Confirms there IS a linear signal in the data, but the non-linear model captures more.

### Model 2: XGBoost (the canonical)

**What it is:** Gradient-boosted decision trees. Imagine asking many short, simple questions:

- "Is momentum > 0.7?" → split stocks into two groups
- For each group, ask another question: "Is volatility < 0.3?" → split again
- After 5-6 such splits, you have a small "decision tree" that predicts return based on which leaf the stock ends up in

XGBoost builds **many such trees** (~50–800) where each new tree corrects mistakes from the previous one (this is "boosting"). The final prediction is a weighted average of all the trees.

**Why it works well here:**
- **Captures non-linear interactions**: "high momentum + low volatility" is treated differently from just "high momentum"
- **Handles outliers gracefully**: trees use rank-based splits, not raw values
- **Doesn't need feature scaling**: the tree splits on relative ordering, not magnitudes
- **Lots of regularization knobs** (max depth, learning rate, subsample, etc.) so you can tune it to avoid overfitting

**Result:** Sharpe +1.14, FF5 alpha +18.7%/yr at t=+6.85. **Best model.**

### Model 3: NN (Neural Network — the secondary)

**What it is:** A small 2-layer feed-forward neural network with dropout regularization. Each neuron takes a weighted sum of the inputs, applies a non-linear function (ReLU), passes it forward.

**Why it underperformed:**
- Our effective sample size is too small (~250 monthly observations, ~5,000 names but with massive cross-sectional noise)
- NNs typically need millions of training examples to shine
- Even with dropout and small architecture, it overfits the noise
- The cross-sectional signal isn't deep enough to need a deep model

**Result:** Sharpe +0.62 — worse than even Lasso. **Confirms NN is the wrong tool for this data regime.**

### How we know XGBoost is "really" the best — the Diebold-Mariano test

The Diebold-Mariano (DM) test is a **formal statistical test** for "is model A's prediction error reliably smaller than model B's?"

**How it works in plain English:**
1. For each rebalance date, compute the squared prediction error of model A and model B
2. Take the difference: `d_t = error_A_t² − error_B_t²`
3. If model B is genuinely better, `d_t` should be reliably positive on average
4. Test that statistically (with Newey-West corrections for autocorrelation)

**Our result:**
- XGBoost vs Lasso on MSE: p < 0.01 (highly significant)
- XGBoost vs NN on MSE: p < 0.05 (significant)
- XGBoost wins on every window (full-OOS, long-OOS, test-OOS)

This is **important** — it means we're not just eyeballing "XGBoost is better." We have a statistical test that says "the chance this is by luck is < 1%."

---

## Part 4 — Hyperparameter Tuning (Optuna + TPE)

### What's a hyperparameter?

XGBoost has knobs you can tune: `n_estimators` (how many trees), `max_depth` (how deep each tree), `learning_rate` (how fast it learns), `subsample` (% of data each tree sees), etc. These are **hyperparameters** — knobs we set, not parameters the model learns.

There's no formula for the best values; we have to search.

### What's Optuna with TPE?

**Optuna** is a Python library for hyperparameter optimization. It uses an algorithm called **TPE — Tree-structured Parzen Estimator**, which is a smart Bayesian-style search.

**How it works in plain English:**

1. **Trial 1**: pick random hyperparameter values. Run the model. Measure quality (we use OOS R² on a held-out validation window).
2. **Trial 2**: pick more random values. Run. Measure.
3. **After ~10 trials**: TPE looks at what worked and what didn't. It builds a probabilistic model: "values near (lr=0.02, depth=6) gave good results — let's try more of those."
4. **Subsequent trials**: TPE samples from regions that look promising. It's like a smart explorer — focuses on areas that already paid off, but still tries random new spots.
5. **After 60 trials**: pick the best-performing combination as the final hyperparameters.

### Why TPE over grid search?

A grid search would test, say, 5 values × 5 × 5 × 5 = 625 combinations to be thorough. That's slow and most combinations are wasted. TPE finds equally good (or better) hyperparameters in 60 trials by focusing on the promising regions.

### Critical detail: where we tune

We tune on the **validation window (2017-2018)** and evaluate on the **test window (2019-2024)**. The validation window is INSIDE the training period — it doesn't touch the test data. This avoids "tuning on the future."

```
2002 ─── 2016 ─── 2018 ─── 2024
   Training    Validation   Test
   (build      (pick hyper- (final
    model)      params)      evaluation)
```

### What we tuned

For XGBoost's Phase 24a Optuna study:
- `n_estimators`: 50 → 800 (in steps of 50)
- `max_depth`: 2 → 7
- `learning_rate`: 0.005 → 0.3 (log scale)
- `subsample`: 0.5 → 1.0
- `colsample_bytree`: 0.5 → 1.0
- `min_child_weight`: 1 → 30
- `reg_alpha`, `reg_lambda`: regularization strengths

**Result of best params (Phase 24a, 60 trials):**
- n_estimators=50, max_depth=6, lr=0.023, subsample=0.52, colsample=0.74, min_child_weight=22, reg_alpha=1.00, reg_lambda=9.49
- Validation R² = +0.00545 (compared to a random predictor's 0)

That R² looks tiny (0.5% of variance), but cross-sectional return prediction is INHERENTLY noisy — anything positive and consistent is meaningful, and our 0.005 is comparable to GKX's published numbers.

---

## Part 5 — The Target (What We're Trying to Predict)

### Naive idea: predict next-month return

**Problem with this:** Stocks in the same sector tend to move together. A semiconductor stock prediction is heavily influenced by how all semiconductors are doing that month. The model would learn "semiconductors going up → predict positive" which is just a SECTOR signal, not stock-picking skill.

### Our solution: sector-relative return

We **demean** the target by sector:

```
target_for_stock_i = actual_return_i − average_return_of_stock_i's_sector
```

So the target is **excess return relative to sector peers**, not raw return.

### Why this matters

- **Removes the sector signal** (which would otherwise dominate everything)
- **Forces the model to learn stock-picking, not sector-picking**
- **GKX Layer-2 framework** — this is the standard refinement they recommend

### In code (`src/models.py`)

```python
class XGBoostModel:
    def __init__(self, target_kind="sector_relative", ...):
        ...

    def fit(self, X, y):
        if self.target_kind == "sector_relative":
            # Subtract per-(date, sector) mean from y before training
            y = y - y.groupby(["date", "sector"]).transform("mean")
        ...
```

When prediction time comes, we just use the raw predicted values — sorting by them and picking top-k/bottom-k naturally selects stocks that are predicted to outperform their sector.

---

## Part 6 — Portfolio Construction (k=20 per Sector, Dollar-Neutral)

### The selection rule

After the model gives us a prediction for each stock:

1. **Group by sector** (11 GICS sectors: Tech, Healthcare, Financials, etc.)
2. **Within each sector, rank by prediction**
3. **Take the top 20 stocks** (predicted to outperform their sector most) → longs
4. **Take the bottom 20 stocks** (predicted to underperform their sector most) → shorts

So we end up with ~20 longs and ~20 shorts per sector × 11 sectors = ~440 positions total.

### Equal-weight + dollar-neutral

- **Equal-weight**: each long gets +1/220 = +0.45% of NAV; each short gets −1/220 = −0.45% of NAV
- **Dollar-neutral**: total longs = +$1 of NAV, total shorts = −$1 of NAV → net exposure ≈ $0
- **Gross exposure**: 200% (i.e., long $1 + short $1 = $2 of gross book per $1 of capital)

### Why k=20 specifically?

We ran a **dense k-sweep** (Phase 27): tested k = 1, 2, 3, ..., 30, 35, 40, ..., 100.

| k | Positions per rebalance | Full-OOS Sharpe |
|---|---|---|
| 1 | 22 | +0.56 (too concentrated, turnover kills it) |
| 5 | 110 | +1.01 |
| 10 | 220 | +1.16 |
| **16 (peak)** | **352** | **+1.17** |
| **20 (canonical)** | **440** | **+1.15** |
| 25 | 550 | +1.11 |
| 50 | 1,100 | +0.93 |
| 100 | 2,120 | +0.78 (over-diversified, signal washes out) |

There's a **flat plateau between k=10 and k=20**. We tested if k=20 is "really" the best with bootstrap confidence intervals (Phase 27b): **k=20's Sharpe falls inside the 90% CI of every other k in [10, 20]** — so the differences are not statistically significant. **k=20 is empirically defensible.**

We picked 20 ex-ante (before running the sweep) because it's a round number with good diversification (~440 positions); the sweep just confirmed that choice was in the optimum region.

---

## Part 7 — Walk-Forward Evaluation (How We Test the Model)

### The mistake we DIDN'T make

Most naive backtests do this:
1. Train model on 2002-2018
2. Test on 2019-2024
3. Report the test Sharpe

**Problem:** the model has seen 17 years of data. By 2024, your "2002 training" is 22 years old — markets have changed dramatically. Also, the model's hyperparameters were tuned ONCE in 2018, but markets change.

### What we did instead: walk-forward expanding-window CV

For each rebalance month from Feb 2012 to Dec 2024:
1. **Train** the model on the prior 120 months (10 years) only
2. **Predict** the next month
3. **Realize** that prediction's P&L
4. **Move forward 12 months**, retrain with fresh data
5. Repeat

This is more realistic because:
- The model is always trained on **recent** data, not 20-year-old data
- The training window **slides forward**, like a real-time deployment would work
- Each test prediction is on data the model has **never seen**

### Block-gated refit

We don't refit the model every single month (expensive). We refit at every **12-month block boundary**, then use that fit for 12 monthly predictions. This is faster and matches standard industry practice.

### The result: 155 monthly test predictions

From Feb 2012 to Dec 2024 = 155 months of genuinely out-of-sample backtests. Every prediction was made by a model that had NEVER seen that month's data. **This is what makes the +1.15 Sharpe defensible.**

---

## Part 8 — Headline Numbers (And What They Mean)

### The headline table

| Window | Sharpe | Annual return | FF5 alpha/yr | t-stat |
|---|---|---|---|---|
| **Full-OOS 2012-2024** | **+1.15** | +34.7%/yr | **+18.73%** | **+6.85** |
| Long-OOS 2015-2024 | +0.97 | +31.9%/yr | +19.10% | +6.00 |
| Test-OOS 2019-2024 | +1.00 | +39.4%/yr | +21.17% | +5.00 |

### Plain-English meaning of each metric

**Sharpe ratio (+1.15)**: Risk-adjusted return. (Annual return − cash) / annual volatility. A Sharpe of 1 means you earn 1 unit of return for each unit of risk taken. Anything above 1 is "good." Above 2 is "exceptional." Our +1.15 is solidly good — but not implausibly high (which would be suspicious).

**Annual return (+34.7%/yr)**: What $1 grew at, per year, after costs. Compounded over 13 years = +4,600% cumulative (from $1 to $47).

**FF5 alpha (+18.73%/yr at t = +6.85)**: After running our return through a Fama-French 5-factor regression and stripping out everything the FF5 factors can explain (market, size, value, profitability, investment), there's **18.73%/yr left over that the FF5 cannot explain**. That's our genuine cross-sectional skill. The t = +6.85 means this is **statistically significant beyond any reasonable doubt** (p < 0.001).

### The S&P comparison

| Strategy | CAGR | $1 grew to (13 yrs) | Sharpe |
|---|---|---|---|
| S&P 500 passive | +14.3%/yr | $5.63 | +0.99 |
| Our model (10 bps cost) | +34.7%/yr | $47.00 | **+1.14** |
| β-hedged pure alpha (uncorrelated) | +16.5%/yr | $7.13 | +0.70 |

**Read:**
- We beat the S&P by **+0.15 Sharpe** (+1.14 vs +0.99) — modest but real
- We earn 2.4× the S&P's annual return — but half of that is from leveraged market exposure (β=+1.5)
- The β-hedged pure alpha is +614% cumulative (1.26× the S&P's +463%) — the deployable headline

---

## Part 9 — Robustness Battery (Proving It's Not Luck)

### The seven rigor checks

#### 1. Sanity gate (3/3 PASS)

We have three "obviously good/obviously bad" predictors:
- **RandomModel**: random predictions → Sharpe should be ~0 (✓ we get −0.51)
- **OracleModel**: cheats and looks at next-month return → Sharpe should be huge (✓ +99)
- **UniformModel**: predicts the same value for all stocks → portfolio should be empty/flat (✓ flat)

If the engine were buggy, these wouldn't pass. They all pass → engine is clean.

#### 2. Feature-shuffle placebo

Run the exact same recipe but **randomly permute which feature vector goes to which ticker** within each month. If the model's "signal" is real, this should destroy it.

**Result:** Sharpe collapses from +1.15 → −0.94 (mean of 2 seeds). The 2.1-Sharpe drop confirms the model NEEDS real features to make money. **This rules out engine-side or cost-side leakage.**

#### 3. Block bootstrap on Sharpe

Resample contiguous 6-month blocks (with replacement) from our 155-month return stream, recompute Sharpe, repeat 10,000 times. This gives us a **distribution** of plausible Sharpe values.

**Result:** P(true Sharpe ≤ 0) = 0.0002 long-OOS. The 90% CI is [+0.54, +1.44] — far above zero.

#### 4. Deflated Sharpe Ratio (DSR)

When you try 25 different model configurations, the best one's Sharpe is inflated by **multiple testing**. The DSR (Bailey & López de Prado 2014) deflates the headline by the expected-max-under-the-null.

**Result:** DSR = 0.85-0.88 at N=25. This means: even after penalizing for trying 25 configs, there's an 85-88% posterior probability that our true Sharpe exceeds what the BEST of 25 random strategies would have given. **Comfortably passes.**

#### 5. Carhart 6-factor momentum control

The natural skeptic asks: "Is your alpha just the momentum premium repackaged?" We run FF5 + UMD (Carhart 4-factor + size + value) and check what happens to alpha.

**Result:** Alpha actually **RISES** to +20.1%/yr at t=+7.40, with UMD loading = −0.43 (negative!). Our strategy is **momentum-averse** — meaning when you add the momentum factor as a control, alpha gets *bigger*, not smaller. The +18.7% is NOT repackaged momentum.

#### 6. Cost-grid stress test

At 10 bps/side: Sharpe +1.14
At 15 bps/side: Sharpe +1.07 (still > S&P's +0.99)
At 25 bps/side: Sharpe ~+0.99 (ties S&P)
At 30 bps/side: Sharpe +0.95 (below S&P by 0.04)
At 50 bps/side: Sharpe +0.55

We beat the S&P at any realistic moderate-AUM cost (10-15 bps/side per AQR research).

#### 7. Dense k-sweep + plateau-zoom bootstrap

Already discussed in Part 6. k=20 is statistically indistinguishable from every k in [10, 20].

#### Bonus check: single-name leave-one-out (Phase 26)

Top-1 contributor (LSCG) moves Sharpe by +0.089 when dropped. Suggestive of single-name fragility but underpowered (n=11). **Honest caveat** in §6.

---

## Part 10 — The Audit (Bugs I Caught and Fixed)

### Bug 1: Q-filter dropping legitimate stocks

**The bug:** Our bankruptcy filter was `len(ticker) >= 4 and ticker.endswith("Q")`. Sharadar appends "Q" to delisted bankruptcy tickers (LEHMQ, ENRNQ, SIVBQ). But this rule **also wrongly dropped NDAQ (Nasdaq Inc., alive) and IONQ (IonQ Inc., alive)**.

**The fix:** Add a SHARADAR table check: `endswith("Q") AND isdelisted == "Y"`. Bowen implemented `is_bankruptcy_ticker()` in `src/data_loader.py`; I migrated all 8 callsites in my phase scripts.

**Impact:** Full-OOS Sharpe moved +1.04 → +1.15 (Bowen decomposed: 2/3 from NDAQ being a real large-cap we'd wrongly excluded, 1/3 from IONQ — a 2021 SPAC that's the marginal fragility concern).

### Bug 2: INCLUDE_FEATURES subset not applied

**The bug:** My canonical driver (`24_canonical_with_chmom.py`) read the features parquet file but didn't restrict to the declared 14-feature list. Bowen added `maxret` and `mom36m` to the SAME parquet for a Phase 24b test. The canonical SILENTLY became a 16-feature run.

**The result:** What was being labeled as "Phase 24-RT (the canonical)" was actually the Phase 24b config we'd separately tested and REJECTED.

**The fix:** Add one line: `features = features[list(INCLUDE_FEATURES) + ["sector"]]` to enforce the declared feature set.

**Impact:** Sharpe +1.01 → +1.15 (back to the correct 14-feature canonical).

### Why these two bugs partially cancelled

Both bugs lowered the headline:
- Q-filter dropped legit names (NDAQ +IONQ contributed positively to true alpha, so excluding them lowered Sharpe by ~0.12)
- INCLUDE_FEATURES forced 16-feature variant which had ~0.10 lower Sharpe than 14-feature

The buggy pkl showed +1.08; the corrected pkl shows +1.15. Both bugs partially cancelled, which is why we didn't catch them until late.

### Why this matters for the project

- The audit story IS the methodological contribution (per Bowen and per the report's §7 conclusion)
- Showing we caught our own bugs DEMONSTRATES INTELLECTUAL HONESTY
- The path +1.49 (leaky) → −0.31 (audit applied to S&P) → +1.15 (honest, broad universe, bugs fixed) is what we tell the examiner

---

## Part 11 — Honest Counterweights (What Doesn't Work / Caveats)

### 1. Not market-neutral

Realized Mkt-β = +1.5. The strategy IS leveraged to the market. Long leg makes +37.9%/yr alone; short leg makes −2%/yr (it's a near-zero-P&L hedge). **~55% of the headline return is leveraged market exposure; ~45% is genuine cross-sectional skill.**

### 2. Alpha lives in the down-cap tail

On the strict rolling top-2,000 (large-cap end of the universe), FF5 α collapses to **+1.8%/yr at t = 0.96 (not significant)**. SMB-β also collapses from +1.26 to +0.15.

**Meaning:** the headline alpha is a small/mid-cap effect. Confirms Gu-Kelly-Xiu (2020) §IV.D: ML alpha lives where institutional money does NOT trade (because capacity-constrained at deployable AUM).

### 3. Capacity-constrained at large AUM

175% monthly turnover × ~440 small-cap positions = ~770% NAV throughput per month. At $100M AUM, that's $770M of monthly small-cap notional. Real-world impact would push costs above 30 bps/side at large scale.

The Sharpe edge survives only at moderate AUM (~$100M-$1B small-cap-tilted).

### 4. Single-name fragility (underpowered, but real)

LSCG alone moves Sharpe by +0.089. IONQ alone (the SPAC) moves Sharpe by +0.042. We have measurable name-fragility in the down-cap tail.

### 5. Single OOS path

13 years of monthly data on US equities. We did not test sub-monthly rebalancing, intraday, or post-2024 data. The DSR penalizes for trial count but it's still one historical sample.

### 6. Monthly-frequency regime overlay can't catch fast crashes

Andrea's HMM-based regime overlay works on strict-S&P (DD −25.5% → −19.9%) but is net-zero on the broad book because the COVID Feb-Mar 2020 crash happens INSIDE one month — the overlay de-levers from the prior month-end label which was "calm."

---

## Part 12 — Anticipated Q&A (Pre-Loaded Answers)

### Q: "How do you know it's not a backtest artefact?"

**A:** "Three layers of evidence: (1) the engine sanity gate passes 3/3 — RandomModel gives Sharpe near 0, Oracle gives huge Sharpe, Uniform gives flat returns. (2) The feature-shuffle placebo: same recipe but with feature→ticker mapping randomly permuted gives Sharpe of −0.94 (collapses by 2 full Sharpe units). The strategy NEEDS real features. (3) Bowen wrote an independent verification script (`canonical_qfix_validate.py`) that reproduces our headline within 0.02 Sharpe."

### Q: "Why XGBoost over the neural net?"

**A:** "We tested all three with the same interface and ran a Diebold-Mariano test. XGBoost beats Lasso at p < 0.01 and beats NN at p < 0.05 on both MSE and Sharpe across every window. The NN underperforms because our effective sample size — ~250 monthly observations × ~5,000 names with non-stationary cross-sectional noise — is too small for deep models. Trees handle the noisy, fat-tailed feature distributions better."

### Q: "How do you avoid overfitting in hyperparameter tuning?"

**A:** "We use Optuna's TPE sampler over 60 trials, with the objective being out-of-sample R² on a HELD-OUT validation window (2017-2018) that's INSIDE the training period and BEFORE the test window. Then we evaluate on the test window (2019-2024) that the tuning never saw. We also bumped the Deflated Sharpe trial count from N=10 to N=25 to honestly account for all the configurations we tried across the project."

### Q: "Your β is +1.5 — isn't this just leveraged market exposure?"

**A:** "Partially yes — about 55% of our annual return comes from being 1.5× exposed to the bull market. But ~45% is FF5-residual alpha that survives controlling for market, size, value, profitability, investment, AND Carhart momentum. The +1.5 β isn't a leverage knob we chose — it's an emergent property of the model picking small-caps long and large-caps short. We could β-hedge it (Sharpe drops to +0.70 but stays uncorrelated with the S&P at correlation ≈ 0) — that's the 'pure alpha' product."

### Q: "Why 14 features specifically?"

**A:** "GKX (2020) ranks 13 features by importance — we use those 13 plus `chmom` (their #4). We tested adding `maxret` (#5) and `mom36m` (#2) in Phase 24b — the 16-feature variant performed WORSE on validation R² and on walk-forward Sharpe. More features added complexity without signal — bias-variance trade-off at our sample size. `maxret` and `mom36m` stay in the features parquet for the sensitivity record but are excluded from the canonical INCLUDE_FEATURES."

### Q: "Where does the +4,600% cumulative come from?"

**A:** "Just compounding (1 + 0.347)^12.9 ≈ 47, so $1 → $47 over 12.9 years at our +34.7%/yr CAGR. Same math as the S&P, which compounded (1.143)^12.9 = $5.63 from $1. We're 8.4× the S&P final wealth because earning +20.4 pp/yr ABOVE the S&P compounds to a large ratio over 13 years. The β-hedged pure alpha is $7.13 (1.26× the S&P) — the more honest deployable headline."

### Q: "What's your loss function?"

**A:** "Squared loss on the (date, ticker) target = next-month realized return demeaned by per-(date, sector) mean. The demeaning is the GKX Layer-2 refinement — by removing sector-level common signal before training, the model focuses on cross-sectional stock-picking rather than sector-picking."

### Q: "Why monthly rebalancing — why not weekly or daily?"

**A:** "Three reasons. (1) Sharadar's fundamentals update at the 10-Q/10-K cadence — sub-monthly rebalancing on fundamentals features gives no extra information. (2) At 175% monthly turnover and 10 bps/side, we already pay ~216 bps/yr in costs — going weekly would 3× that and erode the edge. (3) GKX 2020 uses monthly — we maintain apples-to-apples comparison. The cost is the regime overlay's monthly-frequency timing lag (couldn't catch COVID Feb-Mar 2020). Sub-monthly is in the 'future work' list."

---

## Part 13 — Quick Reference: Key Numbers Memorized

| Number | Meaning |
|---|---|
| **+1.15** | Full-OOS Sharpe ratio |
| **+18.73%/yr** | FF5 alpha |
| **t = +6.85** | Statistical significance (p < 0.001) |
| **+34.7%/yr** | Annual return CAGR |
| **+1.5** | Market β (sensitivity to S&P) |
| **+1.26** | SMB-β (small-cap tilt) |
| **−0.43** | UMD-β (momentum loading, NEGATIVE) |
| **+20.1%/yr** | Carhart 6F alpha (rises when UMD added) |
| **0.85-0.88** | DSR at N=25 trials |
| **155 months** | OOS sample size |
| **~4,400** | Median names/month in panel |
| **~440** | Positions per rebalance (k=20 × 11 sectors × 2 sides) |
| **~180%** | Monthly L1 turnover |
| **~2.9 pp/yr** | Cost drag at 10 bps/side |
| **+0.99** | S&P 500 Sharpe over same window |
| **+463%** | S&P cumulative over same window |
| **+4,600%** | Strategy cumulative (gross XGBoost) |
| **+614%** | β-hedged pure alpha cumulative (honest headline) |
| **8.4×** | Wealth ratio vs S&P after 13 years |
| **25 bps/side** | Cost level at which we tie S&P on Sharpe |

---

## Part 14 — The 60-Second Summary (For When Someone Asks "What Did You Do?")

> I built an XGBoost-based stock-picking model trained on 14 firm-level features (GKX 2020 stack) with sector-relative targets, hyperparameter-tuned via Optuna, and evaluated by walk-forward expanding-window cross-validation on a broad survivorship-free US equity panel (2002-2024 Sharadar data, ~4,400 names/month). The strategy picks k=20 longs and shorts per GICS sector, dollar-neutral, costs 10 bps/side. The full-OOS Sharpe is +1.15, FF5 alpha +18.7%/yr at t=+6.85, all statistical robustness checks pass (placebo, sanity gates, DSR at N=25, Carhart momentum control, dense k-sweep, bootstrap). Caught and corrected two methodological bugs during the project (Q-filter dropping alive tickers, INCLUDE_FEATURES not enforced). Compared to passive S&P 500: we beat it on Sharpe (+1.14 vs +0.99), on return (+34.7% vs +14.3%/yr), and on calendar-year win rate (84.6%). Honest counterweights: not market-neutral (β=+1.5), alpha concentrated in down-cap tail, capacity-binding at very large AUM.

---

*Document compiled 2026-05-24. Reference: every claim in this document maps back to a phase script, a DECISIONS.md entry, or a number in REPORT.md.*
