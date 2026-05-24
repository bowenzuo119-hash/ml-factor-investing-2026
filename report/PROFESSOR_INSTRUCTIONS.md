# Final Deliverables — Official Course Instructions

> **Source:** Course materials provided by the professor / teaching team.
> This file consolidates the grading rubric, deliverables specification,
> suggestions/hints, and the one-page submission template — verbatim where
> possible. Reference document only; do not edit without flagging the team.

---

## Deliverables and Grading

Grading will be carried out on a **presentation** and the answers to questions and general discussion following it, to one of the **senior teaching coordinators**. Precise details will be announced closer to the end (it depends on how many teams there are), but it should normally be a **short presentation + at least 10 minutes of questions and discussion**.

> **Note:** is your responsibility to form or find a team (but the teaching team is happy to help coordinate — don't hesitate to ask us). Do not leave this task too late.

### What exactly will be graded?

Grading is based on **how well you convey your understanding of the machine learning aspects that relate to your project (and specifically those covered in the course)**. For example:

- If you **created a new dataset and formulated a new problem**, you'll talk a lot (and expect questions) about how you built/preprocessed/prepared it and used it to answer a particular question or overcome a particular problem.
- If you **coded a decision tree algorithm from scratch**, we'll discuss your implementation, its running time complexity, performance, and many details about decision trees.
- If you **compared the results of many algorithms from scikit-learn on benchmark datasets**, then you will show good general knowledge about evaluation methods, different performance metrics, and the comparative advantages of the different methods and how this is reflected in the results.
- If you **choose to tag on to an existing Kaggle competition**, you'll need to justify and motivate your chosen approach, and distinguish it from other approaches.

*Et cetera.*

### Deliverables: What to submit here?

A brief outline of your topic and team (see the template attached) as a **single-page pdf**. Plus (optionally) a version of your presentation slides in pdf form, additional experimental results, or any other material that didn't make it into the final presentation; **all of which as a single pdf document**.

This submitted material is **not graded directly** but serves as a reference to the examiner during and after your presentation; to formulate questions (and to know what to ask about, and who best to direct the questions to), and to recall the main elements of your project later.

---

## Suggestions and Hints

A rough outline of how to proceed (only as a suggestion):

1. **Define the question/problem statement** you will approach
2. **Obtain/curate a suitable dataset** (ensure that you have permission to gather this data); check that is sufficient to answer 1.
3. **Choose/create:** performance metric(s)/loss function(s), model(s), algorithm(s) you will use/build
4. **Write the code and set up** the experimental/analytical framework
5. **Run experiments** to answer the question/evaluate success (negative answers are fine if well-explained), illustrate, and carefully interpret the results.
6. Additionally you could pull out some **'nuggets of knowledge'** from your analysis (findings that were perhaps unexpected/surprising, particularly interesting, or provocative; something you learned about the problem)
7. **Identify limitations**, speculate about future steps, and what you have learned/would have done differently
8. **Revisit all the above steps again** to fine-tune everything.

### Hints

- Go for a topic that you are **interested in or want to learn more about**
- **Prompting LLMs is, in itself, not machine learning** — we prefer that you train a rudimentary simplistic language model, rather than simply trying to fit an LLM into your pipeline
- Try to **avoid having to work with large volumes of non-tabular data** (e.g., images) on the large architectures which imply its use; or at least be aware of the challenges: it requires substantial computation, tiresome hyper-parameter tuning; and interpretation of results can be challenging.
- **Do not do a project on reinforcement learning** (which comes at the end of the course).
- **Make sure to limit the scope appropriately.** Top reason for top teams not getting top grades was being too ambitious, trying to do too much and not conclusively achieving (or not concisely defining) any of their original objectives.
- **In experimental comparisons, never forget to compare to some kind of baseline** (linear regression, Naive Bayes, predicting the majority class, etc.) as a reference
- **Focus on interpretation of results** rather than simply reporting them
- **Recognize limitations** of your study (indeed, make sure to limit it to a suitable scope — given the limited time frame)
- Complex questions and discussion arise from **relatively simple methods and tasks** — yet another reason to start simple, and aim to set a clear limited scope
- It is often **the data which makes an interesting problem.** Data is everywhere: you may gather it 'manually' (data-entry from observations, questionnaires, ...) or get it from some online source, or some offline source, data repositories, ... You may turn an existing collection into a dataset, modify an existing dataset, or just take an off-the-shelf (e.g., from Kaggle) dataset.
- If you do collect/curate your own data (and this is great!), make sure it is **enough** (at least several hundred data points — but this depends greatly on the problem/context) and make sure it is really suited to your objectives
- If you're not sure about anything, **don't hesitate to discuss** with one of the teaching coordinators prior to committing to a project, or a particular idea/approach.

### Attribution / Citations (important)

It is of **utmost importance** to **unambiguously mention/reference/cite whatever you based your project on** (blog post, code on GitHub, some existing Kaggle competition, public data set, ...). You can copy figures, copy code, take data, from anywhere you like as long as this is permitted (e.g., the data/algorithm is not legally restricted), as long as you **explicitly acknowledge this clearly** (in your slides/presentation).

---

## One-Page Submission Template

The professor provided a one-page PDF template (`Project_Template.pdf`). Reproduced below in markdown form:

### Header

```
Title of your project
Team #
```

### Authors block — 4 columns, one per person

For each person, a photo + name + **3 keywords**:

| Person #1 | Person #2 | Person #3 | Person #4 |
|---|---|---|---|
| Keyword 1 | Keyword 1 | Keyword 1 | Keyword 1 |
| Keyword 2 | Keyword 2 | Keyword 2 | Keyword 2 |
| Keyword 3 | Keyword 3 | Keyword 3 | Keyword 3 |

> *The keywords above should aim to give a rough idea of contributions in terms of material/scope; who did what in the pipeline; so that we know whom to expect questions from, and to distribute questions evenly. By default we assume that all authors contributed equally in terms of effort. Any optional comments about the team composition.*

### Section: Highlights (4–6)

- Highlight 1 (these should be the main 'takeaway' points for an audience to your presentation)
- Highlight 2
- Highlight 3
- Highlight 4
- *(optional)* Highlight 5
- *(optional)* Highlight 6

### Section: Brief Report

> Provide an indication of the **scope**, and **how you proceeded**, **what was left out of the presentation** (if anything).
> Please highlight **keywords** in bold like that; anything that relates to material/algorithms seen in the course.
> This is a bit like an **abstract of a paper, except extended** to give a feel of your working.
>
> **Hint:** we will attempt to align questions with the scope as stated here.
> Also: please **state your main references** if you had anything beyond the material of the course.

### Section: Description of Appendices (Optional)

> Don't feel the temptation to keep a record and show everything you did, or display every line of code you wrote. Much like in an exam, **'latent effort' will show through**. But, if there is something you feel you'd like to show, append it to the rest of the file, but mention here what it is and where to find it, e.g., 'additional experiments on pages 4–9'.

---

## Our project's working draft — instantiation of the template

To be filled in before submission. Suggested values based on the current state of the report:

### Title

*"Machine-Learning Factor Investing on a Survivorship-Free US Equity Universe"* (working title from REPORT.md)

### Team

Bowen Zuo · Nicolas Couto Mota · Andrea Fontana — 3-person team, 5-week project.

### Keywords per person (3 each)

| **Bowen Zuo** (Person A) | **Nicolas Couto Mota** (Person B) | **Andrea Fontana** (Person C) |
|---|---|---|
| Walk-forward backtest engine | XGBoost / Lasso / NN regressors | Hidden Markov Model (HMM) regime detection |
| Survivorship-free data lane (Sharadar PIT) | Feature engineering (GKX 14-feature stack) | Gaussian Mixture Model (GMM) regime detection |
| Fama-French 5-factor + Carhart + bootstrap robustness | Optuna hyperparameter tuning + Diebold-Mariano | Walk-forward expanding-window OOS evaluation |

### Highlights (4–6 candidates, pick 4)

- Built a survivorship-free, point-in-time ML factor pipeline that produces a **full-OOS Sharpe of +1.15 / FF5 alpha of +18.73%/yr at t=+6.85** over 2012–2024 on a broad ~4,400-name US equity universe — first statistically significant alpha after factor adjustment in the project lineage.
- The alpha **survives every rigor check we ran**: Carhart momentum control, block bootstrap (P(SR≤0)=0.0002), Deflated Sharpe Ratio at N=25 trials (0.85–0.88), cost-grid stress (α significant up to ~50 bps/side), and a feature-shuffle placebo (+1.15 → −0.94 when features are scrambled — rules out leakage).
- **Audit-driven methodology**: we caught and corrected a survivorship leak in our own engine (Sharpe +1.49 → −0.31 after PIT applied to S&P-500-only), then rebuilt the canonical on the broad survivorship-free Sharadar universe.
- **Honest down-cap finding** (GKX-style): on the strict rolling top-2,000 large/mid-cap sub-universe the FF5 alpha collapses to +1.8%/yr at t=0.96 (not significant) — confirming Gu-Kelly-Xiu (2020) that ML cross-sectional alpha lives in the small/mid-cap tail, with capacity and single-name fragility as the binding limits at deployable AUM.
- Regime-overlay ablation: HMM-based leverage overlay is net-zero on the broad book (COVID monthly-frequency lag), but works on strict-S&P (max DD −25.5% → −19.9%) — universe-dependent finding.

### Main references (beyond course material)

- Gu, S., Kelly, B. & Xiu, D. (2020). *Empirical Asset Pricing via Machine Learning.* Review of Financial Studies 33(5).
- Bailey, D. & López de Prado, M. (2014). *The Deflated Sharpe Ratio.* JPM 40(5).
- Fama, E. F. & French, K. R. (2015). *A Five-Factor Asset Pricing Model.* JFE 116(1).
- Carhart, M. M. (1997). *On Persistence in Mutual Fund Performance.* JoF 52(1).
- Newey, W. K. & West, K. D. (1987). *A Simple, Positive Semi-Definite, HAC Covariance Matrix.* Econometrica 55(3).
- Full reference list in `report/REPORT.md` §9.

### Data attribution

- **SHARADAR / Nasdaq Data Link** (SF1, SEP, DAILY, TICKERS, SP500, ACTIONS tables) — premium subscription, used under the licence terms attached to the Sharadar / Nasdaq Data Link account.
- **Kenneth French Data Library** (Fama-French 5-factor + momentum factor monthly returns) — public, used per Kenneth French's terms.
- **FRED** (Federal Reserve Economic Data: GS10, GS2, DBAA, DAAA) — public.
- **yfinance** (^GSPC, ^VIX) — public; used as data source for the regime model's macro features.

### Appendix contents (if attaching beyond the one-pager)

- Full report: `report/REPORT.md` (~800 lines)
- Decision log: `DECISIONS.md` (1,233 lines, complete chronological provenance)
- Pre-PR audit checklist: `report/PRE_PR_CHECKLIST.md`
- Long-form companions: `report/DATA_AND_ENGINE_SECTION.md`, `report/ALPHA_MODEL_SECTION.md` (marked superseded but retained for audit)
- All result artefacts under `results/` (24 key files verified present)
- All phase scripts under `notebooks/persona/` (Bowen), `notebooks/personb/` (Nicolas), `notebooks/personc/` (Andrea)
- Source code under `src/` (backtest engine, models, metrics, sanity gates)

---

## Project state vs. professor's grading criteria — self-check

| Professor's criterion | Our coverage |
|---|---|
| Convey understanding of ML aspects covered in the course | XGBoost (gradient boosting), Lasso (regularised linear regression), small NN, walk-forward cross-validation, feature engineering, hyperparameter tuning via Optuna, model comparison via Diebold-Mariano test, OOS evaluation methodology |
| Compare to a baseline | Three-model comparison (Lasso linear baseline, XGBoost canonical, NN secondary) reported in §3 + §5; Random / Oracle / Uniform sanity gates in `src.sanity` |
| Focus on interpretation, not just reporting | §6 honest counterweights: not market-neutral, down-cap concentration, capacity binding, regime overlay COVID-lag, single-name fragility |
| Recognise limitations | §6 covers cost/capacity, name fragility, single OOS path, monthly-frequency regime detection, q-filter bug history |
| Limit scope appropriately | Single tightly-defined question ("does ML factor alpha survive on a survivorship-free US universe under realistic costs?"); scope explicitly bounded in §7 (no sub-monthly, no intraday, no post-2024 data) |
| Citations / attribution | §9 References + data attribution in §2 + DECISIONS.md commit-level provenance |
| Nuggets of knowledge | The +1.49 → −0.31 → +1.15 audit journey; the down-cap concentration finding; the placebo-shuffle as a leakage test; the feature-shuffle Sharpe collapse +1.15 → −0.94 |
| Negative results well-explained | The strict-S&P-500 canonical produces no significant alpha — kept as a positive finding (confirms academic literature) |

---

*File created 2026-05-24. Update if the professor / teaching team revises the deliverables specification or template.*
