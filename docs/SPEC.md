# CODEX MASTER PROMPT: `quantrisk` — Portfolio Construction & Risk Engine

You are a senior quantitative developer. Build the complete project below, end to end, in the current empty folder, without asking me any questions. The project is published at **https://github.com/randomaban-beep/quantrisk** (public, currently empty) and must be pushed there as described in Section 0.3. Where something is ambiguous, make a reasonable, industry-standard choice and log it in `PROGRESS.md` under "Decisions". I will not intervene.

The finished project must be resume-ready: correct, tested, reproducible, honestly reported, and explainable in a quant or data-analyst interview.

---

## 0. OPERATING RULES (read first, obey throughout)

### 0.1 Usage-limit discipline (critical)
My AI usage allowance is limited. Work efficiently:

1. **First action:** save this entire document as `docs/SPEC.md`. Then create a short `AGENTS.md` (max 40 lines) containing: project goal in 3 lines, the folder map, the commands (`make ...`), the coding rules in Section 0.2, and the instruction "Read `PROGRESS.md` first, then only the relevant stage section of `docs/SPEC.md`. Never re-read the whole spec."
2. Create `PROGRESS.md` with: a checklist of Stages 0–7, a "Decisions" section, a "Known issues" section, and a "Next exact step" line. Update it after every stage.
3. Work **one stage at a time** (Section 2). After each stage: run `make test` and `make lint`, run the stage once, `git commit`, `git push` (Section 0.3), update `PROGRESS.md`, print a summary of at most 8 lines, then continue to the next stage automatically.
4. Keep terminal output small: use `pytest -q -x`, `head`, and never print whole DataFrames. Do not re-open files you just wrote. Prefer targeted edits over rewriting files.
5. Allow at most **2 fix attempts** per failing test or lint error. If still failing, log it in "Known issues" and continue.
6. Never re-run the full pipeline unless code feeding it changed. Cache data and results to disk (`data/`, `results/`) and reuse them.
7. During development use the `--dev` flag (5 tickers, last 3 years) and only run the full universe once per stage when needed.
8. If you detect that you are near a usage limit or are told capacity is low, immediately finalize `PROGRESS.md` with the exact next step and stop cleanly. If a future session starts with "Resume", read `PROGRESS.md` and continue from "Next exact step".

### 0.2 Coding rules
- Python 3.11+, type hints, short docstrings, deterministic (`seed = 42`), no hardcoded paths, no notebooks.
- Config in `config/*.yaml`; no magic numbers in code.
- Use `logging`, not `print`, inside the library.
- All numbers in the README and report must be inserted **programmatically from result files**. Never type a result by hand and never invent numbers.
- No look-ahead bias anywhere. Weights decided with data through day `t` can only earn returns after day `t` (see Section 3.4).
- Never fabricate market data. Synthetic data is allowed only inside tests and must be generated in `tests/` fixtures. If real data cannot be downloaded after retries and the fallback source, stop that stage, record it in `PROGRESS.md`, and do not report results.
- Be honest in the report: show where strategies lose, state limitations, and do not claim outperformance unless the numbers and statistical tests support it.

### 0.3 GitHub rules
- **Remote:** `https://github.com/randomaban-beep/quantrisk.git`, default branch `main`.
- In Stage 0: `git init -b main`, then `git remote add origin https://github.com/randomaban-beep/quantrisk.git`. If no git identity is configured, set a local one: `user.name = randomaban-beep`, `user.email = 248483731+randomaban-beep@users.noreply.github.com`.
- After **every stage commit**, run `git push -u origin main`. Use clear conventional commit messages (e.g. `feat: add HRP optimizer`, `test: add Kupiec test`), one logical change per commit, no giant "initial commit".
- **Authentication:** use whatever credentials are already configured on this machine (browser sign-in, Git Credential Manager, or `gh`). Never ask me for a token, never write a token or password into any file or into the remote URL, and never commit secrets, `.env` files, or API keys.
- If a push fails (authentication or network), do **not** retry in a loop: log "Push pending" with the exact command in `PROGRESS.md`, keep committing locally, and try one more push at the end of each later stage. At the very end, if pushes still fail, print the exact commands I need to run.
- Never force-push. Never commit `data/`, `.venv/`, caches, or files over 5 MB. `results/`, `figures/`, and `powerbi/` **are committed** (they are the proof of work), but keep each file small (store large time series as compressed parquet only if under 5 MB, otherwise gitignore them and regenerate with `make all`).
- Add an MIT `LICENSE` file (copyright holder: randomaban-beep).

---

## 1. PROJECT OVERVIEW

**Name:** `quantrisk`

**Goal:** A multi-asset portfolio construction and risk-management engine. It builds portfolios with seven methods (including mean-variance, risk parity, and hierarchical risk parity), backtests them walk-forward with transaction costs, measures risk with VaR/CVaR, backtests the VaR models with formal statistical tests (Kupiec, Christoffersen, Basel traffic light), runs stress tests, and presents everything in a report, a dashboard, and Power BI-ready tables, with a SQL analytics layer.

**Universe (config-driven, `config/universe.yaml`), 12 liquid ETFs across asset classes:**

| Ticker | Asset class |
|---|---|
| SPY, QQQ, IWM | US equity (large, growth, small) |
| EFA, EEM | International developed and emerging equity |
| IEF, TLT | US Treasuries (intermediate, long) |
| LQD, HYG | Investment grade and high yield credit |
| VNQ | Real estate |
| GLD | Gold |
| DBC | Broad commodities |

- Data start: `2007-06-01` (HYG inception limits this). End: today's date (`datetime.date.today()`).
- Source: `yfinance` with `auto_adjust=True`, close prices. Retry 3 times with backoff. Fallback source: Stooq via `pandas-datareader`.
- Risk-free rate: `^IRX` (13-week T-bill yield, quoted in percent), converted to a daily rate: `rf_daily = (IRX/100)/252`, forward-filled. If unavailable, use 0 and log it in "Decisions".
- Cleaning: forward-fill gaps of up to 3 days, drop dates where all assets are missing, assert no remaining NaN after the first valid date, assert monotonic unique index.
- Returns: simple daily returns. Annualization factor: 252.

---

## 2. STAGES

### Stage 0: Setup
- `git init`, `.gitignore` (data caches, `.venv`, `__pycache__`), virtual env, `requirements.txt` (pinned minimum versions): `numpy pandas scipy scikit-learn matplotlib yfinance pandas-datareader pyarrow duckdb pyyaml streamlit pytest ruff`.
- Folder structure:

```
quantrisk/
  AGENTS.md  PROGRESS.md  README.md  LICENSE  Makefile  requirements.txt  pyproject.toml
  .github/workflows/ci.yml
  config/    universe.yaml  backtest.yaml  risk.yaml  stress.yaml
  docs/      SPEC.md  methodology.md
  src/quantrisk/
    data.py  db.py  estimators.py  optimizers.py  backtest.py  metrics.py
    var_models.py  var_backtests.py  stress.py  stats.py  sensitivity.py
    plots.py  report.py  exports.py  cli.py
  sql/       01_monthly_returns.sql ... (see Stage 6)
  tests/
  results/   (generated)  figures/ (generated)  powerbi/ (generated)
  data/      (generated, gitignored)
  app/       dashboard.py
```
- `Makefile` targets: `setup data backtest risk stress sensitivity report sql dashboard test lint all dev`.
- CLI: `python -m quantrisk.cli run --stage {data,portfolios,risk,stress,sensitivity,report,all} [--dev]`.
- `.github/workflows/ci.yml`: GitHub Actions on push and pull request, Python 3.11, install `requirements.txt`, run `ruff check .` and `pytest -q`. CI must **not** need network or market data (tests use mocks and synthetic fixtures only).
- Set up the remote per Section 0.3.
- Commit: "chore: scaffold project structure", then push.

### Stage 1: Data layer and DuckDB
- `data.py`: download, clean, validate, cache to `data/raw/prices.parquet`, compute returns to `data/processed/returns.parquet`, fetch rf series.
- `db.py`: load prices, returns, rf, and (later) weights, NAV, metrics, VaR, stress tables into `data/quant.duckdb`.
- Print a data-quality summary (rows, date range, missing-value counts) to `results/data_quality.csv`.
- Tests: no NaN after cleaning, index monotonic, returns shape, cache reuse (second call does not hit the network, use a mock).

### Stage 2: Estimators and portfolio optimizers
**Covariance (`estimators.py`):** sample, Ledoit-Wolf shrinkage (`sklearn.covariance.LedoitWolf`, default), and EWMA (lambda 0.94). Expected returns: historical mean shrunk toward the cross-sectional mean with `mu_shrinkage = 0.5` (config).

**Seven strategies (`optimizers.py`)**, all **long-only, fully invested, per-asset max weight 40%** (configurable, bounds `[0, 0.40]`):

1. `equal_weight` (1/N benchmark)
2. `sixty_forty`: 60% SPY / 40% IEF (static benchmark)
3. `inverse_vol`: weights proportional to 1/sigma
4. `min_variance`: minimize `w'Σw`
5. `max_sharpe`: maximize `(w'μ - rf)/sqrt(w'Σw)` using shrunk μ and Ledoit-Wolf Σ (solve with SLSQP via a convex-friendly reformulation or direct maximization from multiple starts; document the choice)
6. `risk_parity` (equal risk contribution): minimize `Σ_i Σ_j (RC_i - RC_j)^2` with SLSQP under the bounds and sum-to-one constraint, initialized from inverse-vol
7. `hrp`: Hierarchical Risk Parity, implemented from scratch following López de Prado (2016): correlation distance `d = sqrt(0.5*(1-ρ))`, linkage (config, default `single`), quasi-diagonalization, recursive bisection with inverse-variance cluster allocation. If any weight exceeds the cap, clip and renormalize iteratively.

Every optimizer: pure function `weights = f(returns_window, cfg) -> pd.Series`, with a fallback to inverse-vol (logged) if the solver fails.

**Portfolio analytics helpers:** risk contributions (`RC_i = w_i (Σw)_i / sqrt(w'Σw)`), diversification ratio, effective number of assets (`1/ΣW_i^2`), effective number of bets (exponential of entropy of normalized risk contributions), concentration (HHI), asset-class aggregation of weights and risk.

**Tests:**
- all weights sum to 1 (tolerance 1e-8), within bounds, no NaN
- `risk_parity` risk contributions equal within 1% when caps are not binding
- `min_variance` variance <= variance of equal weight on random covariance matrices
- `hrp` on a hand-built 4-asset block-correlation example gives expected cluster ordering and weights summing to 1
- `inverse_vol` and `equal_weight` against closed-form results
- Commit: "Stage 2: estimators and optimizers".

### Stage 3: Walk-forward backtest engine and performance metrics
See Section 3 for the exact mechanics. Implement `backtest.py` and `metrics.py`. Outputs saved to `results/`:
- `nav.parquet` (daily net NAV per strategy), `returns_net.parquet`, `returns_gross.parquet`, `weights.parquet` (long format: date, strategy, asset, weight, both target and drifted), `turnover.csv`, `costs.csv`, `metrics_summary.csv`.
- Tests: Section 3.6.
- Commit: "Stage 3: backtest and metrics".

### Stage 4: Risk engine, VaR/CVaR and formal backtests
See Section 4. Implement `var_models.py`, `var_backtests.py`. Outputs: `results/var_forecasts.parquet`, `results/var_backtest_summary.csv`, `results/component_risk_latest.csv`.
- Commit: "Stage 4: VaR engine and backtests".

### Stage 5: Stress testing, statistical tests, sensitivity
See Sections 5 and 6. Implement `stress.py`, `stats.py`, `sensitivity.py`. Outputs: `results/stress_historical.csv`, `results/stress_hypothetical.csv`, `results/stress_correlation.csv`, `results/sharpe_bootstrap.csv`, `results/sensitivity.csv`, `results/subperiods.csv`.
- Commit: "Stage 5: stress, stats, sensitivity".

### Stage 6: Visuals, SQL layer, report, dashboard, Power BI exports
See Section 7. Commit: "Stage 6: reporting layer".

### Stage 7: Final QA, README, resume bullets
See Section 8. Commit: "docs: final README, report and resume bullets", then push. Verify the push succeeded with `git status` and `git log origin/main -1`.

---

## 3. BACKTEST ENGINE SPECIFICATION (`backtest.py`)

### 3.1 Parameters (`config/backtest.yaml`)
| Parameter | Baseline | Notes |
|---|---|---|
| `estimation_window` | 252 trading days | rolling, ends at the rebalance date |
| `rebalance_freq` | monthly | last trading day of each month; also support quarterly |
| `execution_lag_days` | 1 | conservative: trade one day after the signal |
| `cost_bps` | 5 | per unit of one-way turnover (commission + spread) |
| `slippage_bps` | 2 | per unit of one-way turnover |
| `max_weight` | 0.40 | |
| `min_history_days` | `estimation_window` | first rebalance after warm-up |
| `initial_capital` | 1.0 | NAV indexed at 1.0 |

### 3.2 Procedure
1. On each rebalance date `t`, take returns from the window ending at `t` (inclusive), compute target weights with each optimizer.
2. The target weights become effective after `execution_lag_days` trading days (trade at that day's close).
3. Between rebalances, weights **drift** with asset returns (buy-and-hold).
4. At each rebalance trade: `turnover = Σ|w_target - w_drifted|`; deduct `turnover * (cost_bps + slippage_bps)/1e4` from that day's portfolio return.
5. Record gross and net returns separately, target and drifted weights, turnover, and costs.
6. Before the first rebalance the strategy is in cash earning `rf` (excluded from performance statistics, which start at the first invested day).

### 3.3 Benchmarks
`equal_weight` and `sixty_forty` use the same engine (same rebalance frequency and costs) for a fair comparison. Also compute buy-and-hold `SPY` as a reference series.

### 3.4 No look-ahead (hard rule)
Weights formed with data through day `t` never earn a return dated `<= t + execution_lag_days`. Add a dedicated test (Section 3.6).

### 3.5 Metrics (`metrics.py`), computed on net returns unless stated; annualized with 252
**Return and risk:** CAGR (geometric), annualized volatility, Sharpe ratio (excess over rf), Sortino ratio (downside deviation relative to rf), Calmar ratio (CAGR / |max drawdown|), maximum drawdown, max drawdown duration (days), average drawdown, Ulcer index, skewness, excess kurtosis, best/worst day, best/worst month, % positive months.
**Tail risk:** historical VaR and CVaR (95% and 99%, daily).
**Relative to benchmark:** alpha and beta vs SPY (regression on daily excess returns, alpha annualized), information ratio and tracking error vs `equal_weight`, up/down capture vs SPY.
**Portfolio structure and trading:** annualized turnover, average and total costs, average effective number of assets, average effective number of bets, average diversification ratio, average and max single-asset weight, asset-class allocation over time.
Output a tidy `metrics_summary.csv` (rows = strategies, columns = metrics) and a long-format version for Power BI.

### 3.6 Tests
- Weights drift correctly between rebalances (hand-computed 2-asset example).
- Cost accounting: zero turnover means zero cost; full rotation costs `2 * bps` correctly per convention (document the convention).
- NAV equals cumulative product of net returns.
- **No look-ahead test:** inject a "future shock" return on day `t+1` after a rebalance date and assert weights at `t` are unchanged; also assert the shifted-signal property.
- Metrics against hand-computed values on a small known series (CAGR, vol, Sharpe, max drawdown, Sortino).

---

## 4. RISK ENGINE SPECIFICATION

### 4.1 VaR / CVaR models (`var_models.py`)
Applied to **each strategy's holdings**: on each day `t`, take the weights actually held at the start of day `t`, apply them to historical asset returns in a window ending `t-1` to obtain a pro-forma portfolio return series, then forecast the one-day-ahead VaR and CVaR (Expected Shortfall). Compare with the realized gross return on day `t`. Confidence levels: 95% and 99%. Window: 500 days (config; also test 250).

Four models:
1. **Historical simulation** (empirical quantile).
2. **Parametric Gaussian** (variance-covariance, EWMA lambda 0.94 covariance).
3. **Cornish-Fisher** (skewness and kurtosis adjusted quantile).
4. **Filtered historical simulation (FHS)**: standardize returns by EWMA volatility, take the empirical quantile of standardized returns, rescale by the current EWMA volatility.

Report VaR as a positive loss number. Vectorize with NumPy; no Python loops over days where avoidable.

### 4.2 Formal VaR backtests (`var_backtests.py`)
For every strategy x model x confidence level:
- Number of exceptions, expected exceptions, exception rate.
- **Kupiec proportion-of-failures (POF) test** (unconditional coverage), LR statistic and p-value (chi-square, 1 df).
- **Christoffersen independence test** (exception clustering) and **conditional coverage test** (chi-square, 2 df). Handle zero-exception and degenerate cases safely (define `0*log(0)=0`).
- **Basel traffic light** for 99% VaR over the trailing 250 days (green 0-4, yellow 5-9, red 10+ exceptions), reported over time.
- **Expected Shortfall check:** ratio of mean realized loss on exception days to mean forecast ES on those days (ideal is about 1).
- Pass/fail flags at the 5% significance level.
Save the full table as `var_backtest_summary.csv` and a model-ranking table (which model passes most tests per strategy).

### 4.3 Risk decomposition (latest date)
For each strategy's latest weights: parametric component VaR and marginal VaR per asset, percentage contribution to risk by asset and by asset class. Save `component_risk_latest.csv`.

### 4.4 Tests
- Historical VaR on a known sample equals `np.quantile`.
- Parametric VaR on large simulated normal data matches the analytical value within tolerance.
- Kupiec test on simulated exceptions with correct coverage (p-value high on average) and with wrong coverage (p-value low); known textbook example value for LR.
- Christoffersen test detects artificially clustered exceptions.
- Component VaRs sum to total portfolio VaR.
- VaR forecasts at day `t` use only data through `t-1` (no look-ahead test).

---

## 5. STRESS TESTING (`stress.py`, `config/stress.yaml`)

Apply to the **latest weights of each strategy**.

1. **Historical scenarios** (replay actual asset returns over the window, report cumulative portfolio return and max drawdown within the window):
   - Global Financial Crisis: 2008-09-01 to 2009-03-09
   - 2011 Euro-crisis selloff: 2011-07-22 to 2011-10-03
   - Q4 2018 selloff: 2018-09-20 to 2018-12-24
   - COVID crash: 2020-02-19 to 2020-03-23
   - 2022 rates and inflation shock: 2022-01-03 to 2022-10-14
   Also report each strategy's **realized** backtest return in the same windows (what actually happened under that strategy).
2. **Hypothetical shocks** (instantaneous asset returns defined in `stress.yaml` as clearly labeled illustrative assumptions, not forecasts): "Equity crash", "Rates +200bp", "Stagflation", "Credit spread blowout", "Flight to quality". Report portfolio P&L per scenario and the top 3 contributing assets.
3. **Correlation stress:** keep each asset's volatility, raise all off-diagonal correlations to rho in {0.5, 0.8, 0.95} (blended with the estimated matrix), recompute portfolio volatility and 99% parametric VaR, and report the increase versus baseline.
Tests: scenario P&L equals `w · shock` for hand-built inputs; correlation stress with rho=1 gives volatility equal to the weighted sum of volatilities.

---

## 6. STATISTICAL VALIDATION AND SENSITIVITY

### 6.1 `stats.py`
- **Stationary bootstrap** (Politis-Romano, expected block length 21 days, 5000 resamples, seed 42): 95% confidence intervals for Sharpe, CAGR, and max drawdown of each strategy.
- **Paired bootstrap test of the Sharpe-ratio difference** of each strategy versus `equal_weight` (two-sided p-value) and versus `sixty_forty`.
- State plainly in the output table which differences are not statistically significant.

### 6.2 `sensitivity.py`
One-at-a-time variations around the baseline (not a full grid), report Sharpe, CAGR, max drawdown, turnover per strategy:
- `estimation_window` in {126, 252, 504}
- `rebalance_freq` in {monthly, quarterly}
- `cost_bps + slippage_bps` in {0, 7, 15, 30}
- covariance estimator in {sample, ledoit_wolf, ewma}
- `max_weight` in {0.25, 0.40, 1.00}
Cache every run to disk keyed by config hash so nothing is recomputed. Save `sensitivity.csv`.

### 6.3 Sub-period analysis
Calendar-year returns table per strategy, plus regimes: 2008-09 crisis, 2010-2019 expansion, 2020 COVID, 2022 rate shock, and 2023-present. Save `subperiods.csv`.

---

## 7. OUTPUTS: FIGURES, SQL, REPORT, DASHBOARD, POWER BI

### 7.1 Figures (`plots.py`, matplotlib, 150 dpi, saved to `figures/`)
1. Cumulative net NAV (log scale), all strategies vs SPY
2. Underwater (drawdown) chart
3. Rolling 1-year Sharpe and rolling 1-year volatility
4. Stacked weights over time (small multiples per strategy)
5. Risk contribution by asset (latest), risk parity vs others
6. HRP dendrogram (latest date) and correlation heatmap (latest estimation window)
7. Long-only efficient frontier (latest date, Ledoit-Wolf covariance) with each strategy plotted
8. VaR forecast vs realized returns with exception markers (baseline strategy `risk_parity`, 99%, FHS and historical)
9. Heatmap of Kupiec p-values (strategy x model)
10. Basel traffic light zone over time
11. Stress test bar charts (historical and hypothetical)
12. Sensitivity chart (Sharpe vs estimation window and vs cost level)
Clear titles, labeled axes, consistent colors per strategy, no overlapping text.

### 7.2 SQL analytics layer (`sql/`, run against `data/quant.duckdb` via `make sql`, outputs to `results/sql/`)
Write six readable queries using window functions and CTEs:
1. `01_monthly_returns.sql`: monthly return pivot per strategy
2. `02_drawdown_windows.sql`: running peak and drawdown via window functions
3. `03_worst_days.sql`: ten worst days per strategy with the dominant contributing asset
4. `04_regime_performance.sql`: performance by regime
5. `05_var_exceptions_by_year.sql`: VaR exceptions by year, model, and strategy
6. `06_turnover_cost_drag.sql`: yearly turnover and cost drag
The report must reference these queries as the SQL component of the project.

### 7.3 Auto-generated report (`report.py` -> `results/report.md`)
Sections: Executive summary (3-5 bullets generated from results), Data and universe, Methodology (links to `docs/methodology.md`), Performance table, Risk and drawdowns, VaR/CVaR backtest findings (which model is best and why, with test results), Stress results, Statistical significance, Sensitivity, Limitations, Reproducibility. Embed figures via relative links. All numbers pulled programmatically.

### 7.4 `docs/methodology.md`
Concise write-up (max 2 pages) of each method with formulas and references: López de Prado (2016) HRP; Maillard, Roncalli, Teïletche (2010) equal risk contribution; Markowitz (1952); Ledoit and Wolf (2004); Kupiec (1995); Christoffersen (1998); BCBS (1996) traffic light; Politis and Romano (1994); Cornish-Fisher expansion; filtered historical simulation (Barone-Adesi et al.).

### 7.5 Dashboard (`app/dashboard.py`, Streamlit)
Read-only from `results/`. Tabs: Performance (NAV, metrics table, strategy selector), Allocation (weights, risk contributions), Risk (VaR vs returns, backtest table, traffic light), Stress (historical and hypothetical), Sensitivity. Fast, no recomputation, loads cached files. `make dashboard` launches it.

### 7.6 Power BI exports (`exports.py` -> `powerbi/`)
Tidy CSVs with consistent keys: `dim_strategy`, `dim_asset` (with asset class), `dim_date`, `fact_returns`, `fact_nav`, `fact_weights`, `fact_metrics_long`, `fact_var_backtest`, `fact_stress`, `fact_sensitivity`. Include a short `powerbi/README.md` describing the relationships (star schema) and 8 suggested DAX-free visuals.

---

## 8. FINAL QA, README AND RESUME BULLETS

### 8.1 Final QA checklist (Codex verifies and ticks in `PROGRESS.md`)
- [ ] `make all` runs from a clean clone and finishes without error
- [ ] `make test` passes; `make lint` clean (ruff)
- [ ] No look-ahead tests pass for weights and VaR
- [ ] All seven strategies produce valid weights at every rebalance
- [ ] Every number in README and report traces to a file in `results/`
- [ ] All 12 figures exist and render correctly
- [ ] Dashboard starts without error; Power BI CSVs have no NaN in keys
- [ ] Limitations section is written and honest
- [ ] Repo size is reasonable (no large caches committed), no secrets or tokens in history
- [ ] CI workflow file exists and passes locally with `ruff check .` and `pytest -q`
- [ ] `git push` to `https://github.com/randomaban-beep/quantrisk` succeeded and `origin/main` matches local `main`
- [ ] README renders correctly with images, the CI badge, and working relative links

### 8.2 README.md (generated partly from results by `report.py`)
Start with the title, a one-line tagline, and the CI badge: `![CI](https://github.com/randomaban-beep/quantrisk/actions/workflows/ci.yml/badge.svg)`. Include: one-paragraph description, a hero figure (NAV chart) near the top, key results table (generated), 3 headline findings (generated, honest even if unflattering), architecture diagram (ASCII), how to run (`git clone https://github.com/randomaban-beep/quantrisk.git && cd quantrisk && make setup && make all`), project structure, methodology summary, limitations, references, and a "How to talk about this project" section.

**Limitations to state explicitly:** ETF universe chosen with hindsight and limited history; results are in-sample with respect to design choices; simplified cost model and no market impact; long-only with a 40% cap changes the textbook results of unconstrained methods; estimation error in expected returns affects max-Sharpe; VaR models assume a stable relationship in the estimation window; hypothetical shocks are illustrative assumptions.

### 8.3 Resume bullets (write to `docs/RESUME_BULLETS.md`)
Generate 4 concise, quantified bullets using the real numbers computed by the project (Sharpe range, max drawdown comparison, number of strategies, VaR backtest result such as the share of tests passed by the best model, number of stress scenarios, tests count). Do not exaggerate; if a result is not impressive, phrase the bullet around the methodology and rigor instead. Each bullet set must end with the repo link `github.com/randomaban-beep/quantrisk`. Also generate: a 2-line project summary for LinkedIn, a GitHub repository description (max 120 characters) and 8 topic tags (e.g. `quantitative-finance`, `portfolio-optimization`, `risk-management`, `value-at-risk`, `hierarchical-risk-parity`, `backtesting`, `python`, `duckdb`) saved in `docs/GITHUB_SETUP.md`, plus the one-line `gh repo edit` command that applies them, and 6 likely interview questions with short model answers based on this project (e.g., why HRP vs mean-variance, why Kupiec and Christoffersen, how look-ahead bias was prevented, what the cost sensitivity shows).

---

## 9. DEFINITION OF DONE
The project is complete when: all stages are checked in `PROGRESS.md`, all tests pass, `make all` reproduces every file in `results/`, `figures/`, and `powerbi/`, the README and report contain real computed numbers only, `docs/RESUME_BULLETS.md` and `docs/GITHUB_SETUP.md` exist, and everything is pushed to `https://github.com/randomaban-beep/quantrisk` (or the exact manual push commands are printed). Finish by printing a final summary of at most 15 lines: headline results, test count, any items in "Known issues", and the commands to run the project and dashboard.

**Begin now with Section 0.1 step 1, then Stage 0.**