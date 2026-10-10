# QuantRisk results

## Executive summary

- The highest observed net Sharpe in this run was **min_variance** at 1.034; the lowest was **max_sharpe** at 0.902.
- Across the VaR backtest rows, 8 Kupiec tests rejected nominal coverage; fhs had the highest Kupiec pass share.
- The most adverse illustrative shock in the saved table was **Equity crash** for equal_weight (-19.33%).
- The source panel has 12 assets and 4871 price rows; the saved backtest uses 5 assets from 2023-10-05 through 2026-10-09. See limitations below before interpreting performance.

## Data and universe

Adjusted ETF close data from Yahoo Finance with a Stooq fallback; simple returns, short-gap cleaning, and DuckDB persistence. The current saved data-quality table records 4871 observations from 2007-06-01 through 2026-10-09 with 0 missing prices.

## Methodology

See [methodology](../docs/methodology.md). The backtest is walk-forward with delayed close execution, daily weight drift, and turnover-based costs.

## Performance

| strategy | cagr | volatility | sharpe | max_drawdown | sortino | var_95 | cvar_95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| equal_weight | 0.2127 | 0.1767 | 0.9577 | -0.1795 | 1.4240 | 0.0154 | 0.0239 |
| sixty_forty | 0.2127 | 0.1767 | 0.9577 | -0.1795 | 1.4240 | 0.0154 | 0.0239 |
| inverse_vol | 0.2165 | 0.1729 | 0.9930 | -0.1734 | 1.4809 | 0.0149 | 0.0234 |
| min_variance | 0.2138 | 0.1622 | 1.0340 | -0.1482 | 1.5461 | 0.0136 | 0.0224 |
| max_sharpe | 0.1951 | 0.1702 | 0.9017 | -0.1652 | 1.3334 | 0.0144 | 0.0242 |
| risk_parity | 0.2167 | 0.1730 | 0.9932 | -0.1732 | 1.4806 | 0.0148 | 0.0234 |
| hrp | 0.2152 | 0.1715 | 0.9936 | -0.1702 | 1.4863 | 0.0153 | 0.0232 |

## Risk, drawdowns, and VaR

See `var_backtest_summary.csv` for formal coverage and independence tests. The observed pass share ranks fhs highest by Kupiec coverage; it is a descriptive ranking and does not establish universal model superiority.

![NAV](../figures/01_cumulative_nav.png)

![Drawdown](../figures/02_underwater.png)

## Stress results

Scenario P&L is `w · shock` and the hypothetical inputs are labeled as illustrative assumptions. The result table is in `stress_hypothetical.csv`.

## Statistical significance and sensitivity

Paired Sharpe comparisons and their explicit significance labels are in `sharpe_paired_tests.csv`. One-factor runs are in `sensitivity.csv` (70 rows); runs are cached by configuration hash.

## Limitations

- ETF universe chosen with hindsight and limited history
- Results are in-sample with respect to design choices
- Simplified cost model and no market impact
- Long-only 40% cap changes textbook unconstrained optimizer results
- Expected-return estimation error affects maximum Sharpe
- VaR models assume stable relationships within each estimation window
- Hypothetical shocks are illustrative assumptions, not forecasts

## Reproducibility

Run `make setup`, `make data`, `make backtest`, `make risk`, `make stress`, `make sensitivity`, and `make report`. The saved result files and configuration YAMLs are the source for numeric tables.
