# quantrisk

Portfolio construction and risk management engine for liquid ETFs.

![CI](https://github.com/randomaban-beep/quantrisk/actions/workflows/ci.yml/badge.svg)

![Net NAV](figures/01_cumulative_nav.png)

Walk-forward portfolio construction with seven strategy families, transaction costs, VaR/CVaR forecasts, formal coverage backtests, stress tests, sensitivity analysis, DuckDB, SQL, and Power BI-ready tables.

## Key results (generated from `results/metrics_summary.csv`)

| strategy | cagr | volatility | sharpe | max_drawdown | sortino | var_95 | cvar_95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| equal_weight | 0.2127 | 0.1767 | 0.9577 | -0.1795 | 1.4240 | 0.0154 | 0.0239 |
| sixty_forty | 0.2127 | 0.1767 | 0.9577 | -0.1795 | 1.4240 | 0.0154 | 0.0239 |
| inverse_vol | 0.2165 | 0.1729 | 0.9930 | -0.1734 | 1.4809 | 0.0149 | 0.0234 |
| min_variance | 0.2138 | 0.1622 | 1.0340 | -0.1482 | 1.5461 | 0.0136 | 0.0224 |
| max_sharpe | 0.1951 | 0.1702 | 0.9017 | -0.1652 | 1.3334 | 0.0144 | 0.0242 |
| risk_parity | 0.2167 | 0.1730 | 0.9932 | -0.1732 | 1.4806 | 0.0148 | 0.0234 |
| hrp | 0.2152 | 0.1715 | 0.9936 | -0.1702 | 1.4863 | 0.0153 | 0.0232 |

## Headline findings

- Best net Sharpe in the saved run: **min_variance** (1.034); worst: **max_sharpe** (0.902).
- 8 VaR backtest rows reject Kupiec coverage at the configured significance threshold; fhs has the highest observed pass share.
- The most adverse hypothetical scenario shown is Equity crash for equal_weight (-19.33%); shock values are assumptions, not forecasts.

Saved backtest scope: 5 ETF assets, 2023-10-05 to 2026-10-09. The source price panel has 12 assets and covers 2007-06-01 to 2026-10-09.

## Architecture

```text
Market data -> cleaning/cache -> estimators -> optimizers -> walk-forward backtest
                                         |                     |
                                         v                     v
                                   VaR and stress -> DuckDB / SQL -> report / dashboard / Power BI
```

## Run

```bash
git clone https://github.com/randomaban-beep/quantrisk.git
cd quantrisk
make setup
make all
make dashboard
```

## Project structure

See `src/quantrisk`, `config`, `sql`, `app`, `results`, `figures`, and `powerbi`.

## Methodology and limitations

See [methodology](docs/methodology.md) and [the generated report](results/report.md). Limitations:

- ETF universe chosen with hindsight and limited history
- Results are in-sample with respect to design choices
- Simplified cost model and no market impact
- Long-only 40% cap changes textbook unconstrained optimizer results
- Expected-return estimation error affects maximum Sharpe
- VaR models assume stable relationships within each estimation window
- Hypothetical shocks are illustrative assumptions, not forecasts


## How to talk about this project

Explain the lagged walk-forward signal timing, covariance estimation choices, optimizer constraints, formal VaR coverage tests, and why a high backtest metric is not proof of future performance.
