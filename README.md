# quantrisk

Portfolio construction and risk management engine for liquid ETFs.

![CI](https://github.com/randomaban-beep/quantrisk/actions/workflows/ci.yml/badge.svg)

![Net NAV](figures/01_cumulative_nav.png)

Walk-forward portfolio construction with seven strategy families, transaction costs, VaR/CVaR forecasts, formal coverage backtests, stress tests, sensitivity analysis, DuckDB, SQL, and Power BI-ready tables.

## Key results (generated from `results/metrics_summary.csv`)

| strategy | cagr | volatility | sharpe | max_drawdown | sortino | var_95 | cvar_95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| equal_weight | 0.1606 | 0.1414 | 0.8468 | -0.1490 | 1.2565 | 0.0130 | 0.0191 |
| sixty_forty | 0.1204 | 0.1050 | 0.7615 | -0.1072 | 1.1199 | 0.0095 | 0.0145 |
| inverse_vol | 0.1245 | 0.1092 | 0.7701 | -0.1067 | 1.1312 | 0.0099 | 0.0149 |
| min_variance | 0.1189 | 0.1000 | 0.7810 | -0.0860 | 1.1439 | 0.0089 | 0.0138 |
| max_sharpe | 0.1093 | 0.1036 | 0.6739 | -0.0916 | 0.9833 | 0.0096 | 0.0144 |
| risk_parity | 0.1212 | 0.1071 | 0.7558 | -0.1006 | 1.1065 | 0.0101 | 0.0147 |
| hrp | 0.1218 | 0.1072 | 0.7597 | -0.0999 | 1.1164 | 0.0101 | 0.0146 |

## Headline findings

- Best net Sharpe in the saved run: **equal_weight** (0.847); worst: **max_sharpe** (0.674).
- 16 VaR backtest rows reject Kupiec coverage at the configured significance threshold; fhs has the highest observed pass share.
- The most adverse hypothetical scenario shown is Equity crash for equal_weight (-26.00%); shock values are assumptions, not forecasts.

Saved backtest scope: 5 ETF assets, 2023-10-05 to 2026-10-09. The source price panel has 5 assets and covers 2007-06-01 to 2026-10-09.

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

## References

See the [methodology references](docs/methodology.md#references).

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

Career and interview materials: [resume bullets](docs/RESUME_BULLETS.md), [GitHub setup and interview questions](docs/GITHUB_SETUP.md).
