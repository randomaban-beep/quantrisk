# GitHub setup

Description (99 characters): `Walk-forward ETF portfolio optimization, VaR backtesting, stress tests, and SQL analytics in Python`

Topics: quantitative-finance, portfolio-optimization, risk-management, value-at-risk, hierarchical-risk-parity, backtesting, python, duckdb

Apply the description and topics:

```bash
gh repo edit randomaban-beep/quantrisk --description "Walk-forward ETF portfolio optimization, VaR backtesting, stress tests, and SQL analytics in Python" --add-topic quantitative-finance --add-topic portfolio-optimization --add-topic risk-management --add-topic value-at-risk --add-topic hierarchical-risk-parity --add-topic backtesting --add-topic python --add-topic duckdb
```

## Interview questions and model answers

### Why HRP instead of mean-variance?

HRP uses a hierarchical correlation structure and avoids direct inversion of a noisy covariance matrix; it remains sensitive to the sample and linkage choice.
### Why Kupiec and Christoffersen?

Kupiec checks unconditional exception frequency, while Christoffersen tests whether exceptions cluster over time.
### How was look-ahead bias prevented?

Signals use data through t, execute after the configured lag at the close, and new holdings only earn returns after the execution date.
### What does cost sensitivity show?

The one-factor runs compare Sharpe, CAGR, drawdown, and turnover across the saved cost assumptions; net results depend on turnover and the simplified cost model.
### Why shrink covariance?

Ledoit–Wolf shrinks a noisy sample covariance toward a structured target to improve conditioning and reduce estimation variance.
### What is a main limitation?

The ETF universe and design choices are selected with hindsight, the sample is limited, and backtest performance is not out-of-sample evidence.
