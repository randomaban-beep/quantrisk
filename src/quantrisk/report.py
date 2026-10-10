"""Result-driven report and README generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

LIMITATIONS = (
    "ETF universe chosen with hindsight and limited history",
    "Results are in-sample with respect to design choices",
    "Simplified cost model and no market impact",
    "Long-only 40% cap changes textbook unconstrained optimizer results",
    "Expected-return estimation error affects maximum Sharpe",
    "VaR models assume stable relationships within each estimation window",
    "Hypothetical shocks are illustrative assumptions, not forecasts",
)


def build_report(results: str | Path = "results", figures: str | Path = "figures") -> str:
    """Generate Markdown report and project README from saved numeric outputs."""
    root = Path(results)
    metrics = pd.read_csv(root / "metrics_summary.csv", index_col="strategy")
    var = pd.read_csv(root / "var_backtest_summary.csv")
    sensitivity = pd.read_csv(root / "sensitivity.csv")
    stress = pd.read_csv(root / "stress_hypothetical.csv")
    data_quality = pd.read_csv(root / "data_quality.csv").iloc[0]
    weights = pd.read_parquet(root / "weights.parquet")
    nav = pd.read_parquet(root / "nav.parquet")
    backtest_assets = int(weights["asset"].nunique())
    backtest_start = nav.index.min().date()
    backtest_end = nav.index.max().date()
    best = metrics["sharpe"].idxmax()
    worst = metrics["sharpe"].idxmin()
    failures = int((~var["kupiec_pass"]).sum())
    best_var = var.groupby("model")["kupiec_pass"].mean().sort_values(ascending=False)
    best_model = str(best_var.index[0]) if len(best_var) else "unavailable"
    best_scenario = stress.loc[stress.groupby("scenario").portfolio_pnl.idxmin()].sort_values("portfolio_pnl").iloc[0]
    table_frame = metrics[["cagr", "volatility", "sharpe", "max_drawdown", "sortino", "var_95", "cvar_95"]]
    headers = ["strategy", *table_frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for strategy, row in table_frame.iterrows():
        lines.append("| " + " | ".join([str(strategy), *(f"{float(value):.4f}" for value in row)]) + " |")
    table = "\n".join(lines)
    report = f"""# QuantRisk results

## Executive summary

- The highest observed net Sharpe in this run was **{best}** at {metrics.loc[best, 'sharpe']:.3f}; the lowest was **{worst}** at {metrics.loc[worst, 'sharpe']:.3f}.
- Across the VaR backtest rows, {failures} Kupiec tests rejected nominal coverage; {best_model} had the highest Kupiec pass share.
- The most adverse illustrative shock in the saved table was **{best_scenario['scenario']}** for {best_scenario['strategy']} ({best_scenario['portfolio_pnl']:.2%}).
- The source panel has {int(data_quality['assets'])} assets and {int(data_quality['rows'])} price rows; the saved backtest uses {backtest_assets} assets from {backtest_start} through {backtest_end}. See limitations below before interpreting performance.

## Data and universe

Adjusted ETF close data from Yahoo Finance with a Stooq fallback; simple returns, short-gap cleaning, and DuckDB persistence. The current saved data-quality table records {int(data_quality['rows'])} observations from {data_quality['start']} through {data_quality['end']} with {int(data_quality['missing_values'])} missing prices.

## Methodology

See [methodology](../docs/methodology.md). The backtest is walk-forward with delayed close execution, daily weight drift, and turnover-based costs.

## Performance

{table}

## Risk, drawdowns, and VaR

See `var_backtest_summary.csv` for formal coverage and independence tests. The observed pass share ranks {best_model} highest by Kupiec coverage; it is a descriptive ranking and does not establish universal model superiority.

![NAV](../figures/01_cumulative_nav.png)

![Drawdown](../figures/02_underwater.png)

## Stress results

Scenario P&L is `w · shock` and the hypothetical inputs are labeled as illustrative assumptions. The result table is in `stress_hypothetical.csv`.

## Statistical significance and sensitivity

Paired Sharpe comparisons and their explicit significance labels are in `sharpe_paired_tests.csv`. One-factor runs are in `sensitivity.csv` ({len(sensitivity)} rows); runs are cached by configuration hash.

## Limitations

""" + "\n".join(f"- {item}" for item in LIMITATIONS) + "\n"
    report += "\n## Reproducibility\n\nRun `make setup`, `make data`, `make backtest`, `make risk`, `make stress`, `make sensitivity`, and `make report`. The saved result files and configuration YAMLs are the source for numeric tables.\n"
    (root / "report.md").write_text(report, encoding="utf-8")
    readme = f"""# quantrisk

Portfolio construction and risk management engine for liquid ETFs.

![CI](https://github.com/randomaban-beep/quantrisk/actions/workflows/ci.yml/badge.svg)

![Net NAV](figures/01_cumulative_nav.png)

Walk-forward portfolio construction with seven strategy families, transaction costs, VaR/CVaR forecasts, formal coverage backtests, stress tests, sensitivity analysis, DuckDB, SQL, and Power BI-ready tables.

## Key results (generated from `results/metrics_summary.csv`)

{table}

## Headline findings

- Best net Sharpe in the saved run: **{best}** ({metrics.loc[best, 'sharpe']:.3f}); worst: **{worst}** ({metrics.loc[worst, 'sharpe']:.3f}).
- {failures} VaR backtest rows reject Kupiec coverage at the configured significance threshold; {best_model} has the highest observed pass share.
- The most adverse hypothetical scenario shown is {best_scenario['scenario']} for {best_scenario['strategy']} ({best_scenario['portfolio_pnl']:.2%}); shock values are assumptions, not forecasts.

Saved backtest scope: {backtest_assets} ETF assets, {backtest_start} to {backtest_end}. The source price panel has {int(data_quality['assets'])} assets and covers {data_quality['start']} to {data_quality['end']}.

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

"""
    readme += "\n".join(f"- {item}" for item in LIMITATIONS) + "\n"

    readme += """

## How to talk about this project

Explain the lagged walk-forward signal timing, covariance estimation choices, optimizer constraints, formal VaR coverage tests, and why a high backtest metric is not proof of future performance.
"""
    Path("README.md").write_text(readme, encoding="utf-8")
    return report
