"""Generate resume, LinkedIn, and GitHub setup copy from result files."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = "github.com/randomaban-beep/quantrisk"


def write_resume_materials(test_counts: dict[str, int] | None = None) -> None:
    """Write quantified career materials using saved metrics and QA counts."""
    metrics = pd.read_csv("results/metrics_summary.csv", index_col="strategy")
    var = pd.read_csv("results/var_backtest_summary.csv")
    historical = pd.read_csv("results/stress_historical.csv")
    hypothetical = pd.read_csv("results/stress_hypothetical.csv")
    weights = pd.read_parquet("results/weights.parquet")
    best = metrics["sharpe"].idxmax()
    lo, hi = metrics["sharpe"].min(), metrics["sharpe"].max()
    model_pass = var.groupby("model")["kupiec_pass"].mean().sort_values(ascending=False)
    top_model = str(model_pass.index[0])
    top_pass = float(model_pass.iloc[0])
    tests = test_counts or {"passed": 0, "failed": 0, "collected": 0}
    bullets = [
        f"Built and compared {len(metrics)} long-only portfolio strategies with delayed walk-forward execution, drift-aware weights, and turnover costs across {weights.asset.nunique()} ETFs.",
        f"Produced a dev-scope net Sharpe range of {lo:.2f}–{hi:.2f}; {best} led this sample. Results are exploratory and not evidence of future outperformance.",
        f"Implemented four VaR models at two confidence levels with Kupiec and Christoffersen tests; {top_model} had the highest Kupiec pass share ({top_pass:.0%}).",
        f"Analyzed {historical.scenario.nunique()} historical and {hypothetical.scenario.nunique()} illustrative stress scenarios; automated QA collected {tests.get('collected', 0)} tests ({tests.get('passed', 0)} passed, {tests.get('failed', 0)} failed).",
    ]
    text = "# Resume bullets\n\n" + "\n".join(f"- {bullet}" for bullet in bullets) + f"\n\nRepository: https://{REPO}\n"
    text += "\n## LinkedIn summary\n\nBuilt a Python portfolio-construction and risk engine with seven allocation methods, transaction-aware walk-forward backtesting, formal VaR validation, and stress analysis.\n\nAnalyzed reproducible ETF results with DuckDB, SQL, Streamlit, and Power BI-ready exports; documented execution timing, statistical limitations, and model risks.\n"
    Path("docs/RESUME_BULLETS.md").write_text(text, encoding="utf-8")

    description = "Walk-forward ETF portfolio optimization, VaR backtesting, stress tests, and SQL analytics in Python"
    topics = ["quantitative-finance", "portfolio-optimization", "risk-management", "value-at-risk",
              "hierarchical-risk-parity", "backtesting", "python", "duckdb"]
    command = "gh repo edit randomaban-beep/quantrisk --description \"" + description + "\" " + " ".join(f"--add-topic {topic}" for topic in topics)
    questions = [
        ("Why HRP instead of mean-variance?", "HRP uses a hierarchical correlation structure and avoids direct inversion of a noisy covariance matrix; it remains sensitive to the sample and linkage choice."),
        ("Why Kupiec and Christoffersen?", "Kupiec checks unconditional exception frequency, while Christoffersen tests whether exceptions cluster over time."),
        ("How was look-ahead bias prevented?", "Signals use data through t, execute after the configured lag at the close, and new holdings only earn returns after the execution date."),
        ("What does cost sensitivity show?", "The one-factor runs compare Sharpe, CAGR, drawdown, and turnover across the saved cost assumptions; net results depend on turnover and the simplified cost model."),
        ("Why shrink covariance?", "Ledoit–Wolf shrinks a noisy sample covariance toward a structured target to improve conditioning and reduce estimation variance."),
        ("What is a main limitation?", "The ETF universe and design choices are selected with hindsight, the sample is limited, and backtest performance is not out-of-sample evidence."),
    ]
    setup = f"""# GitHub setup

Description ({len(description)} characters): `{description}`

Topics: {", ".join(topics)}

Apply the description and topics:

```bash
{command}
```

## Interview questions and model answers

""" + "\n".join(f"### {question}\n\n{answer}" for question, answer in questions) + "\n"
    Path("docs/GITHUB_SETUP.md").write_text(setup, encoding="utf-8")
    Path("results/qa_summary.json").write_text(json.dumps(tests, indent=2) + "\n", encoding="utf-8")
