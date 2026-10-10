"""Power BI star-schema CSV exports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


def export_powerbi(results: str | Path = "results", output: str | Path = "powerbi") -> None:
    """Write tidy dimension and fact tables with consistent join keys."""
    root, out = Path(results), Path(output)
    out.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path("config/universe.yaml").read_text(encoding="utf-8"))
    nav = pd.read_parquet(root / "nav.parquet")
    returns = pd.read_parquet(root / "returns_net.parquet")
    weights = pd.read_parquet(root / "weights.parquet")
    strategies = list(nav.columns)
    assets = list(config["tickers"])
    dates = nav.index
    tables = {
        "dim_strategy": pd.DataFrame({"strategy_id": strategies, "strategy_name": strategies}),
        "dim_asset": pd.DataFrame({"asset_id": assets, "ticker": assets,
                                    "asset_class": [config["tickers"][asset] for asset in assets]}),
        "dim_date": pd.DataFrame({"date": dates, "year": dates.year, "month": dates.month,
                                  "quarter": dates.quarter, "day": dates.day}),
        "fact_returns": returns.rename_axis("date").reset_index().melt(id_vars="date", var_name="strategy_id", value_name="return_net"),
        "fact_nav": nav.rename_axis("date").reset_index().melt(id_vars="date", var_name="strategy_id", value_name="nav"),
        "fact_weights": weights.rename(columns={"strategy": "strategy_id", "asset": "asset_id"}),
        "fact_metrics_long": pd.read_csv(root / "metrics_summary.csv", index_col="strategy").rename_axis("strategy_id").reset_index().melt(id_vars="strategy_id", var_name="metric", value_name="value"),
        "fact_var_backtest": pd.read_csv(root / "var_backtest_summary.csv").rename(columns={"strategy": "strategy_id"}),
        "fact_stress": pd.concat([
            pd.read_csv(root / "stress_historical.csv").assign(stress_type="historical"),
            pd.read_csv(root / "stress_hypothetical.csv").assign(stress_type="hypothetical"),
        ], ignore_index=True).rename(columns={"strategy": "strategy_id"}),
        "fact_sensitivity": pd.read_csv(root / "sensitivity.csv").rename(columns={"strategy": "strategy_id"}),
    }
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    (out / "README.md").write_text(
        """# Power BI star schema

`dim_strategy[strategy_id]` joins strategy facts. `dim_asset[asset_id]` joins weights and asset-level tables. `dim_date[date]` joins date-grain facts. `fact_metrics_long` stores metric/value pairs.

Suggested visuals (DAX-free): NAV line, monthly return matrix, allocation stacked area, asset-class weight bars, VaR exception counts, Kupiec p-value heatmap, stress scenario P&L bars, and sensitivity Sharpe lines.
""",
        encoding="utf-8",
    )
