# Power BI star schema

`dim_strategy[strategy_id]` joins strategy facts. `dim_asset[asset_id]` joins weights and asset-level tables. `dim_date[date]` joins date-grain facts. `fact_metrics_long` stores metric/value pairs.

Suggested visuals (DAX-free): NAV line, monthly return matrix, allocation stacked area, asset-class weight bars, VaR exception counts, Kupiec p-value heatmap, stress scenario P&L bars, and sensitivity Sharpe lines.
