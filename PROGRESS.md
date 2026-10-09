# Progress

- [x] Stage 0: Setup
- [ ] Stage 1: Data layer and DuckDB
- [ ] Stage 2: Estimators and portfolio optimizers
- [ ] Stage 3: Walk-forward backtest and metrics
- [ ] Stage 4: Risk engine and VaR backtests
- [ ] Stage 5: Stress, stats, and sensitivity
- [ ] Stage 6: Visuals, SQL, report, dashboard, Power BI
- [ ] Stage 7: Final QA and resume materials

## Decisions
- Stage 0: The workspace was empty. Python project follows the supplied layout.
- Stage 0: Git requires a safe-directory exception in this sandbox because the workspace owner differs from the execution identity; configured project identity is randomaban-beep / GitHub noreply.
- Stage 0: Initial push succeeded to origin/main. The bundled Python runtime lacks scipy and other declared packages; runtime install or a prepared environment will be needed for full validation.

## Known issues
- Stage 1 is not complete: the data CLI, parquet cache, quality summary, and DuckDB loader exist, but a dev run downloaded only 4 of 5 tickers; IWM failed. No full-universe run or reportable backtest is available.
- Stage 2 is not complete: estimators and core allocators exist, but analytics helpers, specified optimizer tests, and validation remain.
- Tests currently pass (2 tests) and Ruff is clean in `.venv312`; test coverage is only for data cleaning and return calculation.
- Push pending: implementation changes are not yet committed/pushed.

## Next exact step
Resolve the partial IWM download with a real-data retry, validate all 12 tickers, finish Stage 1, and push its commit.
