# Progress

- [x] Stage 0: Setup
- [x] Stage 1: Data layer and DuckDB
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
- Stage 1: IWM recovered from Yahoo within the two-attempt cap; Stooq is implemented as its fallback. No IJR swap was needed. Full universe data currently contains 12 tickers, 4,871 price rows, no missing values, returns, rf table, and DuckDB tables.

## Known issues
- Stage 2 is not complete: estimators and core allocators exist, but analytics helpers, specified optimizer tests, and validation remain.
- Test suite passes (3 tests) and Ruff is clean in `.venv312`; Stage 2 coverage is still pending.
- Performance backtests and reports do not exist yet; market data does not imply strategy results.

## Next exact step
Implement the remaining Stage 2 portfolio analytics and optimizer tests, then run tests/lint and commit/push.
