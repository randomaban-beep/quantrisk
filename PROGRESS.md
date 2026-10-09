# Progress

- [x] Stage 0: Setup
- [x] Stage 1: Data layer and DuckDB
- [ ] Stage 2: Estimators and portfolio optimizers
- [x] Stage 3: Walk-forward backtest and metrics
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
- Stage 2 implementation is committed with the risk-parity limitation below; it remains unchecked pending that criterion.
- Stage 2 risk-parity criterion remains unresolved after the allowed two tuning attempts: the deterministic fixture's max-to-min risk-contribution spread is about 1.36%, exceeding the 1% criterion. The full suite therefore has one known failure; the other 9 tests pass and Ruff is clean.
- Stage 3 dev run covers the latest 756 observations and first five configured assets. It is not a full-universe investment result.
- Stage 3 includes close-of-day execution, next-day exposure, cost deduction, weights drift, gross/net return streams, NAV, and core metrics. Dedicated no-look-ahead, drift, cost, NAV, and hand-series metrics checks pass.
- Stage 2 in progress: optimizer analytics and tests are being added after Stage 1 push `082494e`.
- Stage 2: risk parity missed the strict 1% risk-contribution spread in its deterministic fixture after two implementation attempts (observed spread about 1.36%); logged per the two-fix-attempt limit. All other optimizer property tests passed and the latest-window run completed.

## Next exact step
Implement the Stage 4 rolling VaR/CVaR forecasts and formal exception backtests.
