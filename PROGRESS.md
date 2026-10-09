# Progress

- [x] Stage 0: Setup
- [x] Stage 1: Data layer and DuckDB
- [ ] Stage 2: Estimators and portfolio optimizers
- [x] Stage 3: Walk-forward backtest and metrics
- [x] Stage 4: Risk engine and VaR backtests
- [x] Stage 5: Stress, stats, and sensitivity
- [ ] Stage 6: Visuals, SQL, report, dashboard, Power BI
- [ ] Stage 7: Final QA and resume materials

## Decisions
- Stage 0: The workspace was empty. Python project follows the supplied layout.
- Stage 0: Git requires a safe-directory exception in this sandbox because the workspace owner differs from the execution identity; configured project identity is randomaban-beep / GitHub noreply.
- Stage 0: Initial push succeeded to origin/main. The bundled Python runtime lacks scipy and other declared packages; runtime install or a prepared environment will be needed for full validation.
- Stage 1: IWM recovered from Yahoo within the two-attempt cap; Stooq is implemented as its fallback. No IJR swap was needed. Full universe data currently contains 12 tickers, 4,871 price rows, no missing values, returns, rf table, and DuckDB tables.
- Stage 5: Correlation scenarios raise each estimated off-diagonal correlation to at least the configured floor while preserving each asset's volatility.

## Known issues
- Stage 2 implementation is committed with the risk-parity limitation below; it remains unchecked pending that criterion.
- Stage 2 risk-parity criterion remains unresolved after the allowed two tuning attempts: the deterministic fixture's max-to-min risk-contribution spread is about 1.36%, exceeding the 1% criterion. The full suite therefore has one known failure; the other 21 tests pass and Ruff is clean.
- Stage 3 dev run covers the latest 756 observations and first five configured assets. It is not a full-universe investment result.
- Stage 3 includes close-of-day execution, next-day exposure, cost deduction, weights drift, gross/net return streams, NAV, and core metrics. Dedicated no-look-ahead, drift, cost, NAV, and hand-series metrics checks pass.
- Stage 4 dev run generated 500-day-window VaR/ES for four models and two confidence levels, formal coverage summaries, rolling Basel zones, and component VaR. Dedicated VaR model, backtest, decomposition, and no-look-ahead checks pass.
- Stage 3 dev realized backtest covers 2023-present and five ETFs; earlier historical scenario realized-return fields are unavailable (static-weight replays are calculated).

## Next exact step
Implement Stage 6 plots, SQL analytics, automated report, dashboard, and Power BI exports.
