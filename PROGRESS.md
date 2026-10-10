# Progress

- [x] Stage 0: Setup
- [x] Stage 1: Data layer and DuckDB
- [x] Stage 2: Estimators and portfolio optimizers
- [x] Stage 3: Walk-forward backtest and metrics
- [x] Stage 4: Risk engine and VaR backtests
- [x] Stage 5: Stress, stats, and sensitivity
- [x] Stage 6: Visuals, SQL, report, dashboard, Power BI
- [ ] Stage 7: Final QA and resume materials

## Decisions
- Stage 0: The workspace was empty. Python project follows the supplied layout.
- Stage 0: Git requires a safe-directory exception in this sandbox because the workspace owner differs from the execution identity; configured project identity is randomaban-beep / GitHub noreply.
- Stage 0: Initial push succeeded to origin/main. The bundled Python runtime lacks scipy and other declared packages; runtime install or a prepared environment will be needed for full validation.
- Stage 1: IWM recovered from Yahoo within the two-attempt cap; Stooq is implemented as its fallback. No IJR swap was needed. Full universe data currently contains 12 tickers, 4,871 price rows, no missing values, returns, rf table, and DuckDB tables.
- Stage 5: Correlation scenarios raise each estimated off-diagonal correlation to at least the configured floor while preserving each asset's volatility.
- Stage 6: Dev analytics use the latest 756 dates and five assets for backtest/risk; historical static-weight stress and figures can use the full saved panel.
- Stage 7: The ordered `--stage all --dev` CLI workflow completed from cached real market data. GNU `make` is unavailable in this Windows environment, so Makefile targets could not be invoked directly.
- Stage 7: Resume bullets, LinkedIn summary, GitHub setup, topic command, and six interview Q&As were generated from saved results and QA counts.
- Stage 2: Risk parity now solves equal variance contributions in log-weight coordinates using the sample covariance, matching the covariance used to report contributions. The capped case retains a constrained SLSQP fallback.
- Stage 7: Risk-parity regression is fixed; full suite passes (23/23), Ruff passes, and the complete dev pipeline was rerun to refresh saved outputs.
- Stage 7: The five-asset dev universe includes SPY, IWM (or IJR fallback), and IEF so the 60/40 benchmark has both legs.
- Stage 6: All 12 figures render at 150 dpi; SQL exports six query outputs; Power BI exports ten CSV tables plus a schema guide; report and README are generated from saved result files.

## Known issues
- Stage 7: `make all` from a clean clone was not directly verified because GNU make is unavailable in this Windows environment; the complete `--stage all --dev` CLI sequence passed.
- Stage 3 dev run covers the latest 756 observations and first five configured assets. It is not a full-universe investment result.
- Stage 3 includes close-of-day execution, next-day exposure, cost deduction, weights drift, gross/net return streams, NAV, and core metrics. Dedicated no-look-ahead, drift, cost, NAV, and hand-series metrics checks pass.
- Stage 4 dev run generated 500-day-window VaR/ES for four models and two confidence levels, formal coverage summaries, rolling Basel zones, and component VaR. Dedicated VaR model, backtest, decomposition, and no-look-ahead checks pass.
- Stage 3 dev realized backtest covers 2023-present and five ETFs; earlier historical scenario realized-return fields are unavailable (static-weight replays are calculated).
- Dashboard startup was verified with Streamlit on localhost. Power BI table join keys were checked and have no missing values.
- Full suite result: 23 passed, 0 failed; Ruff is clean. Clean-clone `make all` remains unverified because GNU make is unavailable.

## Next exact step
Run `make all` from a clean clone in an environment with GNU make; if it reproduces every generated artifact, check off the remaining Stage 7 item.

## Stage 7 QA checklist

- [ ] `make all` from clean clone completes (GNU make unavailable; CLI dev sequence passed)
- [x] Full test suite passes (23 passed); `make lint` equivalent passes
- [x] No-look-ahead tests for weights and VaR pass
- [x] All seven strategies produce bounded, fully invested weights in the dev backtest
- [x] README and report figures/metrics are generated from saved result files
- [x] Twelve figures exist and render; dashboard starts; Power BI join keys are non-null
- [x] Limitations are stated; caches and market data are gitignored; no tracked file exceeds 5 MB
- [x] CI workflow exists and Ruff is clean
- [x] CI test job passes locally (23 tests); remote workflow status not queried
- [x] Stage commits are pushed and local main matches origin/main
- [x] README contains CI badge, hero figure, and relative links
