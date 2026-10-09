# Project goal
Build a reproducible portfolio construction and risk engine for liquid ETFs.
Implement walk-forward strategies, risk validation, stress analysis, and reporting.
Keep results honest, deterministic, tested, and suitable for technical interviews.

# Folder map
- `src/quantrisk/`: Python library and CLI
- `config/`: YAML settings; `tests/`: offline tests
- `docs/`: specification and methodology
- `results/`, `figures/`, `powerbi/`: generated deliverables
- `data/`: local inputs and DuckDB (gitignored); `app/`: dashboard; `sql/`: analytics

# Commands
Use `make setup`, `make test`, `make lint`, `make data`, `make backtest`,
`make risk`, `make stress`, `make sensitivity`, `make report`, `make sql`,
`make dashboard`, `make dev`, and `make all`.

# Coding rules
- Python 3.11+, type hints, concise docstrings, deterministic seed 42.
- Configure through `config/*.yaml`; avoid hardcoded paths and magic numbers.
- Use logging in library code; never fabricate market data.
- Prevent look-ahead bias; report limitations honestly.
- Read `PROGRESS.md` first, then only the relevant stage section of `docs/SPEC.md`. Never re-read the whole spec.
