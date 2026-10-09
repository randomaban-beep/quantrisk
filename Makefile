.PHONY: setup data backtest risk stress sensitivity report sql dashboard test lint all dev
setup:
	python -m pip install -r requirements.txt
data:
	python -m quantrisk.cli run --stage data
backtest:
	python -m quantrisk.cli run --stage portfolios
risk:
	python -m quantrisk.cli run --stage risk
stress:
	python -m quantrisk.cli run --stage stress
sensitivity:
	python -m quantrisk.cli run --stage sensitivity
report:
	python -m quantrisk.cli run --stage report
sql:
	python -m quantrisk.cli run --stage all
dashboard:
	streamlit run app/dashboard.py
test:
	pytest -q -x
lint:
	ruff check .
all:
	python -m quantrisk.cli run --stage all
dev:
	python -m quantrisk.cli run --stage all --dev
