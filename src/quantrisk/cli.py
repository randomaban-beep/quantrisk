"""Command-line entry point for quantrisk."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from quantrisk.backtest import run_backtest
from quantrisk.data import MissingTickersError, calculate_returns, download_prices, load_risk_free
from quantrisk.db import store_frame
from quantrisk.exports import export_powerbi
from quantrisk.metrics import performance_metrics
from quantrisk.optimizers import portfolio_weights
from quantrisk.plots import make_figures
from quantrisk.report import build_report
from quantrisk.sensitivity import run_sensitivity
from quantrisk.stats import stationary_bootstrap
from quantrisk.stress import correlation_stress, historical_stress, hypothetical_stress
from quantrisk.var_backtests import summarize_var, traffic_light
from quantrisk.var_models import component_var, rolling_forecasts


def main() -> None:
    """Parse arguments and dispatch a project stage."""
    parser = argparse.ArgumentParser(prog="quantrisk")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument(
        "--stage",
        choices=("data", "portfolios", "risk", "stress", "sensitivity", "sql", "report", "all"),
        default="all",
    )
    run.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.stage == "all":
        for stage in ("data", "portfolios", "risk", "stress", "sensitivity", "sql", "report"):
            command = [sys.executable, "-m", "quantrisk.cli", "run", "--stage", stage]
            if args.dev:
                command.append("--dev")
            subprocess.run(command, check=True)
    elif args.stage == "data":
        config = yaml.safe_load(Path("config/universe.yaml").read_text(encoding="utf-8"))
        configured = config["tickers"]
        tickers = list(configured)
        if args.dev:
            tickers = tickers[:5]
        try:
            prices = download_prices(tickers, config["start"], date.today().isoformat())
        except MissingTickersError as exc:
            if "IWM" not in exc.tickers or "IWM" not in configured:
                raise
            # Stooq was already attempted in download_prices. Replace the unavailable symbol.
            asset_class = configured.pop("IWM")
            configured["IJR"] = asset_class
            config_path = Path("config/universe.yaml")
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            progress_path = Path("PROGRESS.md")
            with progress_path.open("a", encoding="utf-8") as progress:
                progress.write("\n- Stage 1: Yahoo and Stooq both failed for IWM after two Yahoo attempts; replaced IWM with IJR (US small-cap ETF) and updated the universe configuration.\n")
            logging.getLogger(__name__).warning("Replaced IWM with IJR after Yahoo and Stooq failures")
            tickers = ["IJR" if ticker == "IWM" else ticker for ticker in tickers]
            prices = download_prices(tickers, config["start"], date.today().isoformat())
        returns = calculate_returns(prices)
        rf = load_risk_free(config["start"], date.today().isoformat())
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        returns.to_parquet("data/processed/returns.parquet")
        store_frame(prices, "prices")
        store_frame(returns, "returns")
        store_frame(rf.to_frame(), "risk_free")
        summary = pd.DataFrame({
            "rows": [len(prices)],
            "start": [prices.index.min()],
            "end": [prices.index.max()],
            "missing_values": [int(prices.isna().sum().sum())],
            "assets": [len(prices.columns)],
        })
        Path("results").mkdir(exist_ok=True)
        summary.to_csv("results/data_quality.csv", index=False)
        logging.getLogger(__name__).info("Saved %d rows for %d assets", len(prices), len(prices.columns))
    elif args.stage == "portfolios":
        returns = pd.read_parquet("data/processed/returns.parquet")
        config = yaml.safe_load(Path("config/backtest.yaml").read_text(encoding="utf-8"))
        Path("results").mkdir(exist_ok=True)
        if args.dev:
            config["estimation_window"] = min(252, len(returns) // 2)
            returns = returns.tail(756).iloc[:, :5]
        rf_path = Path("data/raw/rf.parquet")
        rf = pd.read_parquet(rf_path).iloc[:, 0] if rf_path.exists() else None
        outputs = run_backtest(returns, config, rf)
        for name, frame in outputs.items():
            if name == "weights":
                frame.to_parquet("results/weights.parquet", index=False)
            elif name in {"turnover", "costs"}:
                frame.to_csv(f"results/{name}.csv", index_label="date")
            else:
                frame.to_parquet(f"results/{name}.parquet")
        outputs["nav"].to_parquet("results/nav.parquet")
        metric_rows = {}
        for strategy in outputs["returns_net"]:
            trade_dates = outputs["weights"].loc[
                (outputs["weights"]["strategy"] == strategy)
                & (outputs["weights"]["weight_type"] == "target"), "date"
            ]
            first_active = trade_dates.min()
            active_index = outputs["returns_net"].index > first_active
            metric_rows[strategy] = performance_metrics(
                outputs["returns_net"].loc[active_index, strategy], rf
            )
        metrics = pd.DataFrame(metric_rows).T
        metrics.index.name = "strategy"
        metrics.to_csv("results/metrics_summary.csv")
        latest_window = returns.tail(int(config["estimation_window"]))
        targets = pd.concat([portfolio_weights(latest_window, strategy, float(config["max_weight"])).rename(strategy) for strategy in ("equal_weight", "sixty_forty", "inverse_vol", "min_variance", "max_sharpe", "risk_parity", "hrp")], axis=1)
        targets.index.name = "asset"
        targets.to_csv("results/portfolio_weights_latest.csv")
        logging.getLogger(__name__).info("Backtest saved for %d strategies", len(outputs["returns_net"].columns))
    elif args.stage == "risk":
        returns = pd.read_parquet("data/processed/returns.parquet")
        nav = pd.read_parquet("results/returns_gross.parquet")
        weight_rows = pd.read_parquet("results/weights.parquet")
        settings = yaml.safe_load(Path("config/risk.yaml").read_text(encoding="utf-8"))
        if args.dev:
            returns = returns.tail(756).iloc[:, :5]
        forecast_frames = []
        traffic_frames = []
        for strategy in nav.columns:
            drifted = weight_rows[(weight_rows.strategy == strategy) & (weight_rows.weight_type == "drifted")]
            close_weights = drifted.pivot(index="date", columns="asset", values="weight")
            held = close_weights.reindex(returns.index).ffill().shift(1).fillna(0.0)
            forecasts = rolling_forecasts(
                returns, held, int(settings["window"]), tuple(settings["confidence_levels"]), float(settings["ewma_lambda"])
            )
            if forecasts.empty:
                continue
            forecasts.insert(1, "strategy", strategy)
            forecast_frames.append(forecasts)
            tail = forecasts[forecasts.confidence == 0.99].set_index("date")
            zones = traffic_light(tail["realized"] < -tail["var"], 250)
            traffic_frames.append(pd.DataFrame({"date": zones.index, "strategy": strategy, "zone": zones.astype("string").values}))
        if forecast_frames:
            forecasts = pd.concat(forecast_frames, ignore_index=True)
            Path("results").mkdir(exist_ok=True)
            forecasts.to_parquet("results/var_forecasts.parquet", index=False)
            summary = summarize_var(forecasts, float(settings["significance"]))
            summary.to_csv("results/var_backtest_summary.csv", index=False)
            ranking = summary.groupby("strategy").agg(passed=("kupiec_pass", "sum"), models=("model", "count")).sort_values("passed", ascending=False)
            ranking.to_csv("results/var_model_ranking.csv")
            pd.concat(traffic_frames, ignore_index=True).to_csv("results/basel_traffic_light.csv", index=False)
            latest = pd.read_csv("results/portfolio_weights_latest.csv", index_col="asset")
            components = []
            for strategy in latest.columns:
                part = component_var(latest[strategy], returns.tail(int(settings["window"])))
                part.insert(0, "strategy", strategy)
                components.append(part)
            pd.concat(components, ignore_index=True).to_csv("results/component_risk_latest.csv", index=False)
            logging.getLogger(__name__).info("Saved VaR forecasts for %d strategies", len(forecast_frames))
    elif args.stage == "stress":
        returns = pd.read_parquet("data/processed/returns.parquet")
        strategy_returns = pd.read_parquet("results/returns_net.parquet")
        config = yaml.safe_load(Path("config/backtest.yaml").read_text(encoding="utf-8"))
        stress_config = yaml.safe_load(Path("config/stress.yaml").read_text(encoding="utf-8"))
        weights_by_strategy = {
            strategy: portfolio_weights(returns.tail(int(config["estimation_window"])), strategy,
                                        float(config["max_weight"]), covariance_method=config.get("covariance_method", "ledoit_wolf"))
            for strategy in ("equal_weight", "sixty_forty", "inverse_vol", "min_variance", "max_sharpe", "risk_parity", "hrp")
        }
        historical, hypothetical, correlations = [], [], []
        for strategy, weights in weights_by_strategy.items():
            hist = historical_stress(weights, returns, strategy_returns.get(strategy))
            hist.insert(0, "strategy", strategy)
            historical.append(hist)
            hypo = hypothetical_stress(weights, stress_config["hypothetical"])
            hypo.insert(0, "strategy", strategy)
            hypothetical.append(hypo)
            corr = correlation_stress(weights, returns.tail(int(config["estimation_window"])), tuple(stress_config["correlation_levels"]))
            corr.insert(0, "strategy", strategy)
            correlations.append(corr)
        pd.concat(historical, ignore_index=True).to_csv("results/stress_historical.csv", index=False)
        pd.concat(hypothetical, ignore_index=True).to_csv("results/stress_hypothetical.csv", index=False)
        pd.concat(correlations, ignore_index=True).to_csv("results/stress_correlation.csv", index=False)
        latest = pd.DataFrame(weights_by_strategy)
        latest.index.name = "asset"
        latest.to_csv("results/portfolio_weights_latest.csv")
        returns_net = strategy_returns
        active_returns = returns_net.copy()
        weight_rows = pd.read_parquet("results/weights.parquet")
        for strategy in active_returns:
            trade_dates = weight_rows.loc[
                (weight_rows.strategy == strategy) & (weight_rows.weight_type == "target"), "date"
            ]
            if len(trade_dates):
                active_returns.loc[active_returns.index <= trade_dates.min(), strategy] = float("nan")
        rf_path = Path("data/raw/rf.parquet")
        rf = pd.read_parquet(rf_path).iloc[:, 0] if rf_path.exists() else pd.Series(0.0, index=active_returns.index)
        excess = active_returns.sub(rf.reindex(active_returns.index).fillna(0.0), axis=0)
        bootstrap, paired = stationary_bootstrap(excess, resamples=5000, block_length=21, seed=42)
        bootstrap.to_csv("results/sharpe_bootstrap.csv", index=False)
        paired.to_csv("results/sharpe_paired_tests.csv", index=False)
        subperiod_rows = []
        periods = {"2008-09 crisis": ("2008-01-01", "2009-12-31"), "2010-2019 expansion": ("2010-01-01", "2019-12-31"),
                   "2020 COVID": ("2020-01-01", "2020-12-31"), "2022 rate shock": ("2022-01-01", "2022-12-31"),
                   "2023-present": ("2023-01-01", date.today().isoformat())}
        for strategy in active_returns:
            series_active = active_returns[strategy].dropna()
            for year, series in series_active.groupby(series_active.index.year):
                subperiod_rows.append({"strategy": strategy, "period": str(year), "return": float((1.0 + series).prod() - 1.0)})
            for label, (start, end) in periods.items():
                series = series_active.loc[start:end]
                if len(series):
                    subperiod_rows.append({"strategy": strategy, "period": label, "return": float((1.0 + series).prod() - 1.0)})
        pd.DataFrame(subperiod_rows).to_csv("results/subperiods.csv", index=False)
        logging.getLogger(__name__).info("Saved stress, bootstrap, and subperiod analysis")
    elif args.stage == "sensitivity":
        returns = pd.read_parquet("data/processed/returns.parquet")
        config = yaml.safe_load(Path("config/backtest.yaml").read_text(encoding="utf-8"))
        if args.dev:
            returns = returns.tail(756).iloc[:, :5]
            config["estimation_window"] = min(252, len(returns) // 2)
        run_sensitivity(returns, config).to_csv("results/sensitivity.csv", index=False)
        logging.getLogger(__name__).info("Saved cached sensitivity runs")
    elif args.stage == "sql":
        import duckdb

        root = Path("results")
        nav = pd.read_parquet(root / "nav.parquet")
        returns = pd.read_parquet(root / "returns_net.parquet")
        weights = pd.read_parquet(root / "weights.parquet")
        asset_returns = pd.read_parquet("data/processed/returns.parquet")
        var = pd.read_parquet(root / "var_forecasts.parquet")
        turnover = pd.read_csv(root / "turnover.csv", parse_dates=["date"]).melt(id_vars="date", var_name="strategy", value_name="turnover")
        costs = pd.read_csv(root / "costs.csv", parse_dates=["date"]).melt(id_vars="date", var_name="strategy", value_name="cost")
        turnover_costs = turnover.merge(costs, on=["date", "strategy"])
        db_path = Path("data/quant.duckdb")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with duckdb.connect(str(db_path)) as conn:
            for name, frame in {
                "daily_nav": nav.rename_axis("date").reset_index().melt(id_vars="date", var_name="strategy", value_name="nav"),
                "strategy_returns": returns.rename_axis("date").reset_index().melt(id_vars="date", var_name="strategy", value_name="return_net"),
                "daily_weights": weights[weights.weight_type == "drifted"].rename(columns={"asset": "asset"}),
                "asset_returns": asset_returns.rename_axis("date").reset_index().melt(id_vars="date", var_name="asset", value_name="asset_return"),
                "var_forecasts": var,
                "subperiods": pd.read_csv(root / "subperiods.csv"),
                "turnover_costs": turnover_costs,
            }.items():
                conn.register("input_frame", frame)
                conn.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM input_frame')
                conn.unregister("input_frame")
            output_dir = root / "sql"
            output_dir.mkdir(exist_ok=True)
            for query_path in sorted(Path("sql").glob("*.sql")):
                query = conn.execute(query_path.read_text(encoding="utf-8")).df()
                query.to_csv(output_dir / f"{query_path.stem}.csv", index=False)
        logging.getLogger(__name__).info("Executed six SQL analytics queries")
    elif args.stage == "report":
        returns = pd.read_parquet("data/processed/returns.parquet")
        make_figures(returns=returns)
        export_powerbi()
        build_report()
        logging.getLogger(__name__).info("Generated figures, report, README, and Power BI tables")
    else:
        logging.getLogger(__name__).info("Stage '%s' is not implemented yet.", args.stage)


if __name__ == "__main__":
    main()
