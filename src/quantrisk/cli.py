"""Command-line entry point for quantrisk."""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from quantrisk.data import MissingTickersError, calculate_returns, download_prices, load_risk_free
from quantrisk.db import store_frame


def main() -> None:
    """Parse arguments and dispatch a project stage."""
    parser = argparse.ArgumentParser(prog="quantrisk")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument(
        "--stage",
        choices=("data", "portfolios", "risk", "stress", "sensitivity", "report", "all"),
        default="all",
    )
    run.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.stage == "data":
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
    else:
        logging.getLogger(__name__).info("Stage '%s' is not implemented yet.", args.stage)


if __name__ == "__main__":
    main()
