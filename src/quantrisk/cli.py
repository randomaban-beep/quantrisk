"""Command-line entry point for quantrisk."""

from __future__ import annotations

import argparse
import logging


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
    logging.getLogger(__name__).info("Stage '%s' is not implemented yet.", args.stage)


if __name__ == "__main__":
    main()
