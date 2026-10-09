"""Cached one-at-a-time configuration sensitivity analysis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from quantrisk.backtest import run_backtest
from quantrisk.metrics import performance_metrics


def run_sensitivity(returns: pd.DataFrame, baseline: dict) -> pd.DataFrame:
    """Run one-factor variations and cache output rows by configuration hash."""
    variants = []
    for key, values in {
        "estimation_window": (126, 252, 504),
        "rebalance_freq": ("monthly", "quarterly"),
        "cost_total_bps": (0, 7, 15, 30),
        "covariance_method": ("sample", "ledoit_wolf", "ewma"),
        "max_weight": (0.25, 0.40, 1.0),
    }.items():
        for value in values:
            if key == "estimation_window" and value == baseline["estimation_window"]:
                continue
            if key == "rebalance_freq" and value == baseline["rebalance_freq"]:
                continue
            if key == "cost_total_bps" and value == baseline["cost_bps"] + baseline["slippage_bps"]:
                continue
            if key == "covariance_method" and value == baseline.get("covariance_method", "ledoit_wolf"):
                continue
            if key == "max_weight" and value == baseline["max_weight"]:
                continue
            variant = dict(baseline)
            if key == "cost_total_bps":
                variant["cost_bps"], variant["slippage_bps"] = value, 0
            else:
                variant[key] = value
            variants.append((key, value, variant))
    cache_dir = Path("data/cache/sensitivity")
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, value, cfg in variants:
        key_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
        cache_file = cache_dir / f"{key_hash}.csv"
        if cache_file.exists():
            rows.extend(pd.read_csv(cache_file).to_dict("records"))
            continue
        result = run_backtest(returns, cfg)
        output = []
        for strategy in result["returns_net"]:
            active_start = result["weights"].loc[
                (result["weights"].strategy == strategy) & (result["weights"].weight_type == "target"), "date"
            ].min()
            metrics = performance_metrics(result["returns_net"].loc[result["returns_net"].index > active_start, strategy])
            output.append({"factor": key, "value": value, "strategy": strategy,
                           "sharpe": metrics["sharpe"], "cagr": metrics["cagr"],
                           "max_drawdown": metrics["max_drawdown"],
                           "annualized_turnover": float(result["turnover"][strategy].mean() * 252)})
        pd.DataFrame(output).to_csv(cache_file, index=False)
        rows.extend(output)
    return pd.DataFrame(rows)
