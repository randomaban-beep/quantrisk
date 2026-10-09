"""Walk-forward portfolio backtest with delayed execution and transaction costs."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from quantrisk.optimizers import portfolio_weights

LOGGER = logging.getLogger(__name__)
STRATEGIES = ("equal_weight", "sixty_forty", "inverse_vol", "min_variance", "max_sharpe", "risk_parity", "hrp")


def turnover_cost(pre_trade: np.ndarray, target: np.ndarray, cost_bps: float, slippage_bps: float) -> tuple[float, float]:
    """Compute one-way absolute turnover and its proportional transaction cost."""
    turnover = float(np.abs(target - pre_trade).sum())
    return turnover, turnover * (cost_bps + slippage_bps) / 1e4


def _rebalance_positions(index: pd.DatetimeIndex, frequency: str) -> list[int]:
    """Return the final observation position per month or quarter."""
    periods = index.to_period("M" if frequency == "monthly" else "Q")
    return [int(np.flatnonzero(periods == period)[-1]) for period in periods.unique()]


def run_backtest(
    returns: pd.DataFrame,
    config: dict,
    risk_free: pd.Series | None = None,
    strategies: tuple[str, ...] = STRATEGIES,
) -> dict[str, pd.DataFrame]:
    """Run a close-to-close walk-forward backtest and return tidy outputs."""
    window = int(config["estimation_window"])
    lag = int(config["execution_lag_days"])
    max_weight = float(config["max_weight"])
    frequency = str(config["rebalance_freq"])
    if lag < 1:
        raise ValueError("execution_lag_days must be at least 1 to prevent look-ahead")
    signal_positions = [p for p in _rebalance_positions(returns.index, frequency) if p + 1 >= window]
    outputs: dict[str, list] = {key: [] for key in ("net", "gross", "nav", "turnover", "costs", "weights")}
    rf = risk_free.reindex(returns.index).fillna(0.0) if risk_free is not None else pd.Series(0.0, index=returns.index)
    for strategy in strategies:
        holdings = np.zeros(len(returns.columns), dtype=float)
        invested = False
        pending: dict[int, pd.Series] = {}
        for signal_pos in signal_positions:
            activation_pos = signal_pos + lag
            if activation_pos >= len(returns):
                continue
            target = portfolio_weights(
                returns.iloc[signal_pos - window + 1 : signal_pos + 1],
                strategy,
                max_weight=max_weight,
            )
            pending[activation_pos] = target
        strategy_net = []
        strategy_gross = []
        strategy_turnover = []
        strategy_costs = []
        strategy_nav = []
        nav = float(config.get("initial_capital", 1.0))
        for pos, (dt, daily_returns) in enumerate(returns.iterrows()):
            asset_return = daily_returns.to_numpy(dtype=float)
            gross = float(holdings @ asset_return) if invested else float(rf.iloc[pos])
            turnover = 0.0
            trade_cost = 0.0
            if pos in pending:
                target = pending[pos].reindex(returns.columns).fillna(0.0).to_numpy()
                turnover, trade_cost = turnover_cost(holdings, target, float(config["cost_bps"]), float(config["slippage_bps"]))
            cost = trade_cost
            net = gross - cost
            nav *= 1.0 + net
            strategy_gross.append(gross)
            strategy_net.append(net)
            strategy_turnover.append(turnover)
            strategy_costs.append(cost)
            strategy_nav.append(nav)
            # Returns drift the holdings held at the start of the day.
            if invested:
                growth = holdings * (1.0 + asset_return)
                denominator = growth.sum()
                holdings = growth / denominator if denominator else holdings
            # Trades happen at today's close; target weights only earn returns after today.
            if pos in pending:
                target = pending[pos].reindex(returns.columns).fillna(0.0).to_numpy()
                for asset, weight in zip(returns.columns, target, strict=True):
                    outputs["weights"].append((dt, strategy, asset, float(weight), "target"))
                holdings = target
                invested = True
            if invested:
                for asset, weight in zip(returns.columns, holdings, strict=True):
                    outputs["weights"].append((dt, strategy, asset, float(weight), "drifted"))
        index = returns.index
        outputs["net"].append(pd.Series(strategy_net, index=index, name=strategy))
        outputs["gross"].append(pd.Series(strategy_gross, index=index, name=strategy))
        outputs["nav"].append(pd.Series(strategy_nav, index=index, name=strategy))
        outputs["turnover"].append(pd.Series(strategy_turnover, index=index, name=strategy))
        outputs["costs"].append(pd.Series(strategy_costs, index=index, name=strategy))
    return {
        "returns_net": pd.concat(outputs["net"], axis=1),
        "returns_gross": pd.concat(outputs["gross"], axis=1),
        "nav": pd.concat(outputs["nav"], axis=1),
        "turnover": pd.concat(outputs["turnover"], axis=1),
        "costs": pd.concat(outputs["costs"], axis=1),
        "weights": pd.DataFrame(outputs["weights"], columns=["date", "strategy", "asset", "weight", "weight_type"]),
    }
