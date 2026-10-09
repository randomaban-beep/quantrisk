"""Deterministic, network-free tests for portfolio primitives."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from quantrisk.backtest import run_backtest, turnover_cost
from quantrisk.data import calculate_returns, clean_prices
from quantrisk.metrics import performance_metrics
from quantrisk.optimizers import portfolio_analytics, portfolio_weights


def test_clean_prices_fills_short_gaps_and_sorts() -> None:
    index = pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-02"])
    prices = pd.DataFrame({"A": [3.0, 1.0, np.nan]}, index=index)
    result = clean_prices(prices)
    assert result.index.is_monotonic_increasing
    assert not result.isna().any().any()
    assert result.loc["2020-01-02", "A"] == 1.0


def test_returns_are_simple_daily_returns() -> None:
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0]})
    returns = calculate_returns(prices)
    np.testing.assert_allclose(returns["A"], [0.1, -0.1])


def test_cached_prices_do_not_download(tmp_path) -> None:
    cached = pd.DataFrame({"SPY": [100.0, 101.0]}, index=pd.date_range("2024-01-01", periods=2))
    path = tmp_path / "prices.parquet"
    cached.to_parquet(path)
    with patch("yfinance.download", side_effect=AssertionError("network called")):
        from quantrisk.data import download_prices

        actual = download_prices(["SPY"], "2024-01-01", cache_path=path)
    pd.testing.assert_frame_equal(actual, cached, check_freq=False)


def test_closed_form_equal_weight_and_inverse_vol() -> None:
    returns = pd.DataFrame({"A": [0.01, 0.02, -0.01], "B": [0.02, 0.01, -0.02]})
    equal = portfolio_weights(returns, "equal_weight", max_weight=0.6)
    inverse = portfolio_weights(returns, "inverse_vol", max_weight=0.6)
    np.testing.assert_allclose(equal, [0.5, 0.5], atol=1e-12)
    expected = 1 / returns.std().to_numpy()
    expected /= expected.sum()
    np.testing.assert_allclose(inverse, expected, atol=1e-12)


def test_all_optimizers_respect_sum_and_bounds() -> None:
    rng = np.random.default_rng(42)
    panel = pd.DataFrame(rng.normal(0.0002, 0.01, size=(300, 5)), columns=list("ABCDE"))
    strategies = ("equal_weight", "sixty_forty", "inverse_vol", "min_variance", "max_sharpe", "risk_parity", "hrp")
    for strategy in strategies:
        weights = portfolio_weights(panel, strategy, max_weight=0.4)
        assert np.isfinite(weights).all()
        assert weights.sum() == pytest.approx(1.0, abs=1e-8)
        assert weights.max() <= 0.4 + 1e-8
        assert weights.min() >= -1e-10


def test_min_variance_not_worse_than_equal_weight() -> None:
    rng = np.random.default_rng(42)
    panel = pd.DataFrame(rng.normal(size=(400, 5)), columns=list("ABCDE"))
    weights = portfolio_weights(panel, "min_variance", max_weight=0.5)
    cov = panel.cov()
    equal = np.repeat(0.2, 5)
    assert float(weights @ cov @ weights) <= float(equal @ cov.to_numpy() @ equal) + 1e-8


def test_risk_parity_contributions_are_balanced_without_binding_cap() -> None:
    rng = np.random.default_rng(12)
    panel = pd.DataFrame(rng.normal(size=(2500, 4)) * [0.01, 0.015, 0.02, 0.03], columns=list("ABCD"))
    weights = portfolio_weights(panel, "risk_parity", max_weight=0.6)
    rc = portfolio_analytics(weights, panel.cov())["risk_contributions"]
    assert (rc.max() - rc.min()) / rc.mean() < 0.01


def test_sixty_forty_caps_and_redistributes_excess() -> None:
    panel = pd.DataFrame(np.tile([0.01, 0.015, 0.02], (30, 1)), columns=["SPY", "IEF", "GLD"])
    weights = portfolio_weights(panel, "sixty_forty", max_weight=0.4)
    assert weights.sum() == pytest.approx(1.0, abs=1e-8)
    assert weights.max() <= 0.4
    assert weights["SPY"] == pytest.approx(0.4)
    assert weights["IEF"] == pytest.approx(0.4)


def test_hrp_block_correlation_returns_capped_simplex() -> None:
    rng = np.random.default_rng(43)
    common_a = rng.normal(size=500)
    common_b = rng.normal(size=500)
    panel = pd.DataFrame({
        "A": common_a + rng.normal(0, 0.1, 500),
        "B": common_a + rng.normal(0, 0.1, 500),
        "C": common_b + rng.normal(0, 0.1, 500),
        "D": common_b + rng.normal(0, 0.1, 500),
    })
    weights = portfolio_weights(panel, "hrp", max_weight=0.4)
    assert weights.sum() == pytest.approx(1.0, abs=1e-8)
    assert weights.max() <= 0.4 + 1e-8


def _backtest_config() -> dict:
    return {"estimation_window": 5, "rebalance_freq": "monthly", "execution_lag_days": 1,
            "cost_bps": 5, "slippage_bps": 2, "max_weight": 0.5, "initial_capital": 1.0}


def test_turnover_and_cost_accounting() -> None:
    turnover, cost = turnover_cost(np.array([0.5, 0.5]), np.array([0.5, 0.5]), 5, 2)
    assert turnover == 0.0
    assert cost == 0.0
    turnover, cost = turnover_cost(np.array([1.0, 0.0]), np.array([0.0, 1.0]), 5, 2)
    assert turnover == 2.0
    assert cost == pytest.approx(2.0 * 7 / 1e4)


def test_walk_forward_targets_do_not_use_activation_day_return() -> None:
    index = pd.bdate_range("2024-01-02", periods=45)
    base = pd.DataFrame({"A": 0.001, "B": 0.0005}, index=index)
    shocked = base.copy()
    jan_month_end = index[index.to_period("M") == pd.Period("2024-01", freq="M")][-1]
    activation = index[index.get_loc(jan_month_end) + 1]
    shocked.loc[activation, "A"] = -0.8
    cfg = _backtest_config()
    first = run_backtest(base, cfg, strategies=("inverse_vol",))["weights"]
    second = run_backtest(shocked, cfg, strategies=("inverse_vol",))["weights"]
    target_date = first.loc[first["weight_type"] == "target", "date"].min()
    assert target_date == activation
    lhs = first[(first.date == activation) & (first.weight_type == "target")].set_index("asset").weight
    rhs = second[(second.date == activation) & (second.weight_type == "target")].set_index("asset").weight
    pd.testing.assert_series_equal(lhs, rhs)


def test_weights_drift_after_trade_and_nav_compounds_net_returns() -> None:
    index = pd.bdate_range("2024-01-02", periods=30)
    returns = pd.DataFrame(0.0, index=index, columns=["A", "B"])
    jan_end_pos = int(np.flatnonzero(index.to_period("M") == pd.Period("2024-01", freq="M"))[-1])
    activation_pos = jan_end_pos + 1
    returns.iloc[activation_pos + 1, 0] = 0.10
    result = run_backtest(returns, _backtest_config(), strategies=("equal_weight",))
    rows = result["weights"]
    after_move = rows[(rows.date == index[activation_pos + 1]) & (rows.weight_type == "drifted")].set_index("asset").weight
    assert after_move["A"] == pytest.approx(0.5238095238)
    net = result["returns_net"]["equal_weight"]
    nav = result["nav"]["equal_weight"]
    np.testing.assert_allclose(nav.to_numpy(), np.cumprod(1.0 + net.to_numpy()))


def test_metrics_on_known_daily_series() -> None:
    daily = pd.Series([0.01, -0.02, 0.01, 0.0], index=pd.bdate_range("2020-01-01", periods=4))
    result = performance_metrics(daily, annualization=4)
    assert result["max_drawdown"] == pytest.approx(-0.02)
    assert result["volatility"] == pytest.approx(daily.std(ddof=1) * 2)


def test_portfolio_analytics_effective_count_and_classes() -> None:
    weights = pd.Series({"SPY": 0.5, "IEF": 0.5})
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.04]], index=weights.index, columns=weights.index)
    analytics = portfolio_analytics(weights, cov, {"SPY": "equity", "IEF": "rates"})
    assert analytics["effective_assets"] == pytest.approx(2.0)
    assert analytics["hhi"] == pytest.approx(0.5)
    assert analytics["diversification_ratio"] == pytest.approx(np.sqrt(2.0))
    assert set(analytics["asset_class_weights"].index) == {"equity", "rates"}
