"""Deterministic, network-free tests for portfolio primitives."""

import numpy as np
import pandas as pd

from quantrisk.data import calculate_returns, clean_prices


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
