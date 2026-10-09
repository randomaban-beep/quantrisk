"""Expected return and covariance estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def expected_returns(window: pd.DataFrame, shrinkage: float = 0.5, annualization: int = 252) -> pd.Series:
    """Historical means shrunk toward the cross-sectional mean."""
    means = window.mean()
    return (1.0 - shrinkage) * means + shrinkage * means.mean()


def covariance(window: pd.DataFrame, method: str = "ledoit_wolf", ewma_lambda: float = 0.94) -> pd.DataFrame:
    """Estimate covariance with sample, Ledoit-Wolf, or EWMA methods."""
    if method == "sample":
        matrix = window.cov().to_numpy()
    elif method == "ledoit_wolf":
        matrix = LedoitWolf().fit(window.to_numpy()).covariance_
    elif method == "ewma":
        values = window.to_numpy()
        weights = (1.0 - ewma_lambda) * ewma_lambda ** np.arange(len(values) - 1, -1, -1)
        weights /= weights.sum()
        centered = values - np.average(values, axis=0, weights=weights)
        matrix = (centered * weights[:, None]).T @ centered
    else:
        raise ValueError(f"Unknown covariance estimator: {method}")
    return pd.DataFrame(matrix, index=window.columns, columns=window.columns)
