"""One-day Value-at-Risk and Expected Shortfall forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def _ewma_cov(values: np.ndarray, decay: float) -> np.ndarray:
    """Estimate a zero-mean EWMA covariance matrix from most recent data."""
    cov = np.zeros((values.shape[1], values.shape[1]))
    for row in values:
        cov = decay * cov + (1.0 - decay) * np.outer(row, row)
    return cov


def forecast_window(
    history: pd.DataFrame,
    weights: pd.Series,
    confidence_levels: tuple[float, ...] = (0.95, 0.99),
    ewma_lambda: float = 0.94,
) -> dict[tuple[str, float], tuple[float, float]]:
    """Fit all four risk models to observations strictly before the forecast day."""
    aligned = history.reindex(columns=weights.index).dropna(how="any")
    w = weights.to_numpy(dtype=float)
    portfolio_history = aligned.to_numpy() @ w
    cov = _ewma_cov(aligned.to_numpy(), ewma_lambda)
    mu = float(portfolio_history.mean())
    sigma = float(np.sqrt(max(w @ cov @ w, 0.0)))
    skew = float(pd.Series(portfolio_history).skew())
    excess_kurtosis = float(pd.Series(portfolio_history).kurtosis())
    ewma_variance = 0.0
    ewma_path = np.empty(len(portfolio_history))
    for i, value in enumerate(portfolio_history):
        ewma_variance = ewma_lambda * ewma_variance + (1.0 - ewma_lambda) * value**2
        ewma_path[i] = np.sqrt(ewma_variance)
    current_vol = float(np.sqrt(ewma_lambda * ewma_variance + (1.0 - ewma_lambda) * portfolio_history[-1] ** 2))
    standardized = portfolio_history / np.maximum(ewma_path, 1e-12)
    result: dict[tuple[str, float], tuple[float, float]] = {}
    for confidence in confidence_levels:
        tail = 1.0 - confidence
        quantile = float(np.quantile(portfolio_history, tail))
        result[("historical", confidence)] = (-quantile, -float(portfolio_history[portfolio_history <= quantile].mean()))
        z = float(norm.ppf(tail))
        gaussian_var = -(mu + z * sigma)
        gaussian_es = -(mu - sigma * norm.pdf(z) / tail)
        result[("gaussian", confidence)] = (gaussian_var, gaussian_es)
        z_cf = z + (z**2 - 1.0) * skew / 6.0 + (z**3 - 3.0 * z) * excess_kurtosis / 24.0 - (2.0 * z**3 - 5.0 * z) * skew**2 / 36.0
        cf_var = -(mu + z_cf * sigma)
        cf_quantile = -cf_var
        cf_tail = portfolio_history[portfolio_history <= cf_quantile]
        result[("cornish_fisher", confidence)] = (cf_var, -float(cf_tail.mean()) if len(cf_tail) else cf_var)
        fhs_q = float(np.quantile(standardized, tail))
        fhs_tail = standardized[standardized <= fhs_q]
        result[("fhs", confidence)] = (-fhs_q * current_vol, -float(fhs_tail.mean()) * current_vol)
    return result


def rolling_forecasts(
    returns: pd.DataFrame,
    held_weights: pd.DataFrame,
    window: int = 500,
    confidence_levels: tuple[float, ...] = (0.95, 0.99),
    ewma_lambda: float = 0.94,
) -> pd.DataFrame:
    """Generate one-day forecasts using history through day t-1 and holdings at t open."""
    records = []
    for pos in range(window, len(returns)):
        dt = returns.index[pos]
        hist = returns.iloc[pos - window : pos]
        daily_weights = held_weights.loc[dt].reindex(returns.columns).fillna(0.0)
        if daily_weights.sum() <= 0.0:
            continue
        forecasts = forecast_window(hist, daily_weights, confidence_levels, ewma_lambda)
        realized = float(returns.iloc[pos].reindex(daily_weights.index).fillna(0.0) @ daily_weights)
        for (model, confidence), (var, es) in forecasts.items():
            records.append((dt, model, confidence, float(var), float(es), realized))
    return pd.DataFrame(records, columns=["date", "model", "confidence", "var", "es", "realized"])


def component_var(weights: pd.Series, returns: pd.DataFrame, confidence: float = 0.99, ewma_lambda: float = 0.94) -> pd.DataFrame:
    """Decompose Gaussian VaR into marginal and component contributions by asset."""
    aligned = returns.reindex(columns=weights.index).dropna(how="any")
    cov = _ewma_cov(aligned.to_numpy(), ewma_lambda)
    w = weights.reindex(aligned.columns).to_numpy(dtype=float)
    sigma = float(np.sqrt(max(w @ cov @ w, 0.0)))
    z = float(norm.ppf(confidence))
    marginal = z * (cov @ w) / max(sigma, 1e-15)
    component = w * marginal
    result = pd.DataFrame({"asset": aligned.columns, "weight": w, "marginal_var": marginal, "component_var": component})
    result["risk_contribution_pct"] = result["component_var"] / max(result["component_var"].sum(), 1e-15)
    return result
