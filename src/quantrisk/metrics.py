"""Portfolio performance and drawdown metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def performance_metrics(returns: pd.Series, risk_free: pd.Series | None = None, annualization: int = 252) -> dict[str, float]:
    """Compute core geometric return, risk, and tail-risk measures."""
    values = returns.dropna().astype(float)
    if values.empty:
        return {"cagr": np.nan, "volatility": np.nan, "sharpe": np.nan, "max_drawdown": np.nan}
    rf = risk_free.reindex(values.index).fillna(0.0) if risk_free is not None else pd.Series(0.0, index=values.index)
    nav = (1.0 + values).cumprod()
    years = len(values) / annualization
    cagr = float(nav.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan
    volatility = float(values.std(ddof=1) * np.sqrt(annualization))
    excess = values - rf
    sharpe = float(excess.mean() / values.std(ddof=1) * np.sqrt(annualization)) if values.std(ddof=1) else np.nan
    downside = np.minimum(excess.to_numpy(), 0.0)
    downside_dev = float(np.sqrt(np.mean(downside**2)) * np.sqrt(annualization))
    peak = nav.cummax()
    drawdown = nav / peak - 1.0
    monthly = (1.0 + values).resample("ME").prod() - 1.0
    losses = -values
    return {
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": sharpe,
        "sortino": float(excess.mean() * annualization / downside_dev) if downside_dev else np.nan,
        "max_drawdown": float(drawdown.min()),
        "average_drawdown": float(drawdown.mean()),
        "calmar": float(cagr / abs(drawdown.min())) if drawdown.min() else np.nan,
        "ulcer_index": float(np.sqrt(np.mean(np.square(drawdown.to_numpy())))),
        "var_95": float(losses.quantile(0.95)),
        "cvar_95": float(losses[losses >= losses.quantile(0.95)].mean()),
        "var_99": float(losses.quantile(0.99)),
        "cvar_99": float(losses[losses >= losses.quantile(0.99)].mean()),
        "skewness": float(values.skew()),
        "excess_kurtosis": float(values.kurtosis()),
        "best_day": float(values.max()),
        "worst_day": float(values.min()),
        "best_month": float(monthly.max()),
        "worst_month": float(monthly.min()),
        "positive_months_pct": float((monthly > 0).mean()),
    }


def max_drawdown_duration(returns: pd.Series) -> int:
    """Return the longest number of daily observations below prior NAV peaks."""
    nav = (1.0 + returns.dropna()).cumprod()
    underwater = nav < nav.cummax()
    groups = (~underwater).cumsum()
    return int(underwater.groupby(groups).sum().max()) if underwater.any() else 0
