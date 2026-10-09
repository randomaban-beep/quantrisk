"""Historical, hypothetical, and correlation stress analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

HISTORICAL_SCENARIOS = {
    "Global Financial Crisis": ("2008-09-01", "2009-03-09"),
    "2011 Euro-crisis selloff": ("2011-07-22", "2011-10-03"),
    "Q4 2018 selloff": ("2018-09-20", "2018-12-24"),
    "COVID crash": ("2020-02-19", "2020-03-23"),
    "2022 rates and inflation shock": ("2022-01-03", "2022-10-14"),
}


def hypothetical_stress(weights: pd.Series, shocks: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Compute one-step scenario P&L and the three largest asset contributions."""
    rows = []
    for scenario, shock in shocks.items():
        pnl = weights * pd.Series(shock).reindex(weights.index).fillna(0.0)
        leaders = pnl.abs().nlargest(3)
        rows.append({"scenario": scenario, "classification": "illustrative assumption, not a forecast",
                     "portfolio_pnl": float(pnl.sum()),
                     "top_assets": ";".join(f"{asset}:{pnl[asset]:.4f}" for asset in leaders.index)})
    return pd.DataFrame(rows)


def historical_stress(
    weights: pd.Series,
    asset_returns: pd.DataFrame,
    strategy_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """Replay fixed latest holdings and realized strategy returns over set windows."""
    rows = []
    aligned_weights = weights.reindex(asset_returns.columns).fillna(0.0)
    portfolio = asset_returns @ aligned_weights
    for name, (start, end) in HISTORICAL_SCENARIOS.items():
        period = portfolio.loc[start:end].dropna()
        if period.empty:
            continue
        nav = (1.0 + period).cumprod()
        dd = nav / nav.cummax() - 1.0
        realized = strategy_returns.loc[start:end].dropna() if strategy_returns is not None else pd.Series(dtype=float)
        rows.append({"scenario": name, "start": start, "end": end,
                     "static_weights_return": float(nav.iloc[-1] - 1.0),
                     "static_weights_max_drawdown": float(dd.min()),
                     "realized_strategy_return": float((1.0 + realized).prod() - 1.0) if len(realized) else np.nan})
    return pd.DataFrame(rows)


def correlation_stress(
    weights: pd.Series,
    returns: pd.DataFrame,
    levels: tuple[float, ...] = (0.5, 0.8, 0.95),
) -> pd.DataFrame:
    """Raise off-diagonal correlations to each floor, preserving asset volatilities."""
    cov = returns.reindex(columns=weights.index).cov().to_numpy()
    vol = np.sqrt(np.maximum(np.diag(cov), 0.0))
    corr = cov / np.outer(np.maximum(vol, 1e-15), np.maximum(vol, 1e-15))
    w = weights.reindex(returns.columns).fillna(0.0).to_numpy()
    base_vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
    z_99 = float(norm.ppf(0.99))
    base_var = z_99 * base_vol
    rows = [{"rho_floor": "baseline", "portfolio_volatility": base_vol, "var_99": base_var,
             "increase_vs_baseline": 0.0, "var_99_increase": 0.0}]
    for level in levels:
        stressed_corr = corr.copy()
        mask = ~np.eye(len(vol), dtype=bool)
        stressed_corr[mask] = np.maximum(stressed_corr[mask], level)
        np.fill_diagonal(stressed_corr, 1.0)
        stressed_cov = stressed_corr * np.outer(vol, vol)
        stressed_vol = float(np.sqrt(max(w @ stressed_cov @ w, 0.0)))
        stressed_var = z_99 * stressed_vol
        rows.append({"rho_floor": level, "portfolio_volatility": stressed_vol,
                     "var_99": stressed_var, "increase_vs_baseline": stressed_vol - base_vol,
                     "var_99_increase": stressed_var - base_var})
    return pd.DataFrame(rows)
