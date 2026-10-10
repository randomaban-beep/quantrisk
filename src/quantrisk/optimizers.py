"""Long-only portfolio construction methods."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.optimize import least_squares, minimize
from scipy.spatial.distance import squareform

from quantrisk.estimators import covariance, expected_returns

LOGGER = logging.getLogger(__name__)


def _cap_normalize(weights: np.ndarray, cap: float) -> np.ndarray:
    """Project nonnegative weights onto the capped simplex."""
    if cap * len(weights) < 1.0:
        raise ValueError("Weight cap is infeasible for the asset count")
    result = np.maximum(weights, 0.0)
    result = result / result.sum()
    for _ in range(len(result) * 2):
        over = result > cap
        if not over.any():
            return result
        excess = (result[over] - cap).sum()
        result[over] = cap
        available = result < cap - 1e-12
        capacity = cap - result[available]
        if not available.any():
            raise ValueError("No capacity remains for weight redistribution")
        result[available] += excess * capacity / capacity.sum()
    return result / result.sum()


def _hrp(window: pd.DataFrame, cap: float, method: str = "single") -> np.ndarray:
    """Hierarchical risk parity using recursive inverse-variance bisection."""
    corr = window.corr().fillna(0.0).clip(-1.0, 1.0)
    dist = np.sqrt(np.maximum(0.0, 0.5 * (1.0 - corr.to_numpy())))
    tree = linkage(squareform(dist, checks=False), method=method)
    order = list(range(len(window.columns)))
    clusters = [order]
    # Recover quasi-diagonal order from the linkage tree.
    def leaves(node: int) -> list[int]:
        if node < len(order):
            return [node]
        row = tree[int(node - len(order))]
        return leaves(int(row[0])) + leaves(int(row[1]))

    order = leaves(2 * len(window.columns) - 2)
    weights = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        next_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            split = len(cluster) // 2
            left, right = cluster[:split], cluster[split:]
            cov = window.cov().to_numpy()
            left_var = float(np.mean(cov[np.ix_(left, left)]))
            right_var = float(np.mean(cov[np.ix_(right, right)]))
            alpha = 1.0 - left_var / (left_var + right_var)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
            next_clusters.extend([left, right])
        clusters = next_clusters
    return _cap_normalize(weights.sort_index().to_numpy(), cap)


def portfolio_weights(
    window: pd.DataFrame,
    strategy: str,
    max_weight: float = 0.40,
    covariance_method: str = "ledoit_wolf",
    mu_shrinkage: float = 0.5,
    risk_free: float = 0.0,
) -> pd.Series:
    """Compute target weights for one strategy using a historical window."""
    assets = list(window.columns)
    n_assets = len(assets)
    cap = max(max_weight, 1.0 / n_assets)
    vol_series = window.std().where(window.std() > 1e-12)
    fallback_vol = vol_series.mean()
    vol = vol_series.fillna(fallback_vol if pd.notna(fallback_vol) else 1.0).to_numpy()
    inv_vol = _cap_normalize(1.0 / vol, cap)
    if strategy == "equal_weight":
        weights = _cap_normalize(np.ones(n_assets), cap)
    elif strategy == "sixty_forty":
        raw = np.zeros(n_assets)
        if "SPY" in assets and "IEF" in assets:
            raw[assets.index("SPY")], raw[assets.index("IEF")] = 0.6, 0.4
        else:
            raw = np.ones(n_assets)
        weights = _cap_normalize(raw, cap)
    elif strategy == "inverse_vol":
        weights = inv_vol
    elif strategy == "hrp":
        weights = _hrp(window, cap)
    else:
        cov_frame = (
            window.cov()
            if strategy == "risk_parity"
            else covariance(window, covariance_method)
        )
        cov = cov_frame.to_numpy()
        mu = expected_returns(window, mu_shrinkage).to_numpy()
        if strategy == "min_variance":
            objective = lambda w: float(w @ cov @ w)
        elif strategy == "max_sharpe":
            objective = lambda w: -float((w @ mu - risk_free) / np.sqrt(max(w @ cov @ w, 1e-15)))
        elif strategy == "risk_parity":
            # Solve relative variance-contribution equalities in log-weight
            # coordinates. The softmax enforces positive, fully invested weights
            # and avoids the poor finite-difference scaling of SLSQP near the
            # small absolute variance values typical of daily returns.
            def weights_from_log_ratios(log_ratios: np.ndarray) -> np.ndarray:
                logits = np.append(log_ratios, 0.0)
                logits -= logits.max()
                exp_logits = np.exp(logits)
                return exp_logits / exp_logits.sum()

            def risk_parity_residual(log_ratios: np.ndarray) -> np.ndarray:
                w = weights_from_log_ratios(log_ratios)
                variance_contrib = np.maximum(w * (cov @ w), 1e-30)
                log_rc = np.log(variance_contrib)
                return log_rc[:-1] - log_rc[-1]

            solution = least_squares(
                risk_parity_residual,
                np.log(np.maximum(inv_vol[:-1], 1e-15) / max(inv_vol[-1], 1e-15)),
                xtol=1e-14,
                ftol=1e-14,
                gtol=1e-14,
                max_nfev=2000,
            )
            parity_weights = weights_from_log_ratios(solution.x)
            if solution.success and parity_weights.max() <= cap + 1e-10:
                return pd.Series(parity_weights, index=assets, name=strategy)

            # If an unconstrained solution violates the cap, solve on the capped
            # simplex using SLSQP as the constrained fallback.
            def objective(w: np.ndarray) -> float:
                variance_contrib = np.maximum(w * (cov @ w), 1e-30)
                centered = np.log(variance_contrib) - np.log(variance_contrib).mean()
                return float(centered @ centered)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        result = minimize(
            objective, inv_vol, method="SLSQP", bounds=[(0.0, cap)] * n_assets,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        weights = _cap_normalize(result.x, cap) if result.success else inv_vol
        if not result.success:
            LOGGER.warning("%s solver failed; falling back to inverse volatility", strategy)
    return pd.Series(weights, index=assets, name=strategy)


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Return percentage contributions to portfolio volatility."""
    w = weights.reindex(cov.index).to_numpy()
    variance = float(w @ cov.to_numpy() @ w)
    return pd.Series(w * (cov.to_numpy() @ w) / np.sqrt(variance), index=cov.index)


def portfolio_analytics(
    weights: pd.Series,
    cov: pd.DataFrame,
    asset_classes: dict[str, str] | None = None,
) -> dict[str, float | pd.Series]:
    """Summarize portfolio risk, diversification, concentration, and classes."""
    aligned = weights.reindex(cov.index).fillna(0.0)
    sigma = cov.reindex(index=aligned.index, columns=aligned.index).to_numpy()
    w = aligned.to_numpy()
    asset_vol = np.sqrt(np.maximum(np.diag(sigma), 0.0))
    port_vol = float(np.sqrt(max(w @ sigma @ w, 0.0)))
    contributions = risk_contributions(aligned, cov)
    normalized_rc = contributions.abs() / max(float(contributions.abs().sum()), 1e-15)
    effective_bets = float(np.exp(-(normalized_rc * np.log(normalized_rc.clip(lower=1e-15))).sum()))
    classes = asset_classes or {asset: asset for asset in aligned.index}
    class_weights = aligned.groupby([classes.get(asset, "other") for asset in aligned.index]).sum()
    class_risk = contributions.groupby([classes.get(asset, "other") for asset in contributions.index]).sum()
    return {
        "risk_contributions": contributions,
        "diversification_ratio": float(w @ asset_vol / max(port_vol, 1e-15)),
        "effective_assets": float(1.0 / np.square(w).sum()),
        "effective_bets": effective_bets,
        "hhi": float(np.square(w).sum()),
        "asset_class_weights": class_weights,
        "asset_class_risk": class_risk,
    }
