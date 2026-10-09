"""Formal backtests for one-day VaR exception sequences."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2


def _xlogx(count: float, probability: float) -> float:
    """Compute x log(p), with the 0 log(0) convention."""
    return 0.0 if count == 0 else count * np.log(max(probability, 1e-300))


def kupiec_pof(exceptions: np.ndarray, confidence: float) -> tuple[float, float]:
    """Return Kupiec unconditional coverage likelihood ratio and p-value."""
    hits = np.asarray(exceptions, dtype=bool)
    n = len(hits)
    x = int(hits.sum())
    if n == 0:
        return np.nan, np.nan
    expected = 1.0 - confidence
    observed = x / n
    lr = 2.0 * (
        _xlogx(x, observed) + _xlogx(n - x, 1.0 - observed)
        - _xlogx(x, expected) - _xlogx(n - x, 1.0 - expected)
    )
    return float(max(lr, 0.0)), float(chi2.sf(max(lr, 0.0), 1))


def christoffersen_independence(exceptions: np.ndarray) -> tuple[float, float]:
    """Return first-order Markov independence likelihood ratio and p-value."""
    hits = np.asarray(exceptions, dtype=bool)
    if len(hits) < 2:
        return np.nan, np.nan
    previous, current = hits[:-1], hits[1:]
    n00 = int((~previous & ~current).sum())
    n01 = int((~previous & current).sum())
    n10 = int((previous & ~current).sum())
    n11 = int((previous & current).sum())
    p = (n01 + n11) / max(len(hits) - 1, 1)
    p0 = n01 / max(n00 + n01, 1)
    p1 = n11 / max(n10 + n11, 1)
    ll_ind = _xlogx(n01 + n11, p) + _xlogx(n00 + n10, 1.0 - p)
    ll_markov = _xlogx(n01, p0) + _xlogx(n00, 1.0 - p0) + _xlogx(n11, p1) + _xlogx(n10, 1.0 - p1)
    lr = max(0.0, -2.0 * (ll_ind - ll_markov))
    return float(lr), float(chi2.sf(lr, 1))


def summarize_var(forecasts: pd.DataFrame, significance: float = 0.05) -> pd.DataFrame:
    """Summarize coverage, clustering, conditional coverage, and ES calibration."""
    rows = []
    for (strategy, model, confidence), group in forecasts.groupby(["strategy", "model", "confidence"]):
        hits = (group["realized"].to_numpy() < -group["var"].to_numpy())
        lr_pof, p_pof = kupiec_pof(hits, float(confidence))
        lr_ind, p_ind = christoffersen_independence(hits)
        lr_cc = lr_pof + lr_ind if np.isfinite(lr_ind) else np.nan
        p_cc = float(chi2.sf(lr_cc, 2)) if np.isfinite(lr_cc) else np.nan
        losses = -group.loc[hits, "realized"]
        es_mean = group.loc[hits, "es"].mean()
        rows.append({
            "strategy": strategy, "model": model, "confidence": confidence,
            "observations": len(group), "exceptions": int(hits.sum()),
            "expected_exceptions": len(group) * (1.0 - confidence),
            "exception_rate": float(hits.mean()), "kupiec_lr": lr_pof, "kupiec_p": p_pof,
            "independence_lr": lr_ind, "independence_p": p_ind,
            "conditional_coverage_lr": lr_cc, "conditional_coverage_p": p_cc,
            "es_ratio": float(losses.mean() / es_mean) if len(losses) and es_mean else np.nan,
            "kupiec_pass": bool(p_pof >= significance) if np.isfinite(p_pof) else False,
            "independence_pass": bool(p_ind >= significance) if np.isfinite(p_ind) else False,
            "conditional_coverage_pass": bool(p_cc >= significance) if np.isfinite(p_cc) else False,
        })
    return pd.DataFrame(rows)


def traffic_light(exceptions: pd.Series, window: int = 250) -> pd.Series:
    """Classify rolling 99% exceptions into Basel green/yellow/red zones."""
    rolling = exceptions.astype(int).rolling(window, min_periods=window).sum()
    return pd.cut(rolling, [-1, 4, 9, np.inf], labels=["green", "yellow", "red"])
