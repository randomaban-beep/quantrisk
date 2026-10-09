"""Stationary bootstrap uncertainty and paired Sharpe comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd


def stationary_indices(length: int, resamples: int = 5000, block_length: int = 21, seed: int = 42) -> np.ndarray:
    """Generate Politis-Romano stationary-bootstrap index paths."""
    rng = np.random.default_rng(seed)
    indices = np.empty((resamples, length), dtype=np.int32)
    indices[:, 0] = rng.integers(0, length, size=resamples)
    restart_probability = 1.0 / block_length
    for col in range(1, length):
        restart = rng.random(resamples) < restart_probability
        indices[:, col] = (indices[:, col - 1] + 1) % length
        indices[restart, col] = rng.integers(0, length, size=int(restart.sum()))
    return indices


def _statistics(samples: np.ndarray, annualization: int) -> np.ndarray:
    """Compute annualized Sharpe, CAGR, and maximum drawdown by row."""
    mean = samples.mean(axis=1)
    std = samples.std(axis=1, ddof=1)
    sharpe = np.divide(mean, std, out=np.full_like(mean, np.nan), where=std > 0) * np.sqrt(annualization)
    nav = np.cumprod(1.0 + samples, axis=1)
    years = samples.shape[1] / annualization
    cagr = np.power(nav[:, -1], 1.0 / years) - 1.0
    drawdown = nav / np.maximum.accumulate(nav, axis=1) - 1.0
    return np.column_stack([sharpe, cagr, drawdown.min(axis=1)])


def stationary_bootstrap(
    returns: pd.DataFrame,
    resamples: int = 5000,
    block_length: int = 21,
    seed: int = 42,
    annualization: int = 252,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return percentile confidence intervals and paired Sharpe tests."""
    clean = returns.dropna(how="any")
    index_paths = stationary_indices(len(clean), resamples, block_length, seed)
    summaries = []
    samples_by_strategy = {}
    for strategy in clean.columns:
        samples = clean[strategy].to_numpy()[index_paths]
        stats = _statistics(samples, annualization)
        samples_by_strategy[strategy] = stats[:, 0]
        summaries.append({"strategy": strategy,
                          "sharpe_ci_low": float(np.nanquantile(stats[:, 0], 0.025)),
                          "sharpe_ci_high": float(np.nanquantile(stats[:, 0], 0.975)),
                          "cagr_ci_low": float(np.nanquantile(stats[:, 1], 0.025)),
                          "cagr_ci_high": float(np.nanquantile(stats[:, 1], 0.975)),
                          "max_drawdown_ci_low": float(np.nanquantile(stats[:, 2], 0.025)),
                          "max_drawdown_ci_high": float(np.nanquantile(stats[:, 2], 0.975))})
    paired = []
    for benchmark in ("equal_weight", "sixty_forty"):
        if benchmark not in samples_by_strategy:
            continue
        for strategy, sampled_sharpes in samples_by_strategy.items():
            if strategy == benchmark:
                continue
            diff = sampled_sharpes - samples_by_strategy[benchmark]
            p_value = min(1.0, 2.0 * min(float((diff <= 0).mean()), float((diff >= 0).mean())))
            paired.append({"strategy": strategy, "benchmark": benchmark,
                           "sharpe_difference_p": p_value,
                           "statistically_significant_5pct": p_value < 0.05,
                           "conclusion": "statistically significant" if p_value < 0.05 else "not statistically significant"})
    return pd.DataFrame(summaries), pd.DataFrame(paired)
