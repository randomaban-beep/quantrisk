"""Reproducible report figures from saved project outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf"]


def make_figures(results: str | Path = "results", output: str | Path = "figures", returns: pd.DataFrame | None = None) -> list[Path]:
    """Generate the twelve static figures required by the report."""
    root, out = Path(results), Path(output)
    out.mkdir(parents=True, exist_ok=True)
    nav = pd.read_parquet(root / "nav.parquet")
    net = pd.read_parquet(root / "returns_net.parquet")
    weights = pd.read_parquet(root / "weights.parquet")
    var = pd.read_parquet(root / "var_forecasts.parquet")
    backtest = pd.read_csv(root / "var_backtest_summary.csv")
    files: list[Path] = []

    def save(fig: plt.Figure, filename: str) -> None:
        fig.tight_layout()
        target = out / filename
        fig.savefig(target, dpi=150, bbox_inches="tight")
        plt.close(fig)
        files.append(target)

    fig, ax = plt.subplots(figsize=(10, 5))
    (nav / nav.iloc[0].replace(0, 1)).plot(ax=ax, color=COLORS[:len(nav.columns)])
    ax.set_yscale("log"); ax.set_title("Cumulative net NAV"); ax.set_ylabel("NAV (log scale)")
    save(fig, "01_cumulative_nav.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    drawdown = nav / nav.cummax() - 1.0
    drawdown.plot(ax=ax, color=COLORS[:len(nav.columns)]); ax.set_title("Underwater drawdown"); ax.set_ylabel("Drawdown")
    save(fig, "02_underwater.png")

    rolling_mean = net.rolling(252).mean()
    rolling_std = net.rolling(252).std()
    rolling_sharpe = rolling_mean.div(rolling_std.replace(0, np.nan)) * np.sqrt(252)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    rolling_sharpe.plot(ax=axes[0], color=COLORS[:len(net.columns)]); axes[0].set_title("Rolling one-year Sharpe")
    (rolling_std * np.sqrt(252)).plot(ax=axes[1], color=COLORS[:len(net.columns)]); axes[1].set_title("Rolling one-year volatility")
    save(fig, "03_rolling_risk.png")

    fig, axes = plt.subplots(len(nav.columns), 1, figsize=(11, max(5, 1.7 * len(nav.columns))), sharex=True)
    for ax, strategy in zip(np.atleast_1d(axes), nav.columns, strict=True):
        subset = weights[(weights.strategy == strategy) & (weights.weight_type == "drifted")]
        pivot = subset.pivot(index="date", columns="asset", values="weight")
        pivot.plot.area(ax=ax, legend=False, linewidth=0)
        ax.set_ylabel(strategy)
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=2)
    fig.suptitle("Drifted allocation over time")
    save(fig, "04_weights.png")

    components_path = root / "component_risk_latest.csv"
    components = pd.read_csv(components_path)
    fig, ax = plt.subplots(figsize=(11, 5))
    components.pivot(index="asset", columns="strategy", values="risk_contribution_pct").plot.bar(ax=ax)
    ax.set_title("Latest component risk by asset"); ax.set_ylabel("Share of VaR")
    save(fig, "05_component_risk.png")

    if returns is None:
        returns = pd.read_parquet("data/processed/returns.parquet")
    latest = returns.tail(252)
    corr = latest.corr()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    image = axes[0].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    axes[0].set_xticks(range(len(corr)), corr.columns, rotation=90); axes[0].set_yticks(range(len(corr)), corr.index)
    axes[0].set_title("Latest correlation heatmap"); fig.colorbar(image, ax=axes[0])
    dist = np.sqrt(np.maximum(0.0, 0.5 * (1.0 - corr.to_numpy())))
    dendrogram(linkage(squareform(dist, checks=False), method="single"), labels=list(corr.columns), ax=axes[1])
    axes[1].set_title("HRP single-linkage dendrogram")
    save(fig, "06_hrp_clusters.png")

    cov = latest.cov().to_numpy(); mu = latest.mean().to_numpy()
    rng = np.random.default_rng(42)
    frontier = []
    for vector in rng.dirichlet(np.ones(len(mu)), size=300):
        frontier.append((np.sqrt(vector @ cov @ vector) * np.sqrt(252), vector @ mu * 252))
    fig, ax = plt.subplots(figsize=(8, 5))
    x, y = np.array(frontier).T
    ax.scatter(x, y, s=10, alpha=0.4, label="Long-only portfolios")
    latest_weights = pd.read_csv(root / "portfolio_weights_latest.csv", index_col="asset")
    for strategy in latest_weights:
        vector = latest_weights[strategy].reindex(latest.columns).fillna(0).to_numpy()
        ax.scatter(np.sqrt(vector @ cov @ vector) * np.sqrt(252), vector @ mu * 252, label=strategy, s=35)
    ax.set_title("Long-only efficient frontier"); ax.set_xlabel("Annual volatility"); ax.set_ylabel("Annual arithmetic return")
    ax.legend(fontsize=7, ncol=2)
    save(fig, "07_efficient_frontier.png")

    selected = var[(var.strategy == "risk_parity") & (var.confidence == 0.99) & var.model.isin(["historical", "fhs"])]
    fig, ax = plt.subplots(figsize=(10, 5))
    if not selected.empty:
        group = selected[selected.model == "fhs"]
        ax.plot(group.date, -group.realized, color="black", linewidth=0.7, label="Realized loss")
        ax.plot(group.date, group["var"], label="FHS VaR")
        ax.scatter(group.loc[group.realized < -group["var"], "date"], -group.loc[group.realized < -group["var"], "realized"], color="red", s=12, label="Exception")
    ax.set_title("Risk parity: 99% VaR and realized loss"); ax.legend()
    save(fig, "08_var_exceptions.png")

    pvalues = backtest[backtest.confidence == 0.99].pivot(index="strategy", columns="model", values="kupiec_p")
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(pvalues.fillna(0), cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(pvalues.columns)), pvalues.columns, rotation=35); ax.set_yticks(range(len(pvalues.index)), pvalues.index)
    ax.set_title("Kupiec p-values by model and strategy"); fig.colorbar(image, ax=ax)
    save(fig, "09_kupiec_heatmap.png")

    traffic_path = root / "basel_traffic_light.csv"
    traffic = pd.read_csv(traffic_path, parse_dates=["date"])
    fig, ax = plt.subplots(figsize=(10, 4))
    for strategy, subset in traffic.groupby("strategy"):
        values = subset.zone.map({"green": 0, "yellow": 1, "red": 2})
        ax.step(subset.date, values, where="post", label=strategy)
    ax.set_yticks([0, 1, 2], ["Green", "Yellow", "Red"]); ax.set_title("Basel 99% traffic light zones"); ax.legend(ncol=4, fontsize=7)
    save(fig, "10_basel_traffic_light.png")

    historical = pd.read_csv(root / "stress_historical.csv")
    hypothetical = pd.read_csv(root / "stress_hypothetical.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    historical.pivot(index="scenario", columns="strategy", values="static_weights_return").plot.bar(ax=axes[0])
    hypothetical.pivot(index="scenario", columns="strategy", values="portfolio_pnl").plot.bar(ax=axes[1])
    axes[0].set_title("Historical stress replay"); axes[1].set_title("Illustrative hypothetical shocks")
    save(fig, "11_stress_tests.png")

    sensitivity = pd.read_csv(root / "sensitivity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for strategy, subset in sensitivity[sensitivity.factor == "estimation_window"].groupby("strategy"):
        axes[0].plot(subset.value, subset.sharpe, marker="o", label=strategy)
    for strategy, subset in sensitivity[sensitivity.factor == "cost_total_bps"].groupby("strategy"):
        axes[1].plot(subset.value, subset.sharpe, marker="o", label=strategy)
    axes[0].set_title("Sharpe vs estimation window"); axes[1].set_title("Sharpe vs one-way cost level")
    axes[0].legend(fontsize=7); axes[1].legend(fontsize=7)
    save(fig, "12_sensitivity.png")
    return files
