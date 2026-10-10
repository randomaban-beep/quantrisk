"""Read-only Streamlit dashboard built entirely from saved result files."""

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path("results")
st.set_page_config(page_title="QuantRisk", layout="wide")
st.title("QuantRisk portfolio and risk dashboard")
tabs = st.tabs(["Performance", "Allocation", "Risk", "Stress", "Sensitivity"])

with tabs[0]:
    nav = pd.read_parquet(ROOT / "nav.parquet")
    metrics = pd.read_csv(ROOT / "metrics_summary.csv", index_col="strategy")
    selected = st.multiselect("Strategies", list(nav.columns), default=list(nav.columns))
    st.line_chart(nav[selected])
    st.dataframe(metrics.loc[selected])
with tabs[1]:
    weights = pd.read_parquet(ROOT / "weights.parquet")
    strategy = st.selectbox("Strategy", sorted(weights.strategy.unique()))
    allocation = weights[(weights.strategy == strategy) & (weights.weight_type == "drifted")]
    st.area_chart(allocation.pivot(index="date", columns="asset", values="weight"))
    st.dataframe(pd.read_csv(ROOT / "component_risk_latest.csv").query("strategy == @strategy"))
with tabs[2]:
    var = pd.read_parquet(ROOT / "var_forecasts.parquet")
    strategy = st.selectbox("Risk strategy", sorted(var.strategy.unique()))
    st.dataframe(pd.read_csv(ROOT / "var_backtest_summary.csv").query("strategy == @strategy"))
    st.line_chart(var.query("strategy == @strategy and confidence == 0.99").pivot(index="date", columns="model", values="var"))
    st.dataframe(pd.read_csv(ROOT / "basel_traffic_light.csv").query("strategy == @strategy"))
with tabs[3]:
    st.subheader("Historical stress")
    st.dataframe(pd.read_csv(ROOT / "stress_historical.csv"))
    st.subheader("Hypothetical illustrative assumptions")
    st.dataframe(pd.read_csv(ROOT / "stress_hypothetical.csv"))
with tabs[4]:
    sensitivity = pd.read_csv(ROOT / "sensitivity.csv")
    st.dataframe(sensitivity)
    st.line_chart(sensitivity[sensitivity.factor == "cost_total_bps"].pivot(index="value", columns="strategy", values="sharpe"))
