"""
Drift Research Dashboard — Showcase

Run:  streamlit run src/dashboard/app.py

Demo build — responses are precomputed fixtures generated on synthetic data.
The production inference engine is proprietary and not included.

Four pages:
  Signal Explorer   — regime timeline, factor IC heatmap, screener results
  Portfolio Weights — HRP weights, effective-N
  Risk Attribution  — factor vs specific variance breakdown
  Backtest          — equity curve, drawdown, performance metrics
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Drift — Showcase",
    page_icon="D",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.dashboard.demo import load_demo_data

# ------------------------------------------------------------------ #
# Demo banner (always shown)                                           #
# ------------------------------------------------------------------ #

st.info(
    "**Demo build** — All responses are precomputed fixtures generated on "
    "synthetic data. The production inference engine (regime models, signal "
    "construction, portfolio optimisation) is proprietary and not included "
    "in this repository.",
    icon="i",
)

# ------------------------------------------------------------------ #
# Sidebar                                                              #
# ------------------------------------------------------------------ #

with st.sidebar:
    st.markdown("## Drift")
    st.markdown("*Regime-aware factor research platform*")
    st.divider()
    page = st.radio(
        "Page",
        ["Signal Explorer", "Portfolio Weights", "Risk Attribution", "Backtest"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Showcase build — synthetic data")

# ------------------------------------------------------------------ #
# Load data                                                            #
# ------------------------------------------------------------------ #

@st.cache_data
def _load():
    return load_demo_data()

data = _load()

# ------------------------------------------------------------------ #
# Pages                                                                #
# ------------------------------------------------------------------ #

if page == "Signal Explorer":
    st.header("Signal Explorer")

    col1, col2, col3 = st.columns(3)
    col1.metric("Current Regime", data["current_regime"].upper())
    col2.metric("Regime Confidence", f"{data['regime_conf']:.1%}")
    col3.metric("As of", str(data["as_of"]))

    st.subheader("Factor IC Heatmap")
    ic_df = data["factor_ic"]
    fig = px.imshow(
        ic_df.T,
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        labels={"color": "IC"},
        title="Rolling Factor IC (synthetic)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Screener Results — Nifty 50")
    results = data["screener"]["results"]
    df = pd.DataFrame([{
        "Rank":       r["rank"],
        "Ticker":     r["ticker"],
        "Sector":     r["sector"],
        "Composite":  r["composite_score"],
        "Momentum":   r["factor_scores"].get("momentum", 0.0),
        "Quality":    r["factor_scores"].get("quality", 0.0),
        "Value":      r["factor_scores"].get("value", 0.0),
        "Ann. Vol":   r["annualised_vol"],
        "Class":      r["classification"],
    } for r in results])

    color_map = {"candidate": "#1a7f4b", "watchlist": "#8a7a00", "avoid": "#c0392b"}
    st.dataframe(
        df.style.applymap(
            lambda v: f"color: {color_map.get(v, 'inherit')}",
            subset=["Class"],
        ).format({"Composite": "{:+.3f}", "Momentum": "{:+.3f}",
                  "Quality": "{:+.3f}", "Value": "{:+.3f}", "Ann. Vol": "{:.1%}"}),
        use_container_width=True,
        hide_index=True,
    )

elif page == "Portfolio Weights":
    st.header("Portfolio Weights (HRP)")

    weights = data["weights_hrp"]
    df = pd.DataFrame(
        sorted(weights.items(), key=lambda x: x[1], reverse=True),
        columns=["Ticker", "Weight"],
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(df, x="Ticker", y="Weight", title="HRP Weights",
                     color="Weight", color_continuous_scale="Blues")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.pie(df, names="Ticker", values="Weight", title="Weight Distribution")
        st.plotly_chart(fig2, use_container_width=True)

    eff_n = 1 / sum(w**2 for w in weights.values())
    st.metric("Effective N (1/HHI)", f"{eff_n:.2f}")
    st.dataframe(df.style.format({"Weight": "{:.1%}"}), use_container_width=True, hide_index=True)

elif page == "Risk Attribution":
    st.header("Risk Attribution")

    risk = data["risk"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Ann. Volatility", f"{risk['annualised_vol']:.1%}")
    col2.metric("Factor Variance", f"{risk['factor_variance']:.4f}")
    col3.metric("Specific Variance", f"{risk['specific_variance']:.4f}")

    st.subheader("Factor Contribution to Variance")
    fc = risk["factor_contrib"]
    df_fc = pd.DataFrame(
        sorted(fc.items(), key=lambda x: x[1], reverse=True),
        columns=["Factor", "Variance Contribution"],
    )
    fig = px.bar(df_fc, x="Factor", y="Variance Contribution",
                 title="Factor Variance Contributions", color="Variance Contribution",
                 color_continuous_scale="Oranges")
    st.plotly_chart(fig, use_container_width=True)

    total = risk["factor_variance"] + risk["specific_variance"]
    pie_df = pd.DataFrame({
        "Type": ["Factor", "Specific"],
        "Variance": [risk["factor_variance"], risk["specific_variance"]],
    })
    fig2 = px.pie(pie_df, names="Type", values="Variance",
                  title="Factor vs Specific Variance Split",
                  color_discrete_map={"Factor": "#2980b9", "Specific": "#e67e22"})
    st.plotly_chart(fig2, use_container_width=True)

elif page == "Backtest":
    st.header("Backtest — HRP Strategy")

    m = data["backtest"]["metrics"]
    cols = st.columns(5)
    cols[0].metric("Total Return",   f"{m['total_return']:.1%}")
    cols[1].metric("Ann. Return",    f"{m['ann_return']:.1%}")
    cols[2].metric("Sharpe",         f"{m['sharpe']:.2f}")
    cols[3].metric("Max Drawdown",   f"{m['max_drawdown']:.1%}")
    cols[4].metric("Sortino",        f"{m['sortino']:.2f}")

    ec  = data["equity_curve"]
    dd  = data["drawdown"]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3],
                        subplot_titles=["Equity Curve", "Drawdown"])

    fig.add_trace(go.Scatter(x=ec.index, y=ec.values, name="HRP",
                              line={"color": "#2980b9"}), row=1, col=1)
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name="Drawdown",
                              fill="tozeroy", line={"color": "#e74c3c"},
                              fillcolor="rgba(231,76,60,0.2)"), row=2, col=1)
    fig.update_layout(height=550, showlegend=False)
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Performance Metrics")
    metrics_df = pd.DataFrame([{
        "Metric": k.replace("_", " ").title(), "Value": v
    } for k, v in m.items()])
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
