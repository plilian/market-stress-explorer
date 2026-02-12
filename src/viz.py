from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def plot_price_and_stress(df: pd.DataFrame, price_col: str = "Close", stress_col: str = "stress_score") -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index, y=df[price_col],
        mode="lines", name="Price"
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df[stress_col],
        mode="lines", name="Stress Score",
        yaxis="y2"
    ))

    fig.update_layout(
        title="Price vs Stress Score",
        xaxis_title="Date",
        yaxis=dict(title="Price"),
        yaxis2=dict(title="Stress", overlaying="y", side="right"),
        legend=dict(orientation="h")
    )
    return fig


def plot_feature_breakdown(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col, name in [("vol_z", "Volatility Z"), ("volshock_z", "Volume Shock Z"), ("liq_z", "Liquidity Proxy Z")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=name))

    fig.update_layout(
        title="Stress Components (Z-Scores)",
        xaxis_title="Date",
        yaxis_title="Z"
    )
    return fig
