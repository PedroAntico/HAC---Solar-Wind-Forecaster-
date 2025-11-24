#!/usr/bin/env python3
"""
hac_v6_dashboard.py
Real-Time Dashboard for HAC v6 Solar Wind Forecaster

Features:
- Fetches real-time predictions from HACv6RealTime API
- Visualizes multi-horizon forecasts (speed, Bz, density)
- Includes confidence intervals (80%)
- Automatic refresh every 60 seconds
- Real-time alert panel
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

import dash
from dash import dcc, html
from dash.dependencies import Input, Output

import plotly.graph_objects as go


# ======================================================
# Configuration
# ======================================================

API_URL = "http://localhost:5000/api/v1/forecast"

app = dash.Dash(__name__)
app.title = "HAC v6 - Solar Wind Dashboard"


# ======================================================
# Layout
# ======================================================

app.layout = html.Div([
    html.H1("🌪️ HAC v6 – Real-Time Solar Wind Forecast",
            style={"textAlign": "center", "color": "#1E90FF"}),

    html.Div("AI-Powered Multi-Horizon Space Weather Forecasting",
             style={"textAlign": "center", "color": "#666"}),

    html.Br(),

    # Dropdowns
    html.Div([
        html.Div([
            html.Label("Model Type"),
            dcc.Dropdown(
                id="model_type",
                options=[
                    {"label": "Ensemble (recommended)", "value": "ensemble"},
                    {"label": "Hybrid", "value": "hybrid"},
                    {"label": "LSTM", "value": "lstm"},
                    {"label": "GRU", "value": "gru"},
                ],
                value="ensemble"
            )
        ], style={"width": "30%", "display": "inline-block"}),

        html.Div([
            html.Label("Horizon (hours ahead)"),
            dcc.Dropdown(
                id="horizon",
                options=[{"label": f"{h}h", "value": h} for h in [1, 3, 6, 12, 24, 48]],
                value=24
            )
        ], style={"width": "30%", "display": "inline-block", "marginLeft": "5%"})
    ], style={"padding": "20px"}),

    html.Br(),

    # Graphs
    dcc.Graph(id="speed_plot"),
    dcc.Graph(id="bz_plot"),
    dcc.Graph(id="density_plot"),

    # Alerts
    html.H2("🚨 Real-Time Alerts"),
    html.Div(id="alert_panel", style={"padding": "10px", "fontSize": "18px"}),

    # Auto refresh every 60 seconds
    dcc.Interval(id="refresh", interval=60 * 1000, n_intervals=0)
])


# ======================================================
# Helper Functions
# ======================================================

def fetch_predictions(model_type: str, horizon: int):
    """Fetch latest real-time forecast."""
    try:
        response = requests.get(
            API_URL,
            params={"model_type": model_type, "horizon": horizon},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def build_forecast_figure(pred, target_name, title, y_label):
    """Generate graph for one target component."""
    fig = go.Figure()

    if "predictions" not in pred:
        fig.update_layout(title="No data available")
        return fig

    horizon = list(pred["predictions"].keys())[0]
    pdata = pred["predictions"][horizon]

    if target_name not in pdata:
        fig.update_layout(title="Target not found")
        return fig

    value = pdata[target_name]

    future_time = datetime.utcnow() + timedelta(hours=int(horizon))

    # Forecast point
    fig.add_trace(go.Scatter(
        x=[future_time],
        y=[value],
        mode="markers+text",
        text=[f"{value:.2f}"],
        textposition="top center",
        marker=dict(size=14, color="#FF5733"),
        name="Forecast"
    ))

    # Confidence interval
    if "confidence_intervals" in pdata:
        ci = pdata["confidence_intervals"]
        if "0.8" in ci:
            fig.add_trace(go.Scatter(
                x=[future_time, future_time],
                y=[ci["0.8"]["lower"], ci["0.8"]["upper"]],
                mode="lines",
                line=dict(color="rgba(255,0,0,0.3)", width=10),
                name="80% CI"
            ))

    fig.update_layout(
        title=title,
        xaxis_title="UTC Time",
        yaxis_title=y_label,
        template="plotly_white"
    )

    return fig


def build_alert_panel(pred):
    """Generate alerts panel."""
    if "alerts" not in pred or len(pred["alerts"]) == 0:
        return html.Div("No active alerts", style={"color": "green"})

    blocks = []
    for alert in pred["alerts"]:
        blocks.append(html.Div([
            html.B(f"{alert['level']} ALERT – {alert['target'].upper()}"),
            html.Br(),
            alert["message"],
            html.Br(),
            html.Small(f"Horizon: {alert['horizon']}h")
        ], style={
            "border": "2px solid red",
            "padding": "10px",
            "marginTop": "10px",
            "backgroundColor": "#ffe6e6",
            "borderRadius": "6px"
        }))
    return blocks


# ======================================================
# Dash Callbacks
# ======================================================

@app.callback(
    [
        Output("speed_plot", "figure"),
        Output("bz_plot", "figure"),
        Output("density_plot", "figure"),
        Output("alert_panel", "children")
    ],
    [
        Input("refresh", "n_intervals"),
        Input("model_type", "value"),
        Input("horizon", "value")
    ]
)
def update_dashboard(_, model_type, horizon):

    pred = fetch_predictions(model_type, horizon)

    speed_fig = build_forecast_figure(pred, "speed", "Solar Wind Speed Forecast", "km/s")
    bz_fig = build_forecast_figure(pred, "bz_gsm", "IMF Bz Forecast", "nT")
    density_fig = build_forecast_figure(pred, "density", "Plasma Density Forecast", "particles/cm³")

    alerts = build_alert_panel(pred)

    return speed_fig, bz_fig, density_fig, alerts


# ======================================================
# Run Dashboard
# ======================================================

if __name__ == "__main__":
    print("🚀 Dashboard running at http://localhost:8050")
    app.run(host="0.0.0.0", port=8050, debug=False)
