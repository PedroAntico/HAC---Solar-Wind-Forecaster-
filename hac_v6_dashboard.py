#!/usr/bin/env python3
"""
hac_v6_dashboard.py - Dashboard Dash otimizado
"""

import os
import logging
from datetime import datetime, timedelta

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import pandas as pd

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import requests
except ImportError:
    logger.error("Requests não disponível")
    exit(1)


# Configuração
API_URL = os.environ.get("HAC_API_URL", "http://localhost:5000")
REFRESH_INTERVAL = 5 * 60 * 1000  # 5 minutos


def safe_fetch_forecast(model_type: str, horizon: int):
    """Fetch seguro da API"""
    try:
        url = f"{API_URL}/api/v1/forecast"
        params = {"model_type": model_type, "horizon": horizon}
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}"}
            
    except requests.exceptions.RequestException as e:
        return {"error": f"Erro de conexão: {e}"}
    except Exception as e:
        return {"error": f"Erro inesperado: {e}"}


def build_simple_figure(value: float, title: str, unit: str, horizon: int):
    """Figura otimizada para performance"""
    now = datetime.utcnow()
    forecast_time = now + timedelta(hours=horizon)
    
    fig = go.Figure()
    
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=value,
        number={'suffix': f" {unit}", 'font': {'size': 24}},
        delta={'reference': 0, 'relative': False},
        title={'text': title, 'font': {'size': 16}},
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig


def build_alert_component(alerts):
    """Componente de alertas otimizado"""
    if not alerts:
        return html.Div(
            "✅ Sem alertas ativos",
            style={
                "color": "green",
                "padding": "10px",
                "border": "1px solid green",
                "borderRadius": "5px",
                "backgroundColor": "#f0fff0"
            }
        )
    
    alert_items = []
    for alert in alerts:
        alert_items.append(
            html.Div(
                f"🚨 {alert}",
                style={
                    "color": "red",
                    "padding": "8px",
                    "margin": "5px 0",
                    "border": "2px solid red",
                    "borderRadius": "5px",
                    "backgroundColor": "#ffe6e6",
                    "fontWeight": "bold"
                }
            )
        )
    
    return html.Div(alert_items)


# App Dash
app = dash.Dash(
    __name__,
    title="HAC v6 - Solar Wind Forecast",
    update_title="Carregando..."
)

app.layout = html.Div([
    html.Div([
        html.H1(
            "🌪️ HAC v6 - Solar Wind Forecast",
            style={
                "textAlign": "center",
                "color": "#2c3e50",
                "marginBottom": "10px"
            }
        ),
        html.Div(
            "AI-Powered Multi-Horizon Space Weather Forecasting",
            style={
                "textAlign": "center",
                "color": "#7f8c8d",
                "marginBottom": "20px",
                "fontSize": "16px"
            }
        ),
    ], style={"backgroundColor": "#f8f9fa", "padding": "20px", "borderRadius": "10px"}),
    
    html.Hr(),
    
    # Controles
    html.Div([
        html.Div([
            html.Label("Horizonte (horas)", style={"fontWeight": "bold"}),
            dcc.Dropdown(
                id="horizon-dropdown",
                options=[{"label": f"{h}h", "value": h} for h in [1, 3, 6, 12, 24, 48]],
                value=24,
                clearable=False,
                style={"marginBottom": "10px"}
            ),
        ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top"}),
        
        html.Div([
            html.Label("Tipo de Modelo", style={"fontWeight": "bold"}),
            dcc.Dropdown(
                id="model-type-dropdown",
                options=[{"label": "Hybrid", "value": "hybrid"}],
                value="hybrid",
                clearable=False,
                disabled=True,
                style={"marginBottom": "10px"}
            ),
        ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top", "marginLeft": "4%"}),
    ]),
    
    html.Br(),
    
    # Métricas
    html.Div([
        html.Div([
            dcc.Graph(id="speed-metric", config={'displayModeBar': False})
        ], style={"width": "32%", "display": "inline-block"}),
        
        html.Div([
            dcc.Graph(id="bz-metric", config={'displayModeBar': False})
        ], style={"width": "32%", "display": "inline-block"}),
        
        html.Div([
            dcc.Graph(id="density-metric", config={'displayModeBar': False})
        ], style={"width": "32%", "display": "inline-block"}),
    ]),
    
    # Alertas
    html.Div([
        html.H3("🚨 Alertas Ativos", style={"marginBottom": "10px"}),
        html.Div(id="alert-panel"),
    ], style={"marginTop": "20px", "padding": "15px", "border": "2px solid #e74c3c", "borderRadius": "5px"}),
    
    # Informações
    html.Div([
        html.Div(id="last-update"),
        html.Div(id="api-status")
    ], style={"marginTop": "20px", "fontSize": "12px", "color": "#7f8c8d", "textAlign": "center"}),
    
    # Intervalo de atualização
    dcc.Interval(
        id="refresh-interval",
        interval=REFRESH_INTERVAL,
        n_intervals=0
    ),
    
    # Store para dados
    dcc.Store(id="forecast-data"),
    
], style={
    "fontFamily": "Arial, sans-serif",
    "margin": "20px",
    "maxWidth": "1200px",
    "marginLeft": "auto",
    "marginRight": "auto"
})


@app.callback(
    [Output("forecast-data", "data"),
     Output("last-update", "children"),
     Output("api-status", "children")],
    [Input("refresh-interval", "n_intervals"),
     Input("horizon-dropdown", "value"),
     Input("model-type-dropdown", "value")]
)
def update_forecast_data(n_intervals, horizon, model_type):
    """Atualiza dados da previsão"""
    logger.info(f"Atualizando previsão - Horizonte: {horizon}h")
    
    data = safe_fetch_forecast(model_type, horizon)
    
    last_update = f"Última atualização: {datetime.utcnow().strftime('%H:%M:%S UTC')}"
    
    if "error" in data:
        status = f"❌ Erro: {data['error']}"
    else:
        status = "✅ Conectado à API HAC v6"
    
    return data, last_update, status


@app.callback(
    [Output("speed-metric", "figure"),
     Output("bz-metric", "figure"),
     Output("density-metric", "figure"),
     Output("alert-panel", "children")],
    [Input("forecast-data", "data"),
     Input("horizon-dropdown", "value")]
)
def update_metrics(forecast_data, horizon):
    """Atualiza métricas e alertas"""
    if not forecast_data or "error" in forecast_data:
        # Figuras de erro
        error_fig = go.Figure()
        error_fig.add_annotation(text="Dados indisponíveis", showarrow=False)
        error_fig.update_layout(height=200)
        
        alert_panel = html.Div("❌ Não foi possível carregar os dados", 
                             style={"color": "red"})
        
        return error_fig, error_fig, error_fig, alert_panel
    
    try:
        predictions = forecast_data.get("predictions", {})
        alerts = forecast_data.get("alerts", [])
        
        speed = predictions.get("speed", 0)
        bz = predictions.get("bz_gsm", predictions.get("bz", 0))
        density = predictions.get("density", 0)
        
        speed_fig = build_simple_figure(speed, "Velocidade", "km/s", horizon)
        bz_fig = build_simple_figure(bz, "Bz GSM", "nT", horizon)
        density_fig = build_simple_figure(density, "Densidade", "cm⁻³", horizon)
        
        alert_panel = build_alert_component(alerts)
        
        return speed_fig, bz_fig, density_fig, alert_panel
        
    except Exception as e:
        logger.error(f"Erro ao atualizar métricas: {e}")
        
        error_fig = go.Figure()
        error_fig.add_annotation(text="Erro nos dados", showarrow=False)
        error_fig.update_layout(height=200)
        
        return error_fig, error_fig, error_fig, html.Div(f"Erro: {e}")


if __name__ == "__main__":
    port = int(os.environ.get("HAC_DASH_PORT", "8050"))
    debug = os.environ.get("HAC_DEBUG", "false").lower() == "true"
    
    logger.info(f"🚀 Iniciando Dashboard na porta {port}")
    
    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug,
        dev_tools_hot_reload=debug
    )
