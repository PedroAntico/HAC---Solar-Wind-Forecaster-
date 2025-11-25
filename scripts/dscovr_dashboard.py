#!/usr/bin/env python3
"""
dscovr_dashboard.py - Dashboard interativo estilo DSCOVR/NOAA
CORREÇÃO: Config carregado corretamente
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from hac_v6_config import HACConfig
    from hac_v6_predictor import get_predictor
except ImportError as e:
    logger.error(f"Erro de importação: {e}")
    sys.exit(1)


class DSCOVRDashboard:
    """Dashboard estilo DSCOVR/NOAA"""
    
    def __init__(self, config_path: str = "config.yaml"):
        # ✅ CORREÇÃO: Carrega config corretamente
        self.config = HACConfig(config_path)
        self.predictor = get_predictor(config_path)
        
    def load_current_forecast(self):
        """Carrega previsão atual"""
        try:
            with open("results/latest_forecast.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error("Arquivo de previsão não encontrado")
            return None
    
    def create_dscovr_style_plot(self, forecast_data):
        """Cria visualização estilo DSCOVR"""
        if not forecast_data:
            return None
            
        predictions = forecast_data.get("predictions", {})
        
        # Prepara dados
        horizons = []
        speed = []
        bz = []
        density = []
        alerts = []
        
        for h_str, pred in predictions.items():
            if pred.get("ok", False):
                horizons.append(int(h_str))
                values = pred["values"]
                speed.append(values.get("speed", 0))
                bz.append(values.get("bz_gsm", 0))
                density.append(values.get("density", 0))
                alerts.append(len(pred.get("alerts", [])) > 0)
        
        # Ordena por horizonte
        sorted_indices = np.argsort(horizons)
        horizons = np.array(horizons)[sorted_indices]
        speed = np.array(speed)[sorted_indices]
        bz = np.array(bz)[sorted_indices]
        density = np.array(density)[sorted_indices]
        alerts = np.array(alerts)[sorted_indices]
        
        # Cores baseadas em alertas
        colors = ['red' if alert else 'blue' for alert in alerts]
        
        # Cria figura estilo DSCOVR
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                '🌪️ Solar Wind Speed Forecast',
                '🧲 Bz GSM Component Forecast', 
                '⚛️ Plasma Density Forecast'
            ),
            vertical_spacing=0.08
        )
        
        # Plot Speed
        fig.add_trace(
            go.Bar(
                x=horizons,
                y=speed,
                marker_color=colors,
                name='Speed (km/s)',
                hovertemplate='H+%{x}h: %{y:.0f} km/s<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Plot Bz
        fig.add_trace(
            go.Bar(
                x=horizons,
                y=bz,
                marker_color=colors,
                name='Bz GSM (nT)',
                hovertemplate='H+%{x}h: %{y:.1f} nT<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Plot Density
        fig.add_trace(
            go.Bar(
                x=horizons,
                y=density,
                marker_color=colors,
                name='Density (cm⁻³)',
                hovertemplate='H+%{x}h: %{y:.1f} cm⁻³<extra></extra>'
            ),
            row=3, col=1
        )
        
        # Atualiza layout
        fig.update_layout(
            title_text="🚀 HAC v6 - Solar Wind Forecast (DSCOVR Style)",
            title_x=0.5,
            height=800,
            showlegend=False,
            template="plotly_white"
        )
        
        # Atualiza eixos
        fig.update_xaxes(title_text="Forecast Horizon (hours)", row=3, col=1)
        fig.update_yaxes(title_text="Speed (km/s)", row=1, col=1)
        fig.update_yaxes(title_text="Bz GSM (nT)", row=2, col=1)
        fig.update_yaxes(title_text="Density (cm⁻³)", row=3, col=1)
        
        # Adiciona linhas de referência
        fig.add_hline(y=500, line_dash="dash", line_color="orange", row=1, col=1)
        fig.add_hline(y=0, line_dash="solid", line_color="black", row=2, col=1)
        fig.add_hline(y=10, line_dash="dash", line_color="orange", row=3, col=1)
        
        return fig
    
    def create_alert_panel(self, forecast_data):
        """Cria painel de alertas estilo NOAA"""
        if not forecast_data:
            return "No data available"
            
        all_alerts = []
        for h_str, pred in forecast_data.get("predictions", {}).items():
            if pred.get("ok", False):
                for alert in pred.get("alerts", []):
                    all_alerts.append(f"H+{h_str}h: {alert}")
        
        if not all_alerts:
            return "✅ No active space weather alerts"
        
        alert_html = "<div style='border: 2px solid red; padding: 10px; background-color: #ffe6e6; border-radius: 5px;'>"
        alert_html += "<h3 style='color: red; margin: 0;'>🚨 ACTIVE ALERTS</h3>"
        for alert in all_alerts:
            alert_html += f"<p style='margin: 5px 0; font-weight: bold;'>{alert}</p>"
        alert_html += "</div>"
        
        return alert_html
    
    def generate_dashboard(self, output_path: str = "results/dashboard.html"):
        """Gera dashboard HTML completo"""
        forecast_data = self.load_current_forecast()
        
        if not forecast_data:
            logger.error("❌ Não foi possível carregar dados para dashboard")
            return False
        
        # Cria visualizações
        fig = self.create_dscovr_style_plot(forecast_data)
        alert_panel = self.create_alert_panel(forecast_data)
        
        # HTML completo
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>HAC v6 - Solar Wind Forecast</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ text-align: center; background: linear-gradient(90deg, #1e3c72, #2a5298); 
                         color: white; padding: 20px; border-radius: 10px; }}
                .alert-panel {{ margin: 20px 0; }}
                .info {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🌪️ HAC v6 - AI Solar Wind Forecast</h1>
                <p>DSCOVR/NOAA Style Dashboard</p>
                <p>Generated: {forecast_data.get('generated_at', 'Unknown')}</p>
            </div>
            
            <div class="alert-panel">
                {alert_panel}
            </div>
            
            <div class="info">
                <strong>Forecast Details:</strong> Multi-horizon AI prediction using LSTM/Transformer hybrid model
            </div>
            
            <div id="plot"></div>
            
            <script>
                var plotData = {fig.to_json() if fig else '{}'};
                Plotly.newPlot('plot', plotData.data, plotData.layout);
            </script>
            
            <div class="info">
                <p><strong>Color Code:</strong> 🔴 = Active Alerts | 🔵 = Normal Conditions</p>
                <p><strong>Reference Lines:</strong> Orange dashed lines indicate typical thresholds</p>
            </div>
        </body>
        </html>
        """
        
        # Salva HTML
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(html_content)
        
        logger.info(f"📊 Dashboard salvo: {output_path}")
        return True


def main():
    """Gera dashboard principal"""
    logger.info("🚀 Gerando dashboard DSCOVR-style...")
    
    dashboard = DSCOVRDashboard()
    success = dashboard.generate_dashboard()
    
    if success:
        logger.info("✅ Dashboard gerado com sucesso!")
        print(f"\n🌐 Abra o dashboard: results/dashboard.html")
    else:
        logger.error("❌ Falha ao gerar dashboard")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
