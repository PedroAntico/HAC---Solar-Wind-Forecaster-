#!/usr/bin/env python3
"""
validate_predictions.py - Valida previsões contra dados OMNI reais
CORREÇÃO: Config carregado corretamente
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configuração
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from hac_v6_config import HACConfig
    from hac_v6_features import HACFeatureBuilder
    from hac_v6_predictor import get_predictor
except ImportError as e:
    logger.error(f"Erro de importação: {e}")
    sys.exit(1)


class HACValidator:
    """Validador de previsões HAC v6"""
    
    def __init__(self, config_path: str = "config.yaml"):
        # ✅ CORREÇÃO: Carrega config corretamente
        self.config = HACConfig(config_path)
        self.predictor = get_predictor(config_path)
        self.feature_builder = HACFeatureBuilder(self.config)  # ✅ Agora usa self.config
        
    def load_validation_data(self, days_back: int = 7):
        """Carrega dados históricos para validação"""
        logger.info(f"📊 Carregando {days_back} dias de dados para validação...")
        
        # Aqui você precisaria carregar dados OMNI históricos
        # Por enquanto, vamos usar o feature builder para dados recentes
        datasets = self.feature_builder.build_all()
        
        validation_data = {}
        horizons = self.config.get("horizons")
        
        for horizon in horizons:
            if horizon in datasets:
                X = datasets[horizon]["X"]
                y = datasets[horizon]["y"]  # Targets reais
                
                if len(X) > 0 and len(y) > 0:
                    validation_data[horizon] = {
                        "X": X,
                        "y_real": y,
                        "timestamps": []  # Você precisaria dos timestamps reais
                    }
        
        return validation_data
    
    def validate_horizon(self, horizon: int, validation_data: dict):
        """Valida previsões para um horizonte específico"""
        if horizon not in validation_data:
            return None
            
        data = validation_data[horizon]
        X_data = data["X"]
        y_real = data["y_real"]
        
        predictions = []
        actuals = []
        
        # Para cada ponto de dados, faz previsão e compara com real
        for i in range(len(X_data)):
            try:
                # Previsão
                X_window = X_data[i]
                pred = self.predictor.predict_from_features_array(X_window, horizon)
                
                # Valor real correspondente
                actual = {
                    "speed": y_real[i, 0] if y_real.shape[1] > 0 else 0,
                    "bz_gsm": y_real[i, 1] if y_real.shape[1] > 1 else 0,
                    "density": y_real[i, 2] if y_real.shape[1] > 2 else 0
                }
                
                predictions.append(pred)
                actuals.append(actual)
                
            except Exception as e:
                logger.warning(f"Erro na validação H{horizon} ponto {i}: {e}")
                continue
        
        return {
            "predictions": predictions,
            "actuals": actuals,
            "horizon": horizon
        }
    
    def calculate_metrics(self, validation_results: dict):
        """Calcula métricas de erro"""
        metrics = {}
        
        for horizon, results in validation_results.items():
            if results is None or len(results["predictions"]) == 0:
                continue
                
            preds = results["predictions"]
            actuals = results["actuals"]
            
            # Converte para arrays
            speed_pred = np.array([p["speed"] for p in preds])
            speed_actual = np.array([a["speed"] for a in actuals])
            
            bz_pred = np.array([p["bz_gsm"] for p in preds])
            bz_actual = np.array([a["bz_gsm"] for a in actuals])
            
            density_pred = np.array([p["density"] for p in preds])
            density_actual = np.array([a["density"] for a in actuals])
            
            horizon_metrics = {
                "speed": {
                    "mae": mean_absolute_error(speed_actual, speed_pred),
                    "rmse": np.sqrt(mean_squared_error(speed_actual, speed_pred)),
                    "bias": np.mean(speed_pred - speed_actual)
                },
                "bz_gsm": {
                    "mae": mean_absolute_error(bz_actual, bz_pred),
                    "rmse": np.sqrt(mean_squared_error(bz_actual, bz_pred)),
                    "bias": np.mean(bz_pred - bz_actual)
                },
                "density": {
                    "mae": mean_absolute_error(density_actual, density_pred),
                    "rmse": np.sqrt(mean_squared_error(density_actual, density_pred)),
                    "bias": np.mean(density_pred - density_actual)
                }
            }
            
            metrics[horizon] = horizon_metrics
            
        return metrics
    
    def plot_validation(self, validation_results: dict, metrics: dict, save_path: str = None):
        """Gera gráficos de validação estilo NOAA/SWPC"""
        
        n_horizons = len(validation_results)
        fig, axes = plt.subplots(n_horizons, 3, figsize=(20, 5*n_horizons))
        
        if n_horizons == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (horizon, results) in enumerate(validation_results.items()):
            if results is None:
                continue
                
            preds = results["predictions"]
            actuals = results["actuals"]
            
            # Dados para plotting
            speed_pred = [p["speed"] for p in preds]
            speed_actual = [a["speed"] for a in actuals]
            
            bz_pred = [p["bz_gsm"] for p in preds]
            bz_actual = [a["bz_gsm"] for a in actuals]
            
            density_pred = [p["density"] for p in preds]
            density_actual = [a["density"] for a in actuals]
            
            x_points = range(len(speed_pred))
            
            # Plot Speed
            axes[idx, 0].plot(x_points, speed_actual, 'b-', label='Real', alpha=0.7, linewidth=2)
            axes[idx, 0].plot(x_points, speed_pred, 'r--', label='Predito', alpha=0.8, linewidth=2)
            axes[idx, 0].set_title(f'H{horizon} - Velocidade (km/s)\nMAE: {metrics[horizon]["speed"]["mae"]:.1f}')
            axes[idx, 0].legend()
            axes[idx, 0].grid(True, alpha=0.3)
            
            # Plot Bz
            axes[idx, 1].plot(x_points, bz_actual, 'b-', label='Real', alpha=0.7, linewidth=2)
            axes[idx, 1].plot(x_points, bz_pred, 'r--', label='Predito', alpha=0.8, linewidth=2)
            axes[idx, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
            axes[idx, 1].set_title(f'H{horizon} - Bz GSM (nT)\nMAE: {metrics[horizon]["bz_gsm"]["mae"]:.1f}')
            axes[idx, 1].legend()
            axes[idx, 1].grid(True, alpha=0.3)
            
            # Plot Density
            axes[idx, 2].plot(x_points, density_actual, 'b-', label='Real', alpha=0.7, linewidth=2)
            axes[idx, 2].plot(x_points, density_pred, 'r--', label='Predito', alpha=0.8, linewidth=2)
            axes[idx, 2].set_title(f'H{horizon} - Densidade (cm⁻³)\nMAE: {metrics[horizon]["density"]["mae"]:.1f}')
            axes[idx, 2].legend()
            axes[idx, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📈 Gráfico salvo: {save_path}")
        
        plt.show()
    
    def generate_noaa_style_report(self, metrics: dict):
        """Gera relatório estilo NOAA"""
        print("\n" + "="*80)
        print("🌪️  HAC v6 - RELATÓRIO DE VALIDAÇÃO (Estilo NOAA/SWPC)")
        print("="*80)
        
        for horizon in sorted(metrics.keys()):
            h_metrics = metrics[horizon]
            print(f"\n📊 HORIZONTE +{horizon}h:")
            print("-" * 40)
            
            for param in ["speed", "bz_gsm", "density"]:
                m = h_metrics[param]
                print(f"  {param.upper():<10} | MAE: {m['mae']:6.2f} | RMSE: {m['rmse']:6.2f} | Bias: {m['bias']:6.2f}")
        
        print("\n" + "="*80)
        print("🎯 INTERPRETAÇÃO:")
        print("• MAE  = Erro médio absoluto (menor = melhor)")
        print("• RMSE = Raiz do erro quadrático médio (penaliza outliers)")
        print("• Bias = Tendência sistemática (positivo = superestima)")
        print("="*80)


def main():
    """Validação principal"""
    logger.info("🚀 Iniciando validação HAC v6...")
    
    validator = HACValidator()
    
    # Carrega dados de validação
    validation_data = validator.load_validation_data(days_back=3)
    
    if not validation_data:
        logger.error("❌ Nenhum dado de validação disponível")
        return False
    
    # Valida cada horizonte
    validation_results = {}
    for horizon in validation_data.keys():
        logger.info(f"🔍 Validando horizonte H{horizon}...")
        results = validator.validate_horizon(horizon, validation_data)
        validation_results[horizon] = results
    
    # Calcula métricas
    metrics = validator.calculate_metrics(validation_results)
    
    if not metrics:
        logger.error("❌ Não foi possível calcular métricas")
        return False
    
    # Gera relatório
    validator.generate_noaa_style_report(metrics)
    
    # Plota resultados
    os.makedirs("results/validation", exist_ok=True)
    plot_path = f"results/validation/validation_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
    validator.plot_validation(validation_results, metrics, plot_path)
    
    # Salva métricas
    metrics_path = "results/validation/metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"📊 Métricas salvas: {metrics_path}")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
