#!/usr/bin/env python3
"""
plot_features.py - Análise visual das features de entrada
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from hac_v6_config import HACConfig
    from hac_v6_features import HACFeatureBuilder
except ImportError as e:
    logger.error(f"Erro de importação: {e}")
    sys.exit(1)


class FeatureAnalyzer:
    """Analisador de features HAC v6"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = HACConfig(config_path)
        self.feature_builder = HACFeatureBuilder(config)
    
    def analyze_features(self):
        """Análise completa das features"""
        logger.info("🧪 Analisando features...")
        
        # Gera datasets
        datasets = self.feature_builder.build_all()
        
        if not datasets:
            logger.error("❌ Não foi possível gerar datasets")
            return None
        
        analysis = {}
        
        for horizon, data in datasets.items():
            if data["X"].size == 0:
                continue
                
            X = data["X"]
            y = data["y"]
            
            # Estatísticas básicas
            analysis[horizon] = {
                "samples": X.shape[0],
                "lookback": X.shape[1],
                "features": X.shape[2],
                "X_stats": {
                    "mean": np.mean(X, axis=(0, 1)),
                    "std": np.std(X, axis=(0, 1)),
                    "min": np.min(X, axis=(0, 1)),
                    "max": np.max(X, axis=(0, 1))
                },
                "y_stats": {
                    "speed_mean": np.mean(y[:, 0]) if y.shape[1] > 0 else 0,
                    "speed_std": np.std(y[:, 0]) if y.shape[1] > 0 else 0,
                    "bz_mean": np.mean(y[:, 1]) if y.shape[1] > 1 else 0,
                    "bz_std": np.std(y[:, 1]) if y.shape[1] > 1 else 0,
                    "density_mean": np.mean(y[:, 2]) if y.shape[1] > 2 else 0,
                    "density_std": np.std(y[:, 2]) if y.shape[1] > 2 else 0
                }
            }
        
        return analysis
    
    def plot_feature_distributions(self, analysis: dict, save_dir: str = "results/features"):
        """Plota distribuições das features"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Para cada horizonte
        for horizon, data in analysis.items():
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Distribuição de Features - Horizonte H{horizon}', fontsize=16)
            
            # Features de entrada (exemplo das primeiras 4)
            feature_names = [f'Feature_{i}' for i in range(min(4, len(data['X_stats']['mean'])))]
            
            # Plot means
            axes[0, 0].bar(feature_names, data['X_stats']['mean'][:4])
            axes[0, 0].set_title('Média das Features')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Plot std
            axes[0, 1].bar(feature_names, data['X_stats']['std'][:4])
            axes[0, 1].set_title('Desvio Padrão das Features')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Plot targets
            targets = ['Speed', 'Bz', 'Density']
            target_means = [data['y_stats']['speed_mean'], data['y_stats']['bz_mean'], data['y_stats']['density_mean']]
            target_stds = [data['y_stats']['speed_std'], data['y_stats']['bz_std'], data['y_stats']['density_std']]
            
            axes[1, 0].bar(targets, target_means)
            axes[1, 0].set_title('Média dos Targets')
            
            axes[1, 1].bar(targets, target_stds)
            axes[1, 1].set_title('Desvio Padrão dos Targets')
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/features_h{horizon}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"📊 Gráficos de features salvos em: {save_dir}")
    
    def plot_feature_correlations(self, save_dir: str = "results/features"):
        """Plota matriz de correlação das features"""
        # Isso requer acesso aos dados brutos
        try:
            # Exemplo - você precisaria adaptar para seus dados
            df = self.feature_builder.df  # Assumindo que o feature builder tem o DataFrame
            
            if df is not None:
                plt.figure(figsize=(12, 10))
                correlation_matrix = df.corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                           fmt='.2f', linewidths=0.5)
                plt.title('Matriz de Correlação - Features HAC v6')
                plt.tight_layout()
                plt.savefig(f'{save_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
                plt.close()
                logger.info("📈 Matriz de correlação salva")
                
        except Exception as e:
            logger.warning(f"Não foi possível gerar matriz de correlação: {e}")
    
    def generate_feature_report(self, analysis: dict):
        """Gera relatório detalhado das features"""
        print("\n" + "="*80)
        print("🔍 RELATÓRIO DE ANÁLISE DE FEATURES HAC v6")
        print("="*80)
        
        for horizon in sorted(analysis.keys()):
            data = analysis[horizon]
            print(f"\n📈 HORIZONTE H{horizon}:")
            print(f"   • Amostras: {data['samples']}")
            print(f"   • Lookback: {data['lookback']}")
            print(f"   • Features: {data['features']}")
            
            print(f"\n   📊 ESTATÍSTICAS DOS TARGETS:")
            print(f"      • Speed:    {data['y_stats']['speed_mean']:6.1f} ± {data['y_stats']['speed_std']:5.1f} km/s")
            print(f"      • Bz:       {data['y_stats']['bz_mean']:6.1f} ± {data['y_stats']['bz_std']:5.1f} nT")
            print(f"      • Density:  {data['y_stats']['density_mean']:6.1f} ± {data['y_stats']['density_std']:5.1f} cm⁻³")
        
        print("\n" + "="*80)


def main():
    """Análise principal de features"""
    logger.info("🚀 Iniciando análise de features...")
    
    analyzer = FeatureAnalyzer()
    
    # Análise
    analysis = analyzer.analyze_features()
    
    if not analysis:
        logger.error("❌ Falha na análise")
        return False
    
    # Relatório
    analyzer.generate_feature_report(analysis)
    
    # Gráficos
    analyzer.plot_feature_distributions(analysis)
    analyzer.plot_feature_correlations()
    
    logger.info("✅ Análise de features concluída!")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
