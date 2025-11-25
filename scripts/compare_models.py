#!/usr/bin/env python3
"""
compare_models.py - Compara diferentes arquiteturas de modelo
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


class ModelComparator:
    """Comparador de diferentes arquiteturas de modelo"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = HACConfig(config_path)
        self.feature_builder = HACFeatureBuilder(config)
        
    def train_comparison_models(self):
        """Treina diferentes arquiteturas para comparação"""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
        from sklearn.preprocessing import StandardScaler
        import joblib
        
        logger.info("🤖 Treinando modelos para comparação...")
        
        # Carrega dados
        datasets = self.feature_builder.build_all()
        horizon = 24  # Foca em um horizonte para comparação
        
        if horizon not in datasets:
            logger.error(f"Horizonte {horizon} não disponível")
            return None
            
        X = datasets[horizon]["X"]
        y = datasets[horizon]["y"]
        
        if X.size == 0 or y.size == 0:
            logger.error("Dados insuficientes")
            return None
        
        # Preprocessamento
        lookback = X.shape[1]
        n_features = X.shape[2]
        n_targets = y.shape[1]
        
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X.reshape(-1, n_features)).reshape(X.shape)
        
        scaler_Y = StandardScaler()
        y_scaled = scaler_Y.fit_transform(y)
        
        # Arquiteturas para comparar
        architectures = {
            "LSTM": self._create_lstm_model(lookback, n_features, n_targets),
            "GRU": self._create_gru_model(lookback, n_features, n_targets),
            "CNN-LSTM": self._create_cnn_lstm_model(lookback, n_features, n_targets),
            "Dense": self._create_dense_model(lookback, n_features, n_targets)
        }
        
        results = {}
        
        for name, model in architectures.items():
            logger.info(f"🏗️  Treinando {name}...")
            
            # Compila
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            # Treina (epochs reduzidas para comparação rápida)
            history = model.fit(
                X_scaled, y_scaled,
                epochs=20,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            # Avaliação
            train_loss = history.history['loss'][-1]
            val_loss = history.history['val_loss'][-1]
            train_mae = history.history['mae'][-1]
            val_mae = history.history['val_mae'][-1]
            
            results[name] = {
                'model': model,
                'history': history,
                'metrics': {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_mae': train_mae,
                    'val_mae': val_mae
                }
            }
            
            logger.info(f"   ✅ {name}: val_loss={val_loss:.4f}, val_mae={val_mae:.4f}")
        
        return results
    
    def _create_lstm_model(self, lookback, n_features, n_targets):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(lookback, n_features)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(n_targets)
        ])
        return model
    
    def _create_gru_model(self, lookback, n_features, n_targets):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import GRU, Dense, Dropout
        
        model = Sequential([
            GRU(64, return_sequences=True, input_shape=(lookback, n_features)),
            Dropout(0.2),
            GRU(32, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(n_targets)
        ])
        return model
    
    def _create_cnn_lstm_model(self, lookback, n_features, n_targets):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
        
        model = Sequential([
            Conv1D(64, 3, activation='relu', input_shape=(lookback, n_features)),
            MaxPooling1D(2),
            Conv1D(32, 3, activation='relu'),
            MaxPooling1D(2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(n_targets)
        ])
        return model
    
    def _create_dense_model(self, lookback, n_features, n_targets):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Flatten
        
        model = Sequential([
            Flatten(input_shape=(lookback, n_features)),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(n_targets)
        ])
        return model
    
    def plot_comparison(self, results, save_path: str = "results/comparison"):
        """Plota comparação entre modelos"""
        os.makedirs(save_path, exist_ok=True)
        
        if not results:
            return
        
        # Gráfico de loss
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        for name, result in results.items():
            plt.plot(result['history'].history['loss'], label=f'{name} Train')
            plt.plot(result['history'].history['val_loss'], label=f'{name} Val', linestyle='--')
        plt.title('Loss Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        for name, result in results.items():
            plt.plot(result['history'].history['mae'], label=f'{name} Train')
            plt.plot(result['history'].history['val_mae'], label=f'{name} Val', linestyle='--')
        plt.title('MAE Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/training_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Gráfico de barras com métricas finais
        names = list(results.keys())
        val_losses = [results[name]['metrics']['val_loss'] for name in names]
        val_maes = [results[name]['metrics']['val_mae'] for name in names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.bar(names, val_losses, color=['blue', 'green', 'red', 'orange'])
        ax1.set_title('Validation Loss (Lower is Better)')
        ax1.set_ylabel('Loss')
        ax1.tick_params(axis='x', rotation=45)
        
        ax2.bar(names, val_maes, color=['blue', 'green', 'red', 'orange'])
        ax2.set_title('Validation MAE (Lower is Better)')
        ax2.set_ylabel('MAE')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📈 Gráficos de comparação salvos em: {save_path}")
    
    def generate_comparison_report(self, results):
        """Gera relatório de comparação"""
        if not results:
            return
            
        print("\n" + "="*80)
        print("🤖 RELATÓRIO DE COMPARAÇÃO DE ARQUITETURAS HAC v6")
        print("="*80)
        
        print(f"\n{'Modelo':<12} {'Val Loss':<10} {'Val MAE':<10} {'Train Loss':<10} {'Train MAE':<10}")
        print("-" * 60)
        
        for name, result in results.items():
            metrics = result['metrics']
            print(f"{name:<12} {metrics['val_loss']:<10.4f} {metrics['val_mae']:<10.4f} "
                  f"{metrics['train_loss']:<10.4f} {metrics['train_mae']:<10.4f}")
        
        # Encontra melhor modelo
        best_model = min(results.items(), key=lambda x: x[1]['metrics']['val_loss'])
        print(f"\n🎯 MELHOR MODELO: {best_model[0]} (val_loss: {best_model[1]['metrics']['val_loss']:.4f})")
        
        print("\n" + "="*80)


def main():
    """Comparação principal de modelos"""
    logger.info("🚀 Iniciando comparação de modelos...")
    
    comparator = ModelComparator()
    
    # Treina e compara modelos
    results = comparator.train_comparison_models()
    
    if not results:
        logger.error("❌ Falha na comparação")
        return False
    
    # Gera relatório
    comparator.generate_comparison_report(results)
    
    # Plota resultados
    comparator.plot_comparison(results)
    
    logger.info("✅ Comparação de modelos concluída!")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
