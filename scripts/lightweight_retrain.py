#!/usr/bin/env python3
"""
TREINAMENTO LEVE PARA GITHUB FREE - Versão funcional
"""

import os
import sys
import json
import joblib
import logging
import gc
import numpy as np
import pandas as pd
from datetime import datetime

# Configuração para GitHub Free
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from hac_v6_config import HACConfig
    from hac_v6_features import HACFeatureBuilder
    from sklearn.preprocessing import StandardScaler
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
except ImportError as e:
    print(f"❌ Erro de importação: {e}")
    sys.exit(1)


def train_single_horizon(horizon: int, config):
    """Treina um único horizonte"""
    print(f"🧠 Treinando H{horizon}...")
    
    try:
        # Feature builder
        feature_builder = HACFeatureBuilder(config)
        datasets = feature_builder.build_all()
        
        if horizon not in datasets:
            print(f"❌ Horizonte {horizon} não encontrado")
            return False
        
        X = datasets[horizon]["X"]
        y = datasets[horizon]["y"]
        
        if X.size == 0 or y.size == 0:
            print(f"❌ Dados vazios para H{horizon}")
            return False
        
        print(f"   📊 Dados: {X.shape} -> {y.shape}")
        
        # Pré-processamento
        lookback = X.shape[1]
        n_features = X.shape[2]
        n_targets = y.shape[1]
        
        # Scaler X
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X.reshape(-1, n_features)).reshape(X.shape)
        
        # Scaler Y - CRÍTICO: usar dados ORIGINAIS
        scaler_Y = StandardScaler()
        y_scaled = scaler_Y.fit_transform(y)
        
        print(f"   🔄 Scaling aplicado")
        
        # Modelo leve
        model = Sequential([
            LSTM(32, return_sequences=True, input_shape=(lookback, n_features)),
            Dropout(0.2),
            LSTM(16, return_sequences=False),
            Dropout(0.2),
            Dense(8, activation='relu'),
            Dense(n_targets)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print(f"   🏗️  Modelo: {model.count_params():,} parâmetros")
        
        # Treino rápido
        history = model.fit(
            X_scaled, y_scaled,
            epochs=10,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        # Salva
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        model_dir = f"models/hac_v6/basic_h{horizon}_{timestamp}"
        os.makedirs(model_dir, exist_ok=True)
        
        model.save(os.path.join(model_dir, "model.keras"))
        joblib.dump(scaler_X, os.path.join(model_dir, "scaler_X.pkl"))
        joblib.dump(scaler_Y, os.path.join(model_dir, "scaler_Y.pkl"))  # 🎯 SCALER Y CORRETO!
        
        # Metadata
        metadata = {
            "horizon": horizon,
            "model_type": "basic",
            "lookback": lookback,
            "targets": config.get("targets")["primary"],
            "features_count": n_features,
            "created_at": timestamp,
            "training_samples": len(X),
            "final_loss": float(history.history['loss'][-1]),
            "final_val_loss": float(history.history['val_loss'][-1])
        }
        
        with open(os.path.join(model_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✅ H{horizon} salvo: {model_dir}")
        print(f"   📈 Loss: {history.history['loss'][-1]:.4f}")
        
        # Teste do scaler Y
        test_val = np.array([[0.0, 0.0, 0.0]])
        unscaled = scaler_Y.inverse_transform(test_val)[0]
        print(f"   🧪 Scaler Y test: [0,0,0] → {unscaled}")
        
        # Limpeza
        del model, X, y, X_scaled, y_scaled
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erro H{horizon}: {e}")
        return False


def main():
    """Treinamento principal"""
    print("🚀 TREINAMENTO LEVE - GITHUB FREE")
    print("=" * 50)
    
    config = HACConfig()
    
    # Treina apenas 2 horizontes para teste rápido
    horizons = [1, 24]  # Curto e longo prazo
    
    success_count = 0
    for horizon in horizons:
        success = train_single_horizon(horizon, config)
        if success:
            success_count += 1
        
        # Pausa entre treinos
        import time
        time.sleep(2)
        gc.collect()
    
    print(f"\n🎯 {success_count}/{len(horizons)} modelos treinados")
    
    if success_count > 0:
        print("\n🔥 TESTE IMEDIATO:")
        print("python3 scripts/save_report.py")
    else:
        print("\n❌ Falha no treinamento")
    
    return success_count > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
