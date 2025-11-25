#!/usr/bin/env python3
"""
Retreina modelos garantindo que scalers Y sejam salvos corretamente
"""

import os
import sys
import joblib
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hac_v6_config import HACConfig
from hac_v6_features import HACFeatureBuilder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def train_model_with_proper_scalers():
    """Versão simplificada do treinamento com scalers corretos"""
    
    config = HACConfig("config.yaml")
    feature_builder = HACFeatureBuilder(config)
    
    print("🧪 Gerando datasets...")
    datasets = feature_builder.build_all()
    
    horizons = config.get("horizons")
    lookback = config.get("training")["main_lookback"]
    
    for horizon in horizons:
        if horizon not in datasets:
            continue
            
        print(f"\n--- Treinando H{horizon} ---")
        
        X = datasets[horizon]["X"]
        y = datasets[horizon]["y"]
        
        if X.size == 0 or y.size == 0:
            print(f"⚠️ Dados insuficientes para H{horizon}")
            continue
        
        # Scaler para features (X)
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        # Scaler SEPARADO para targets (y) - CRÍTICO!
        scaler_Y = StandardScaler()
        y_scaled = scaler_Y.fit_transform(y)
        
        # Modelo simples (exemplo)
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(lookback, X.shape[-1])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(y.shape[1])  # Multi-output
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Treino rápido para teste
        history = model.fit(
            X_scaled, y_scaled,
            epochs=10,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        # Salva com timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = "hybrid"
        model_dir = f"models/hac_v6/{model_type}_h{horizon}_{timestamp}"
        os.makedirs(model_dir, exist_ok=True)
        
        # 🎯 SALVA TUDO CORRETAMENTE
        model.save(os.path.join(model_dir, "model.keras"))
        joblib.dump(scaler_X, os.path.join(model_dir, "scaler_X.pkl"))
        joblib.dump(scaler_Y, os.path.join(model_dir, "scaler_Y.pkl"))  # SCALER Y SEPARADO!
        
        # Metadata
        metadata = {
            "horizon": horizon,
            "model_type": model_type,
            "lookback": lookback,
            "targets": config.get("targets")["primary"],
            "features_count": X.shape[-1],
            "created_at": timestamp,
            "training_samples": len(X)
        }
        
        import json
        with open(os.path.join(model_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Modelo H{horizon} salvo em: {model_dir}")
        print(f"   - Scaler X: {scaler_X.mean_.shape[0]} features")
        print(f"   - Scaler Y: {scaler_Y.mean_.shape[0]} targets")

if __name__ == "__main__":
    train_model_with_proper_scalers()
