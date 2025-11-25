#!/usr/bin/env python3
"""
Treinamento RÁPIDO para GitHub Free - apenas 1 horizonte para teste
"""

import os
import sys
import json
import joblib
import logging
import gc
from datetime import datetime

# Configuração mínima
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


def train_single_model_quick():
    """Treina um único modelo rapidamente para teste"""
    print("🚀 TREINAMENTO RÁPIDO - 1 MODELO PARA TESTE")
    print("=" * 50)
    
    config = HACConfig()
    feature_builder = HACFeatureBuilder(config)
    
    print("🧪 Gerando dados...")
    datasets = feature_builder.build_all()
    
    # Usa apenas o horizonte 24h (mais comum)
    horizon = 24
    
    if horizon not in datasets:
        print(f"❌ Horizonte {horizon}h não encontrado")
        return False
    
    X = datasets[horizon]["X"]
    y = datasets[horizon]["y"]
    
    if X.size == 0 or y.size == 0:
        print("❌ Dados insuficientes")
        return False
    
    print(f"📊 Dados: X{X.shape}, y{y.shape}")
    
    # Pré-processamento
    lookback = X.shape[1]
    n_features = X.shape[2]
    n_targets = y.shape[1]
    
    # Scalers
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X.reshape(-1, n_features)).reshape(X.shape)
    
    scaler_Y = StandardScaler()
    y_scaled = scaler_Y.fit_transform(y)
    
    print("🔧 Criando modelo...")
    
    # Modelo MUITO leve para GitHub Free
    model = Sequential([
        LSTM(16, return_sequences=True, input_shape=(lookback, n_features)),
        Dropout(0.1),
        LSTM(8, return_sequences=False),
        Dropout(0.1),
        Dense(8, activation='relu'),
        Dense(n_targets)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    print(f"🏗️  Modelo criado: {model.count_params():,} parâmetros")
    
    # Treino SUPER rápido
    print("🎯 Treinando (5 épocas rápidas)...")
    history = model.fit(
        X_scaled, y_scaled,
        epochs=5,
        batch_size=8,  # Batch pequeno para memória
        validation_split=0.2,
        verbose=1
    )
    
    # Salva
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_dir = f"models/hac_v6/quick_h{horizon}_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    
    model.save(os.path.join(model_dir, "model.keras"))
    joblib.dump(scaler_X, os.path.join(model_dir, "scaler_X.pkl"))
    joblib.dump(scaler_Y, os.path.join(model_dir, "scaler_Y.pkl"))
    
    # Metadata
    metadata = {
        "horizon": horizon,
        "model_type": "quick_test",
        "lookback": lookback,
        "targets": config.get("targets")["primary"],
        "features_count": n_features,
        "created_at": timestamp,
        "training_samples": len(X),
        "final_loss": float(history.history['loss'][-1]),
        "final_val_loss": float(history.history['val_loss'][-1]),
        "note": "Modelo rápido para teste no GitHub Free"
    }
    
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Modelo salvo: {model_dir}")
    print(f"📈 Loss final: {history.history['loss'][-1]:.4f}")
    
    # Limpeza
    del model, X, y, X_scaled, y_scaled
    gc.collect()
    
    return True


if __name__ == "__main__":
    success = train_single_model_quick()
    
    if success:
        print(f"\n🎉 MODELO CRIADO! Agora teste:")
        print("python3 hac_v6_predictor_github.py")
        print("python3 scripts/save_report.py")
    else:
        print(f"\n❌ Falha no treinamento")
    
    sys.exit(0 if success else 1)
