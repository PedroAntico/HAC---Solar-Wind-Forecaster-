#!/usr/bin/env python3
"""
predict_latest.py
Real-Time Prediction Pipeline for HAC v6 Solar Wind Forecaster

Este script faz:
1. Carrega o config.yaml
2. Carrega o modelo treinado HAC v6 (todos os horizons)
3. Carrega o scaler do X e Y
4. Carrega o último omni_prepared.csv real
5. Reproduz exatamente o feature engineering usado no treinamento
6. Cria a janela (lookback)
7. Gera previsão para TODOS os horizons disponíveis
8. Gera alertas
9. Salva JSON em results/latest_forecast.json

Pronto para dashboard, API, automação e deploy.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import joblib

from hac_v6_config import HACConfig
from hac_v6_features import HACFeatureBuilder
from tensorflow.keras.models import load_model


# =====================================================================================
# CARREGADOR PROFISSIONAL DE MODELOS HAC v6
# =====================================================================================

class HACv6Realtime:
    """Pipeline completo de previsão real-time para HAC v6"""

    def __init__(self, config_path="config.yaml"):
        print("📡 Loading configuration...")
        self.config = HACConfig(config_path)

        # paths
        self.model_dir = self.config.get("paths")["model_dir"]
        self.data_dir = self.config.get("paths")["data_dir"]
        self.results_dir = self.config.get("paths")["results_dir"]
        os.makedirs(self.results_dir, exist_ok=True)

        print("🧠 Loading models...")
        self.models = {}          # models[model_type][horizon] = model
        self.scalers = {}         # scalers[model_type][horizon] = {"X":, "Y":}
        self.metadata = {}        # metadata[model_type][horizon]

        self._load_all_models()

        if not self.models:
            raise RuntimeError("❌ No trained models found in models/hac_v6")

        # É necessário criar um FeatureBuilder para replicar o pipeline
        self.fb = HACFeatureBuilder(self.config)

        print("✅ Real-time system ready.\n")

    # ---------------------------------------------------------------------------------
    #  MÉTODO: CARREGAR TODOS OS MODELOS NO DIRETÓRIO
    # ---------------------------------------------------------------------------------
    def _load_all_models(self):
        for folder in os.listdir(self.model_dir):
            full = os.path.join(self.model_dir, folder)
            if not os.path.isdir(full):
                continue

            parts = folder.split("_")
            if len(parts) < 3:
                continue

            model_type = parts[0]
            horizon = int(parts[1].replace("h", ""))

            model_path = os.path.join(full, "model.h5")
            scaler_X_path = os.path.join(full, "scaler_X.pkl")
            scaler_Y_path = os.path.join(full, "scaler_Y.pkl")
            metadata_path = os.path.join(full, "metadata.json")

            if not os.path.exists(model_path):
                continue

            try:
                model = load_model(model_path)

                scaler_X = joblib.load(scaler_X_path)
                scaler_Y = joblib.load(scaler_Y_path)

                with open(metadata_path, "r") as f:
                    meta = json.load(f)

                self.models.setdefault(model_type, {})
                self.scalers.setdefault(model_type, {})
                self.metadata.setdefault(model_type, {})

                self.models[model_type][horizon] = model
                self.scalers[model_type][horizon] = {"X": scaler_X, "Y": scaler_Y}
                self.metadata[model_type][horizon] = meta

                print(f"✔ Loaded model: {model_type.upper()} (h={horizon})")

            except Exception as e:
                print(f"❌ Failed loading {folder}: {e}")

    # ---------------------------------------------------------------------------------
    #  PREPARAR FEATURES EXATAMENTE COMO NO TREINO
    # ---------------------------------------------------------------------------------
    def prepare_for_prediction(self):
        """Carrega o omni_prepared REAL, aplica engenharia de features e retorno df pronto"""

        csv_path = os.path.join(self.data_dir, "omni_prepared.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ Missing OMNI prepared: {csv_path}")

        df = pd.read_csv(csv_path)
        df["timestamp"] = pd.to_datetime(df["datetime"])

        # aplica engenharia de features
        df_feat = self.fb._engineer_features(df.copy())

        # remove "timestamp" e "datetime"
        df_feat = df_feat.select_dtypes(include=["number"])

        return df_feat

    # ---------------------------------------------------------------------------------
    #  FAZER PREVISÃO PARA UM MODELO
    # ---------------------------------------------------------------------------------
    def _predict_single(self, df_feat, model_type, horizon):

        lookback = self.metadata[model_type][horizon].get("lookback", 168)
        targets = self.metadata[model_type][horizon]["targets"]

        # última janela
        window = df_feat.tail(lookback)
        if len(window) < lookback:
            raise RuntimeError("❌ Not enough data for lookback window")

        scalerX = self.scalers[model_type][horizon]["X"]
        scalerY = self.scalers[model_type][horizon]["Y"]

        # escala as features corretamente
        X_scaled = scalerX.transform(window.values)
        X_scaled = X_scaled.reshape(1, lookback, window.shape[1])

        model = self.models[model_type][horizon]
        y_scaled = model.predict(X_scaled, verbose=0)
        y = scalerY.inverse_transform(y_scaled)[0]

        return {targets[i]: float(y[i]) for i in range(len(targets))}

    # ---------------------------------------------------------------------------------
    #  GERA ALERTAS
    # ---------------------------------------------------------------------------------
    def generate_alerts(self, pred):
        out = []
        th = self.config.get("alerts")["thresholds"]

        if pred["speed"] > th["speed_high"]:
            out.append("HIGH_SPEED")

        if pred["speed"] < th["speed_low"]:
            out.append("LOW_SPEED")

        if abs(pred["bz_gse"]) > th["bz_extreme"]:
            out.append("EXTREME_BZ")

        if pred["density"] > th["density_high"]:
            out.append("HIGH_DENSITY")

        return out

    # ---------------------------------------------------------------------------------
    #  FAZER PREVISÃO COMPLETA (TODOS HORIZONS)
    # ---------------------------------------------------------------------------------
    def run(self):

        df_feat = self.prepare_for_prediction()

        output = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "forecasts": {}
        }

        for model_type in self.models:
            output["forecasts"][model_type] = {}

            for horizon in sorted(self.models[model_type].keys()):
                pred = self._predict_single(df_feat, model_type, horizon)
                pred["alerts"] = self.generate_alerts(pred)

                output["forecasts"][model_type][str(horizon)] = pred

        # salva JSON final
        out_path = os.path.join(self.results_dir, "latest_forecast.json")
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\n📁 Saved forecast → {out_path}")
        print("✅ Real-time forecast completed.")

        return output


# =====================================================================================
# ENTRY POINT
# =====================================================================================

if __name__ == "__main__":
    system = HACv6Realtime()
    result = system.run()

    print("\n🔮 Forecast Preview:")
    print(json.dumps(result["forecasts"], indent=2))
