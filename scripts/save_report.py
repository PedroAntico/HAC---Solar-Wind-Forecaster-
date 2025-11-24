#!/usr/bin/env python3
"""
save_report.py — HAC v6
Gera relatório de previsões usando modelos HAC v6 treinados.
"""

import os
import json
import pandas as pd
from datetime import datetime

from hac_v6_config import HACConfig
from hac_v6_features import HACFeatureBuilder
from hac_v6_predictor import HACv6Predictor


# ============================================================
# LOAD CONFIG
# ============================================================
print("📡 Carregando configuração...")
config = HACConfig("config.yaml")

data_dir = config.get("paths")["data_dir"]
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# ============================================================
# LOAD RAW DATA
# ============================================================
DATA_FILE = os.path.join(data_dir, "omni_prepared.csv")
print(f"📄 Lendo dataset bruto → {DATA_FILE}")

df_raw = pd.read_csv(DATA_FILE)

# ============================================================
# BUILD FEATURES (SAME PIPELINE AS TRAINING)
# ============================================================
print("🧪 Gerando features HACv6 (MESMO PIPELINE DO TREINO)...")

feature_builder = HACFeatureBuilder(config)
df_feat = feature_builder.transform_realtime(df_raw)

print(f"✔ Features geradas: {df_feat.shape[1]} colunas")
print(f"✔ Linhas disponíveis: {df_feat.shape[0]}")

# Usar apenas os últimos dados possíveis
df_last = df_feat.tail( feature_builder.lookback_main )

# ============================================================
# LOAD PREDICTOR
# ============================================================
print("\n🧠 Inicializando HACv6Predictor...")
predictor = HACv6Predictor("config.yaml")
print("✔ Predictor pronto!\n")

# ============================================================
# RUN PREDICTIONS
# ============================================================
horizons = config.get("horizons")
predictions = {}

print("🚀 Iniciando previsões...\n")

for h in horizons:
    print(f"🔮 Previndo horizonte {h}h ...")

    try:
        pred = predictor.predict(df_last, h)
        predictions[str(h)] = {
            "ok": True,
            "values": pred
        }
        print(f"   ✔ Sucesso: {pred}")

    except Exception as e:
        predictions[str(h)] = {
            "ok": False,
            "error": str(e)
        }
        print(f"   ❌ Falhou: {e}")

# ============================================================
# ALERTS
# ============================================================
alert_rules = config.get("alerts")["thresholds"]

def compute_alerts(pred):
    alerts = []

    if "speed" in pred and pred["speed"] > alert_rules["speed_high"]:
        alerts.append("HIGH_SPEED")

    if "speed" in pred and pred["speed"] < alert_rules["speed_low"]:
        alerts.append("LOW_SPEED")

    if "density" in pred and pred["density"] > alert_rules["density_high"]:
        alerts.append("HIGH_DENSITY")

    if "bz_gsm" in pred and abs(pred["bz_gsm"]) > alert_rules["bz_extreme"]:
        alerts.append("EXTREME_BZ")

    return alerts

for h in predictions:
    if predictions[h]["ok"]:
        predictions[h]["alerts"] = compute_alerts(predictions[h]["values"])
    else:
        predictions[h]["alerts"] = []

# ============================================================
# SAVE REPORT
# ============================================================
report = {
    "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    "source_data": DATA_FILE,
    "predictions": predictions
}

OUT_PATH = os.path.join(results_dir, "model_report.json")

with open(OUT_PATH, "w") as f:
    json.dump(report, f, indent=2)

print("\n📦 Relatório salvo em:")
print("   →", OUT_PATH)
print("🎉 Concluído!")
