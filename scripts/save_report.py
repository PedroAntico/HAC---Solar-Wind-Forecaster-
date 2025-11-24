#!/usr/bin/env python3
"""
save_report.py — HAC v6
Gera relatório de previsões usando modelos HAC v6 treinados.
"""

# ------------------------------------------------------------
# Corrige PATH para permitir imports corretos
# ------------------------------------------------------------
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
# BUILD FEATURES (MESMO PIPELINE DO TREINO)
# ============================================================
print("🧪 Gerando features HACv6...")

feature_builder = HACFeatureBuilder(config)

# Este método EXISTE no seu projeto, 100% confirmado pelo treinamento
df_feat = feature_builder.prepare_single(df_raw)

print(f"✔ {df_feat.shape[1]} features geradas")
print(f"✔ {df_feat.shape[0]} linhas disponíveis\n")

lookback = config.get("training")["main_lookback"]
df_last = df_feat.tail(lookback)

# ============================================================
# LOAD PREDICTOR
# ============================================================
print("🧠 Inicializando HACv6Predictor...")
predictor = HACv6Predictor("config.yaml")
print("✔ Predictor pronto!\n")

# ============================================================
# RUN PREDICTIONS
# ============================================================
horizons = config.get("horizons")
predictions = {}

print("🚀 Iniciando previsões...\n")

for h in horizons:
    print(f"🔮 Previndo horizonte {h}h...")

    try:
        pred = predictor.predict(df_last, h)
        predictions[str(h)] = {
            "ok": True,
            "values": pred
        }
        print(f"   ✔ {pred}\n")

    except Exception as e:
        predictions[str(h)] = {
            "ok": False,
            "error": str(e)
        }
        print(f"   ❌ Falhou: {e}\n")

# ============================================================
# ALERTS
# ============================================================
alert_rules = config.get("alerts")["thresholds"]

def compute_alerts(pred):
    alerts = []

    if pred.get("speed", 0) > alert_rules["speed_high"]:
        alerts.append("HIGH_SPEED")

    if pred.get("speed", 0) < alert_rules["speed_low"]:
        alerts.append("LOW_SPEED")

    if pred.get("density", 0) > alert_rules["density_high"]:
        alerts.append("HIGH_DENSITY")

    if abs(pred.get("bz_gsm", 0)) > alert_rules["bz_extreme"]:
        alerts.append("EXTREME_BZ")

    return alerts

for h in predictions:
    item = predictions[h]
    if item["ok"]:
        item["alerts"] = compute_alerts(item["values"])
    else:
        item["alerts"] = []

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

print("📦 Relatório salvo em:")
print(f"   → {OUT_PATH}")
print("🎉 Concluído!")
