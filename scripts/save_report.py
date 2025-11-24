#!/usr/bin/env python3
"""
save_report.py
Gera um arquivo JSON com as predições de todos os horizontes do HAC v6.
"""

import os
import json
import pandas as pd
from datetime import datetime
from hac_v6_predictor import HACv6Predictor


# ---------------------------------------------------------
# Caminhos
# ---------------------------------------------------------

RESULT_DIR = "results"
DATA_FILE = "omni_prepared.csv"


# ---------------------------------------------------------
# Carregar dados e predictor
# ---------------------------------------------------------

print("📥 Carregando dados recentes...")
df = pd.read_csv(DATA_FILE)
df["timestamp"] = pd.to_datetime(df["datetime"])
df = df.drop(columns=["datetime"])

print("🤖 Inicializando predictor...")
predictor = HACv6Predictor()

# todos os modelos disponíveis
available_horizons = sorted(list(predictor.models.keys()))
print(f"🔭 Modelos encontrados: {available_horizons}h")


# ---------------------------------------------------------
# Rodar previsões
# ---------------------------------------------------------

report = {
    "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
    "predictions": {},
    "errors": {}
}

print("🚀 Rodando previsões...")

for h in available_horizons:

    print(f" ▸ Predizendo horizonte {h}h...")

    try:
        pred = predictor.predict(df.copy(), horizon=h)
        report["predictions"][str(h)] = pred
        print(f"   ✅ Sucesso: {pred}")

    except Exception as e:
        report["errors"][str(h)] = str(e)
        print(f"   ❌ Falha: {e}")


# ---------------------------------------------------------
# Salvar arquivo
# ---------------------------------------------------------

os.makedirs(RESULT_DIR, exist_ok=True)
out_path = os.path.join(RESULT_DIR, "model_report.json")

with open(out_path, "w") as f:
    json.dump(report, f, indent=4)

print("\n📄 Relatório salvo em:")
print(f"   {out_path}")
print("✨ Concluído!")
