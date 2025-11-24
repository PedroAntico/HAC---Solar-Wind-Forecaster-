#!/usr/bin/env python3
import os, json
import pandas as pd
from datetime import datetime

from hac_v6.hac_v6_predictor import HACv6Predictor


# ---------------------------------------------------
# Carregar dataset mais recente (omni_prepared.csv)
# ---------------------------------------------------

DATA_FILE = "data_real/omni_prepared.csv"
OUTPUT_FILE = "results/model_report.json"

HORIZONS = [1, 3, 6, 12, 24, 48]


def load_latest_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"❌ Arquivo não encontrado: {DATA_FILE}")

    print(f"📄 Carregando dataset: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)

    # Garantir ordenação por tempo
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

    return df


# ---------------------------------------------------
# Executar previsões
# ---------------------------------------------------

def run_predictions():
    print("📡 Inicializando HACv6Predictor...")
    predictor = HACv6Predictor()

    df = load_latest_data()
    results = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "predictions": {}
    }

    print("🚀 Iniciando previsões...\n")

    for h in HORIZONS:
        print(f"🔮 Previndo horizonte {h}h ...")

        try:
            pred = predictor.predict(df, h)

            results["predictions"][str(h)] = {
                "success": True,
                "values": pred
            }

            print(f"   ✅ Horizonte {h}h OK")

        except Exception as e:
            results["predictions"][str(h)] = {
                "success": False,
                "error": str(e)
            }
            print(f"   ❌ Falhou no horizonte {h}h: {e}")

    return results


# ---------------------------------------------------
# Salvar em JSON
# ---------------------------------------------------

def save_json(report):
    os.makedirs("results", exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(report, f, indent=4)

    print(f"\n📁 Relatório salvo em: {OUTPUT_FILE}")


# ---------------------------------------------------
# Main
# ---------------------------------------------------

if __name__ == "__main__":
    print("=====================================")
    print("   HAC v6 – Relatório de Previsões   ")
    print("=====================================")

    report = run_predictions()
    save_json(report)

    print("🎉 Concluído!")
