#!/usr/bin/env python3
"""
save_report.py
Gera e salva relatórios completos das predições do HAC v6
"""

import os
import json
import pandas as pd
from hac_v6_predictor import HACv6Predictor

RESULT_DIR = "results"

def ensure_results_dir():
    if not os.path.exists(RESULT_DIR):
        print("📁 Creating results/ folder...")
        os.makedirs(RESULT_DIR, exist_ok=True)


def save_json(data, filename):
    path = os.path.join(RESULT_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"💾 Saved: {path}")


def main():
    ensure_results_dir()

    print("📡 Loading predictor...")
    pred = HACv6Predictor()

    print("📘 Loading dataset (omni_prepared.csv)...")
    df = pd.read_csv("data_real/omni_prepared.csv")

    horizons = [1, 3, 6, 12, 24, 48]
    models = ["hybrid", "lstm", "gru", "ensemble"]

    full_report = {}

    for model_type in models:
        print(f"\n🧠 Running predictions for model: {model_type}")
        full_report[model_type] = {}

        for h in horizons:
            print(f"   → Horizon: {h}h")
            result = pred.predict(df, model_type=model_type, horizon=h)
            full_report[model_type][h] = result

            # Save each horizon separately
            save_json(result, f"{model_type}_{h}h.json")

    # Save combined master report
    save_json(full_report, "full_report.json")

    print("\n🏁 All reports saved successfully.")


if __name__ == "__main__":
    main()
