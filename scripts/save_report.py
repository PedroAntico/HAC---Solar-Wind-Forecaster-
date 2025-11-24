#!/usr/bin/env python3
"""
save_report.py
Generates and saves a full prediction report using HAC v6 models.

Compatible with the current HACv6Predictor.predict() signature:
    predict(horizon)
"""

import sys
import os
import json
from datetime import datetime


# ---------------------------------------------------------
# Adjust import path
# ---------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

try:
    from hac_v6_predictor import HACv6Predictor
except Exception as e:
    print("❌ ERROR importing HACv6Predictor:", e)
    sys.exit(1)


# ---------------------------------------------------------
# Setup output directory
# ---------------------------------------------------------
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(RESULTS_DIR, "model_report.json")

HORIZONS = [1, 3, 6, 12, 24, 48]


def main():
    print("🚀 Initializing HACv6Predictor...")

    try:
        predictor = HACv6Predictor()
    except Exception as e:
        print("❌ ERROR initializing HACv6Predictor:", e)
        sys.exit(1)

    print("✅ Predictor ready!")
    print("🔍 Running predictions...\n")

    results = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "predictions": {}
    }

    for h in HORIZONS:
        print(f"➡️ Predicting horizon {h}h ...")

        try:
            pred = predictor.predict(horizon=h)   # <-- FIXED HERE
            results["predictions"][str(h)] = pred

        except Exception as e:
            print(f"❌ Failed at horizon {h}h: {e}")
            results["predictions"][str(h)] = {"error": str(e)}

    # ---------------------------------------------------------
    # Save output
    # ---------------------------------------------------------
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=4)

    print("\n📁 Report saved in:")
    print(f"   {OUTPUT_FILE}")
    print("🎉 Done!")


if __name__ == "__main__":
    main()
