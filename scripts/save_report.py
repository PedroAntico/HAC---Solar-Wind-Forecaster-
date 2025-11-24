#!/usr/bin/env python3
"""
save_report.py
Generates and saves a full prediction report using HAC v6 models.

This script:
- Loads HACv6Predictor
- Runs predictions for all horizons
- Saves results to results/model_report.json
"""

import sys
import os
import json
from datetime import datetime

# ---------------------------------------------------------
# Make sure we can import hac_v6_predictor
# ---------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

try:
    from hac_v6_predictor import HACv6Predictor
except ModuleNotFoundError as e:
    print("❌ ERROR: Could not import HACv6Predictor.")
    print("Details:", e)
    print("\nCheck that hac_v6_predictor.py exists in the project root.")
    sys.exit(1)

# ---------------------------------------------------------
# Create output folder
# ---------------------------------------------------------
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(RESULTS_DIR, "model_report.json")

# ---------------------------------------------------------
# Horizons you want to evaluate
# ---------------------------------------------------------
HORIZONS = [1, 3, 6, 12, 24, 48]

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
def main():

    print("🚀 Initializing HACv6Predictor...")
    
    try:
        predictor = HACv6Predictor()
    except Exception as e:
        print("❌ ERROR initializing HACv6Predictor:")
        print(e)
        sys.exit(1)

    print("✅ Predictor loaded!")
    print("🔍 Running predictions...\n")

    results = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "predictions": {}
    }

    for h in HORIZONS:
        print(f"➡️  Predicting horizon: {h}h ...")
        try:
            pred = predictor.predict(model_type="hybrid", horizon=h)
            results["predictions"][str(h)] = pred
        except Exception as e:
            print(f"❌ Failed at horizon {h}h: {e}")
            results["predictions"][str(h)] = {"error": str(e)}

    # ---------------------------------------------------------
    # Save final report
    # ---------------------------------------------------------
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=4)

    print("\n📁 Report saved in:")
    print(f"   {OUTPUT_FILE}")
    print("\n🎉 Done!")


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
