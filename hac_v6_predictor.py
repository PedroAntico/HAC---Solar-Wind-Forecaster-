#!/usr/bin/env python3
"""
hac_v6_predictor.py
Real-Time Predictor for HAC v6 Solar Wind Forecaster

- Loads trained models
- Loads scalers + metadata
- Prepares input windows for forecasting
- Predicts for any horizon (1–48h)
- Generates alert flags
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta

from hac_v6_config import HACConfig
from tensorflow.keras.models import load_model


class HACv6Predictor:
    """Loads trained HAC models and performs multi-horizon forecasting."""

    def __init__(self, config_path="config.yaml"):
        print("📡 Loading configuration...")
        self.config = HACConfig(config_path)

        self.model_dir = self.config.get("paths")["model_dir"]

        print("🧠 Loading available models...")
        self.models = {}       # models[model_type][horizon] = model
        self.scalers = {}      # scalers[model_type][horizon] = {X, Y}
        self.metadata = {}     # metadata[model_type][horizon] = dict

        self._load_all_models()

        print("✅ Predictor ready!")

    # ----------------------------------------------------------------------
    # Model Loader
    # ----------------------------------------------------------------------
    def _load_all_models(self):
        """Scan model directory and load every saved model automatically."""

        if not os.path.isdir(self.model_dir):
            print(f"Model directory not found: {self.model_dir}")
            return

        for folder in os.listdir(self.model_dir):
            path = os.path.join(self.model_dir, folder)
            if not os.path.isdir(path):
                continue

            # Parse folder name: lstm_h24_20240115_1200
            parts = folder.split("_")
            if len(parts) < 3:
                continue

            model_type = parts[0]
            horizon = int(parts[1].replace("h", ""))

            # Initialize dicts
            self.models.setdefault(model_type, {})
            self.scalers.setdefault(model_type, {})
            self.metadata.setdefault(model_type, {})

            # Read files
            model_path = os.path.join(path, "model.h5")
            scaler_X_path = os.path.join(path, "scaler_X.pkl")
            scaler_Y_path = os.path.join(path, "scaler_Y.pkl")
            metadata_path = os.path.join(path, "metadata.json")

            if not os.path.exists(model_path):
                continue

            try:
                print(f"→ Loading model: {model_type.upper()} horizon={horizon}")
                model = load_model(model_path)

                scaler_X = joblib.load(scaler_X_path)
                scaler_Y = joblib.load(scaler_Y_path)

                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

                # Store
                self.models[model_type][horizon] = model
                self.scalers[model_type][horizon] = {"X": scaler_X, "Y": scaler_Y}
                self.metadata[model_type][horizon] = metadata

            except Exception as e:
                print(f"❌ Failed to load model {folder}: {e}")

    # ----------------------------------------------------------------------
    # Data preparation
    # ----------------------------------------------------------------------
    def prepare_features(self, df: pd.DataFrame, horizon: int, model_type: str):
        """Prepare proper input window for prediction."""

        metadata = self.metadata[model_type][horizon]
        targets = metadata["targets"]

        # Features used in training are ALL columns except targets
        feature_cols = [c for c in df.columns if c not in targets]

        lookback = metadata.get("lookback", 168)

        # Most recent window
        window = df[feature_cols].tail(lookback)
        if len(window) < lookback:
            print("⚠ Not enough data for input window")
            return None

        X = window.values.reshape(1, lookback, len(feature_cols))

        # Scale
        scaler_X = self.scalers[model_type][horizon]["X"]
        X_scaled = scaler_X.transform(window.values)

        X_scaled = X_scaled.reshape(1, lookback, len(feature_cols))
        return X_scaled

    # ----------------------------------------------------------------------
    # Prediction
    # ----------------------------------------------------------------------
    def predict(self, df: pd.DataFrame, model_type="lstm", horizon=24):
        """Predict target values for a specific model + horizon."""

        if model_type not in self.models:
            raise ValueError(f"No model type found: {model_type}")

        if horizon not in self.models[model_type]:
            raise ValueError(f"No trained model for horizon {horizon}h")

        X_scaled = self.prepare_features(df, horizon, model_type)
        if X_scaled is None:
            return None

        model = self.models[model_type][horizon]
        scaler_Y = self.scalers[model_type][horizon]["Y"]
        metadata = self.metadata[model_type][horizon]
        targets = metadata["targets"]

        # Prediction
        y_scaled = model.predict(X_scaled, verbose=0)
        y = scaler_Y.inverse_transform(y_scaled)[0]

        # Build dictionary result
        out = {
            targets[i]: float(y[i])
            for i in range(len(targets))
        }

        # Alerts
        out["alerts"] = self.generate_alerts(out)

        return out

    # ----------------------------------------------------------------------
    # Alerts
    # ----------------------------------------------------------------------
    def generate_alerts(self, prediction):
        """Generate alert flags based on thresholds."""
        thresholds = self.config.get("alerts")["thresholds"]
        alerts = []

        if "speed" in prediction:
            if prediction["speed"] > thresholds["speed_high"]:
                alerts.append("HIGH_SPEED")
            if prediction["speed"] < thresholds["speed_low"]:
                alerts.append("LOW_SPEED")

        if "density" in prediction:
            if prediction["density"] > thresholds["density_high"]:
                alerts.append("HIGH_DENSITY")

        if "bz_gse" in prediction:
            if abs(prediction["bz_gse"]) > thresholds["bz_extreme"]:
                alerts.append("EXTREME_BZ")

        return alerts


# ------------------------------------------------------------
# For direct testing:
# ------------------------------------------------------------
if __name__ == "__main__":
    pred = HACv6Predictor()

    print("\nModels loaded:")
    for m in pred.models:
        print(" -", m, list(pred.models[m].keys()))

    # Fake example for testing
    example = pd.DataFrame(
        np.random.random((300, 8)),
        columns=[f"var{i}" for i in range(8)]
    )

    result = pred.predict(example, model_type="lstm", horizon=24)
    print("\nPrediction example:")
    print(result)
