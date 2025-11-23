#!/usr/bin/env python3
"""
hac_v6_predictor.py
Real-Time Predictor for HAC v6 Solar Wind Forecaster

- Loads trained models (.h5)
- Loads scalers and metadata
- Handles custom metrics used in training
- Predicts multi-horizon solar wind values
- Generates alerts based on thresholds
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta

from hac_v6_config import HACConfig
from tensorflow.keras.models import load_model
import tensorflow as tf


# ==========================================================
# CUSTOM METRICS (needed to load saved models)
# ==========================================================

def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - y_true)))

def mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_pred - y_true)))

def directional_accuracy(y_true, y_pred):
    """
    Compares direction changes (rise/fall) instead of magnitude.
    """
    true_dir = tf.sign(y_true[:, 1:] - y_true[:, :-1])
    pred_dir = tf.sign(y_pred[:, 1:] - y_pred[:, :-1])
    return tf.reduce_mean(tf.cast(tf.equal(true_dir, pred_dir), tf.float32))


# Register for Keras
custom_objects = {
    "rmse": rmse,
    "mse": mse,
    "mae": mae,
    "directional_accuracy": directional_accuracy
}


# ==========================================================
# PREDICTOR CLASS
# ==========================================================

class HACv6Predictor:
    """Loads trained models and performs solar wind forecasting."""

    def __init__(self, config_path="config.yaml"):
        print("📡 Loading configuration...")
        self.config = HACConfig(config_path)

        self.model_dir = self.config.get("paths")["model_dir"]

        print("🧠 Loading available models...")
        self.models = {}       # models[model_type][horizon]
        self.scalers = {}      # scalers[model_type][horizon]
        self.metadata = {}     # metadata[model_type][horizon]

        self._load_all_models()

        print("✅ Predictor ready!")

    # ------------------------------------------------------
    # Load all saved models
    # ------------------------------------------------------
    def _load_all_models(self):

        if not os.path.isdir(self.model_dir):
            print(f"❌ Model directory not found: {self.model_dir}")
            return

        for folder in os.listdir(self.model_dir):
            path = os.path.join(self.model_dir, folder)
            if not os.path.isdir(path):
                continue

            # Folder pattern: hybrid_h24_20251122_223213
            parts = folder.split("_")
            if len(parts) < 3:
                continue

            model_type = parts[0].lower()
            horizon = int(parts[1].replace("h", ""))

            print(f"→ Loading model: {model_type.upper()} horizon={horizon}")

            model_path = os.path.join(path, "model.h5")
            scaler_X_path = os.path.join(path, "scaler_X.pkl")
            scaler_Y_path = os.path.join(path, "scaler_Y.pkl")
            metadata_path = os.path.join(path, "metadata.json")

            if not os.path.exists(model_path):
                print("⚠ Missing model.h5 in folder:", folder)
                continue

            try:
                model = load_model(model_path, custom_objects=custom_objects)
                scaler_X = joblib.load(scaler_X_path)
                scaler_Y = joblib.load(scaler_Y_path)

                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

                # Store
                self.models.setdefault(model_type, {})
                self.scalers.setdefault(model_type, {})
                self.metadata.setdefault(model_type, {})

                self.models[model_type][horizon] = model
                self.scalers[model_type][horizon] = {"X": scaler_X, "Y": scaler_Y}
                self.metadata[model_type][horizon] = metadata

            except Exception as e:
                print(f"❌ Failed to load model {folder}: {e}")

    # ------------------------------------------------------
    # Prepare input window
    # ------------------------------------------------------
    def prepare_features(self, df, horizon, model_type):
        if model_type not in self.metadata:
            print(f"❌ Unknown model type: {model_type}")
            return None

        metadata = self.metadata[model_type][horizon]
        targets = metadata["targets"]
        lookback = metadata.get("lookback", 168)

        feature_cols = [c for c in df.columns if c not in targets]

        window = df[feature_cols].tail(lookback)

        if len(window) < lookback:
            print("⚠ Not enough data to form input window")
            return None

        X = window.values

        scaler_X = self.scalers[model_type][horizon]["X"]
        X_scaled = scaler_X.transform(X)

        X_scaled = X_scaled.reshape(1, lookback, len(feature_cols))

        return X_scaled

    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------
    def predict(self, df, model_type="hybrid", horizon=24):

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

        y_scaled = model.predict(X_scaled, verbose=0)
        y = scaler_Y.inverse_transform(y_scaled)[0]

        prediction = {targets[i]: float(y[i]) for i in range(len(targets))}

        prediction["alerts"] = self.generate_alerts(prediction)

        return prediction

    # ------------------------------------------------------
    # Alerts
    # ------------------------------------------------------
    def generate_alerts(self, prediction):

        thresholds = self.config.get("alerts")["thresholds"]
        alerts = []

        if prediction["speed"] > thresholds["speed_high"]:
            alerts.append("HIGH_SPEED")
        if prediction["speed"] < thresholds["speed_low"]:
            alerts.append("LOW_SPEED")

        if prediction["density"] > thresholds["density_high"]:
            alerts.append("HIGH_DENSITY")

        if abs(prediction["bz_gse"]) > thresholds["bz_extreme"]:
            alerts.append("EXTREME_BZ")

        return alerts


# ----------------------------------------------------------
# Direct run test
# ----------------------------------------------------------
if __name__ == "__main__":
    pred = HACv6Predictor()

    print("\nModels loaded:")
    for m in pred.models:
        print(f" - {m}: horizons {list(pred.models[m].keys())}")

    # Example dummy data
    example = pd.DataFrame(
        np.random.random((300, 3)),
        columns=["speed", "bz_gse", "density"]
    )

    result = pred.predict(example, model_type="hybrid", horizon=24)
    print("\nPrediction example:")
    print(result)
