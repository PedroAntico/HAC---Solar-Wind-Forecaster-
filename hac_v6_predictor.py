#!/usr/bin/env python3
import os, json
import numpy as np
import pandas as pd
import joblib

from hac_v6_config import HACConfig
from tensorflow.keras.models import load_model


class HACv6Predictor:

    def __init__(self, config_path="config.yaml"):
        self.config = HACConfig(config_path)
        self.model_dir = self.config.get("paths")["model_dir"]

        print("📡 Loading models...")
        self.models = {}
        self.scalers = {}
        self.meta = {}

        self._load_all()

        print("✅ Predictor ready!")

    # ---------------------------------------------
    def _load_all(self):

        for folder in os.listdir(self.model_dir):
            path = os.path.join(self.model_dir, folder)
            if not os.path.isdir(path):
                continue

            parts = folder.split("_")
            if len(parts) < 2:
                continue

            model_type = parts[0]
            horizon = int(parts[1].replace("h", ""))

            model_path = os.path.join(path, "model.keras")
            scaler_path = os.path.join(path, "scaler.pkl")
            meta_path = os.path.join(path, "metadata.json")

            if not os.path.exists(model_path):
                continue

            model = load_model(model_path)
            scaler = joblib.load(scaler_path)

            with open(meta_path) as f:
                meta = json.load(f)

            self.models[horizon] = model
            self.scalers[horizon] = scaler
            self.meta[horizon] = meta

    # ---------------------------------------------
    def predict(self, df, horizon):

        if horizon not in self.models:
            raise ValueError(f"No model for horizon {horizon}h")

        model = self.models[horizon]
        scaler = self.scalers[horizon]
        meta = self.meta[horizon]

        lookback = meta["lookback"]

        df = df.tail(lookback)
        X = df.values
        X_scaled = scaler.transform(X)
        X_scaled = X_scaled.reshape(1, lookback, X_scaled.shape[1])

        y_scaled = model.predict(X_scaled, verbose=0)
        y = y_scaled[0]

        return {
            meta["targets"][i]: float(y[i])
            for i in range(len(y))
        }
