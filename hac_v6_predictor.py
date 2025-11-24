#!/usr/bin/env python3
"""
hac_v6_predictor.py
Real-Time Predictor for HAC v6

- Carrega o ÚLTIMO modelo treinado por horizonte
- Carrega scaler_X
- Prepara janela de entrada a partir de um DataFrame
- Faz previsão (y em unidades físicas, sem inversão de escala)
"""

import os
import json
from typing import Dict, Any

import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

from hac_v6_config import HACConfig


class HACv6Predictor:

    def __init__(self, config_path="config.yaml"):
        print("📡 Loading configuration...")
        self.config = HACConfig(config_path)
        self.paths = self.config.get("paths")
        self.model_dir = self.paths["model_dir"]

        self.models: Dict[int, Any] = {}   # horizon → model
        self.scalers: Dict[int, Any] = {}  # horizon → scaler_X
        self.meta: Dict[int, Dict] = {}    # horizon → metadata

        print("🧠 Scanning model directory...")
        self._load_latest_models()
        print("✅ Predictor ready!")

    # ------------------------------------------------------------
    def _load_latest_models(self):
        """Para cada horizonte, pega o diretório mais recente."""

        if not os.path.isdir(self.model_dir):
            print(f"⚠ Model directory not found: {self.model_dir}")
            return

        # Map horizon → list of (folder_name, full_path)
        horizon_folders: Dict[int, list] = {}

        for folder in os.listdir(self.model_dir):
            path = os.path.join(self.model_dir, folder)
            if not os.path.isdir(path):
                continue

            parts = folder.split("_")
            if len(parts) < 3:
                # Ex: hybrid_h24_20250101...
                continue

            model_type = parts[0]
            h_part = parts[1]  # ex: "h24"
            if not h_part.startswith("h"):
                continue

            try:
                horizon = int(h_part[1:])
            except ValueError:
                continue

            horizon_folders.setdefault(horizon, []).append((folder, path))

        # Para cada horizonte, escolhe o mais recente (ordem alfabética do timestamp já funciona)
        for horizon, folders in horizon_folders.items():
            # Ordena por folder name decrescente (o timestamp no final garante ordem temporal)
            folders_sorted = sorted(folders, key=lambda x: x[0], reverse=True)
            latest_name, latest_path = folders_sorted[0]

            model_path = os.path.join(latest_path, "model.keras")
            scaler_X_path = os.path.join(latest_path, "scaler_X.pkl")
            meta_path = os.path.join(latest_path, "metadata.json")

            if not os.path.exists(model_path):
                continue

            print(f"→ Loading H{horizon} from {latest_name} ...")
            model = load_model(model_path)
            scaler_X = joblib.load(scaler_X_path)

            with open(meta_path, "r") as f:
                meta = json.load(f)

            self.models[horizon] = model
            self.scalers[horizon] = scaler_X
            self.meta[horizon] = meta

    # ------------------------------------------------------------
    def prepare_input(self, df: pd.DataFrame, horizon: int) -> np.ndarray:
        """Prepara X (1, lookback, n_features) para o modelo."""

        if horizon not in self.meta:
            raise ValueError(f"No metadata for horizon {horizon}h. Train models first.")

        meta = self.meta[horizon]
        lookback = meta["lookback"]

        # Pega apenas colunas numéricas
        numeric_df = df.select_dtypes(include=["number"])
        if numeric_df.shape[0] < lookback:
            raise ValueError(
                f"Not enough rows ({numeric_df.shape[0]}) for lookback {lookback}"
            )

        window = numeric_df.tail(lookback)
        scaler_X = self.scalers[horizon]

        X_scaled = scaler_X.transform(window.values)
        X_scaled = X_scaled.reshape(1, lookback, X_scaled.shape[1])

        return X_scaled

    # ------------------------------------------------------------
    def predict(self, df: pd.DataFrame, horizon: int = 24) -> Dict[str, float]:
        """Retorna um dicionário {target: valor_previsto} para um horizonte."""

        if horizon not in self.models:
            raise ValueError(f"No model found for horizon {horizon}h")

        model = self.models[horizon]
        meta = self.meta[horizon]
        targets = meta["targets"]

        X_input = self.prepare_input(df, horizon)
        y_pred = model.predict(X_input, verbose=0)[0]  # shape (3,)

        return {
            targets[i]: float(y_pred[i])
            for i in range(len(targets))
        }


# ------------------------------------------------------------
if __name__ == "__main__":
    pred = HACv6Predictor()

    print("\nModels loaded per horizon:")
    for h in sorted(pred.models.keys()):
        print(f" - H{h}h")

    # Exemplo de uso com dados fake
    example = pd.DataFrame(
        np.random.random((300, 10)),
        columns=[f"var{i}" for i in range(10)]
    )

    res = pred.predict(example, horizon=24)
    print("\nExample prediction (fake input):")
    print(res)
