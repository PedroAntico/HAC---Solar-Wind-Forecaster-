#!/usr/bin/env python3
"""
hac_v6_features.py
Feature Engineering & Dataset Builder para HAC v6

Fluxo:
- Carrega omni_prepared.csv
- Faz lags + estatísticas móveis
- Escala APENAS X (features)
- Cria janelas temporais para todos os horizontes
"""

import os
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from hac_v6_config import HACConfig


class HACFeatureBuilder:
    """Gera features e datasets supervisionados para HAC v6."""

    def __init__(self, config: HACConfig):
        self.config = config
        self.scalers: Dict[str, StandardScaler] = {}

        data_dir = self.config.get("paths")["data_dir"]
        csv_path = os.path.join(data_dir, "omni_prepared.csv")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing prepared data: {csv_path}")

        print(f"📂 Loading dataset: {csv_path}")
        df = pd.read_csv(csv_path)

        # Garante que datetime existe se vier na tabela
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])

        self.raw_df = df
        self._validate_required_columns()

    # ------------------------------------------------------------
    def _validate_required_columns(self):
        """Garante que as colunas-alvo existem no CSV."""
        targets = self.config.get("targets")["primary"]
        missing = [c for c in targets if c not in self.raw_df.columns]
        if missing:
            raise ValueError(
                f"❌ Missing target columns in omni_prepared.csv: {missing}\n"
                f"Available columns: {list(self.raw_df.columns)}"
            )

    # ------------------------------------------------------------
    def build_all(self) -> Dict[int, Dict[str, np.ndarray]]:
        """Pipeline principal: engineer → scale → windows."""
        print("🔧 Engineering features...")
        df_feat = self._engineer_features(self.raw_df.copy())

        print("📏 Scaling features...")
        df_scaled = self._scale_features(df_feat)

        print("📦 Building temporal windows...")
        datasets = self._make_all_horizon_windows(df_scaled)

        return datasets

    # ------------------------------------------------------------
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria lags e estatísticas móveis para targets + variáveis relevantes.
        """

        # Remove colunas temporais não-numéricas da engenharia
        for col in ["datetime", "timestamp", "year", "doy", "hour"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Base de features: targets primários + secundários (se existirem)
        targets_cfg = self.config.get("targets")
        primary = targets_cfg.get("primary", [])
        secondary = targets_cfg.get("secondary", [])

        base_cols = [c for c in primary + secondary if c in df.columns]
        if not base_cols:
            raise ValueError(
                "❌ Nenhuma coluna das listas primary/secondary foi encontrada no CSV."
            )

        # Lags
        for col in base_cols:
            df[f"{col}_lag1"] = df[col].shift(1)
            df[f"{col}_lag3"] = df[col].shift(3)

        # Rolling stats
        for col in base_cols:
            df[f"{col}_rm6"] = df[col].rolling(6).mean()
            df[f"{col}_rm12"] = df[col].rolling(12).mean()
            df[f"{col}_std12"] = df[col].rolling(12).std()

        df = df.dropna().reset_index(drop=True)
        return df

    # ------------------------------------------------------------
    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Escala apenas colunas numéricas (features e targets juntos)."""

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_cols:
            raise ValueError("❌ Não há colunas numéricas para escalar.")

        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        self.scalers["X"] = scaler

        print(f"✅ Scaled {len(numeric_cols)} numeric columns.")
        return df

    # ------------------------------------------------------------
    def _make_all_horizon_windows(self, df: pd.DataFrame) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Cria janelas temporais para todos os horizontes definidos no config.
        - X: janelas de lookback de TODAS as colunas numéricas
        - y: targets (em escala ORIGINAL, NÃO escalada)
        """

        lookback = self.config.get("training")["main_lookback"]
        horizons = self.config.get("horizons")
        targets = self.config.get("targets")["primary"]

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        data = df[numeric_cols].values

        # y original (não escalado) → recarregar do raw_df
        df_y = self.raw_df.loc[df.index, :]  # alinha índices pós dropna
        y_array = df_y[targets].values

        datasets: Dict[int, Dict[str, np.ndarray]] = {}

        for h in horizons:
            print(f"🪟 Building horizon {h}h...")

            X_list, y_list = [], []

            for i in range(len(data) - lookback - h):
                X_list.append(data[i:i + lookback])
                y_list.append(y_array[i + lookback + h])

            X_arr = np.array(X_list)
            y_arr = np.array(y_list)

            print(f"   Horizon {h} → X: {X_arr.shape}, y: {y_arr.shape}")

            datasets[h] = {
                "X": X_arr,
                "y": y_arr
            }

        return datasets


# ------------------------------------------------------------
# Execução direta
# ------------------------------------------------------------
if __name__ == "__main__":
    print("🚀 HAC v6 FEATURE BUILDER")
    config = HACConfig("config.yaml")
    builder = HACFeatureBuilder(config)
    datasets = builder.build_all()

    out_dir = os.path.join(config.get("paths")["data_dir"], "prepared")
    os.makedirs(out_dir, exist_ok=True)

    for h, data in datasets.items():
        np.save(os.path.join(out_dir, f"X_{h}.npy"), data["X"])
        np.save(os.path.join(out_dir, f"y_{h}.npy"), data["y"])
        print(f"💾 Saved horizon {h} → {out_dir}")

    print("✅ Feature preparation completed!")
