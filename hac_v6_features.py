#!/usr/bin/env python3
"""
hac_v6_features.py
Feature Engineering & Dataset Builder for HAC v6 Solar Wind Forecaster
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Dict
from hac_v6_config import HACConfig


class HACFeatureBuilder:
    """Generates features and training datasets for HAC v6"""

    def __init__(self, config: HACConfig):
        self.config = config
        self.scalers = {}

        # Load raw prepared data
        csv_path = os.path.join(
            config.get("paths")["data_dir"],
            "omni_prepared.csv"
        )

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing prepared data: {csv_path}")

        print(f"📂 Loading dataset: {csv_path}")
        self.df = pd.read_csv(csv_path)

        # Convert timestamp
        if "datetime" in self.df.columns:
            self.df["timestamp"] = pd.to_datetime(self.df["datetime"])

        # Drop known non-numeric columns early
        drop_cols = ["datetime", "timestamp", "year", "doy", "hour"]
        for col in drop_cols:
            if col in self.df.columns:
                self.df = self.df.drop(columns=[col])

        # Validate columns exist
        self._validate_columns()

    # ------------------------------------------------------------
    def _validate_columns(self):
        """Ensure required fields exist."""

        required = ["speed", "bz_gsm", "density"]

        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"❌ Missing required OMNI columns: {missing}\n"
                f"Available: {list(self.df.columns)}"
            )

    # ------------------------------------------------------------
    def build_all(self):
        """Run the entire feature engineering and dataset preparation"""

        # ------------------------------------------------------------
        # LOW MEMORY MODE (for Codespaces)
        # ------------------------------------------------------------
        low_mem = self.config.get("performance", {}).get("low_memory_mode", False)

        if low_mem:
            print("⚠️ LOW MEMORY MODE ENABLED (Codespaces 2GB)")
            print("⚠️ Trimming dataset to avoid OOM kill...")

            # Keep only last part of data to reduce RAM
            self.df = self.df.tail(3000).reset_index(drop=True)

            # Reduce lookback
            self.config.config_dict["training"]["main_lookback"] = 48

            print("⚠️ Dataset reduced successfully.")

        # ------------------------------------------------------------
        print("🔧 Creating engineered features...")
        df = self._engineer_features(self.df.copy())

        print("📏 Scaling features...")
        df_scaled = self._scale_features(df)

        print("📦 Building supervised learning windows...")
        datasets = self._make_all_horizon_windows(df_scaled)

        return datasets

    # ------------------------------------------------------------
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate lags, rolling stats."""

        feature_cols = [
            "speed", "bz_gsm", "density",
            "temperature", "pressure", "bt"
        ]

        feature_cols = [c for c in feature_cols if c in df.columns]

        # Lags
        for col in feature_cols:
            df[f"{col}_lag1"] = df[col].shift(1)
            df[f"{col}_lag3"] = df[col].shift(3)

        # Rolling windows
        for col in feature_cols:
            df[f"{col}_rm6"] = df[col].rolling(6).mean()
            df[f"{col}_rm12"] = df[col].rolling(12).mean()
            df[f"{col}_std12"] = df[col].rolling(12).std()

        df = df.dropna().reset_index(drop=True)
        return df

    # ------------------------------------------------------------
    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standard scale numeric columns only."""

        numeric_df = df.select_dtypes(include=["number"])
        non_numeric = [c for c in df.columns if c not in numeric_df.columns]

        if non_numeric:
            print(f"⚠️ Removing non-numeric columns before scaling: {non_numeric}")

        df = numeric_df.copy()

        scaler = StandardScaler()
        df[df.columns] = scaler.fit_transform(df[df.columns])

        self.scalers["main"] = scaler

        return df

    # ------------------------------------------------------------
    def _make_all_horizon_windows(self, df: pd.DataFrame) -> Dict:
        """Create temporal windows."""

        lookback = self.config.get("training")["main_lookback"]
        horizons = self.config.get("horizons")

        feature_cols = list(df.columns)
        data = df[feature_cols].values

        targets = df[["speed", "bz_gsm", "density"]].values

        datasets = {}

        for h in horizons:
            print(f"🪟 Building horizon {h}h...")

            X_list, y_list = [], []

            for i in range(len(data) - lookback - h):
                X_list.append(data[i:i + lookback])
                y_list.append(targets[i + lookback + h])

            datasets[h] = {
                "X": np.array(X_list),
                "y": np.array(y_list),
            }

        return datasets


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
