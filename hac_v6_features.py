#!/usr/bin/env python3
"""
hac_v6_features.py
Feature Engineering & Dataset Builder for HAC v6 Solar Wind Forecaster

Prepares the OMNI real-time data into:
- normalized feature matrices
- temporal windows (lookback)
- target arrays for multi-horizon forecasting
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List
from hac_v6_config import HACConfig


class HACFeatureBuilder:
    """Generates features and training datasets for HAC v6"""

    def __init__(self, config: HACConfig):
        self.config = config

        # Scalers for each feature
        self.scalers = {}

        # Load data
        csv_path = os.path.join(
            config.get("paths")["data_dir"],
            "omni_prepared.csv"
        )

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing prepared data: {csv_path}")

        print(f"📂 Loading dataset: {csv_path}")
        self.df = pd.read_csv(csv_path)

        # Your dataset uses 'datetime'
        self.df["timestamp"] = pd.to_datetime(self.df["datetime"])

        # Validate required columns
        self._validate_columns()

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------

    def _validate_columns(self):
        """Ensure dataset has all required columns"""

        required = ["speed", "bz_gsm", "density"]

        missing = [c for c in required if c not in self.df.columns]

        if missing:
            raise ValueError(
                f"❌ Missing required OMNI columns: {missing}. "
                f"Your file contains: {list(self.df.columns)}"
            )

    # ------------------------------------------------------------
    # Main feature engineering pipeline
    # ------------------------------------------------------------

    def build_all(self):
        """Run the full pipeline"""

        print("🔧 Creating engineered features...")
        df = self._engineer_features(self.df.copy())

        print("📏 Scaling features...")
        df_scaled = self._scale_features(df)

        print("📦 Building supervised windows...")
        datasets = self._make_all_horizon_windows(df_scaled)

        return datasets

    # ------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate lags, rolling statistics"""

        # ✔ corrected for your dataset
        feature_cols = [
            "speed", "bz_gsm", "density",
            "temperature", "pressure", "bt"
        ]

        # Filter only existing columns (safeguard)
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
    # Scaling
    # ------------------------------------------------------------

    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standard scale all numeric features"""

        cols = [c for c in df.columns if c not in ["timestamp"]]

        scaler = StandardScaler()
        df[cols] = scaler.fit_transform(df[cols])

        self.scalers["main"] = scaler
        return df

    # ------------------------------------------------------------
    # Window builder
    # ------------------------------------------------------------

    def _make_all_horizon_windows(self, df: pd.DataFrame) -> Dict:
        """Create supervised datasets for all horizons"""

        lookback = self.config.get("training")["main_lookback"]
        horizons = self.config.get("horizons")

        feature_cols = [c for c in df.columns if c != "timestamp"]

        data = df[feature_cols].values

        # ✔ corrected target columns
        targets = df[["speed", "bz_gsm", "density"]].values

        datasets = {}

        for h in horizons:
            print(f"🪟 Building horizon {h}h...")

            X_list = []
            y_list = []

            for i in range(len(data) - lookback - h):
                X_list.append(data[i:i + lookback])
                y_list.append(targets[i + lookback + h])

            datasets[h] = {
                "X": np.array(X_list),
                "y": np.array(y_list),
            }

        return datasets


# ------------------------------------------------------------
# Script mode
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
