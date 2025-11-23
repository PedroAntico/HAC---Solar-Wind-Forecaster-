#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hac_v6_config import HACConfig

class HACFeatureBuilder:

    def __init__(self, config: HACConfig):
        self.config = config
        self.scalers = {}

        csv_path = os.path.join(config.get("paths")["data_dir"], "omni_prepared.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing prepared data: {csv_path}")

        print(f"📂 Loading dataset: {csv_path}")
        self.df = pd.read_csv(csv_path)

        if "datetime" in self.df.columns:
            self.df = self.df.drop(columns=["datetime"], errors="ignore")

        # Required numeric cols
        required = ["speed", "bz_gsm", "density"]
        for r in required:
            if r not in self.df.columns:
                raise ValueError(f"Missing required column: {r}")

        # ============================
        # LOW MEMORY MODE — SAFE
        # ============================
        if config.get("performance")["low_memory_mode"]:
            print("⚠ LOW MEMORY MODE (2GB Codespaces)")
            self.df = self.df.tail(2000).reset_index(drop=True)
            config.config_dict["training"]["main_lookback"] = 48
            keep = ["speed", "bz_gsm", "density", "temperature", "pressure", "bt"]
            self.df = self.df[keep]
            print("⚠ Dataset trimmed to last 2000 rows")

    # ------------------------------------------------------------
    def build_all(self):
        df = self._engineer_features(self.df.copy())
        df = self._scale(df)
        return self._make_windows(df)

    # ------------------------------------------------------------
    def _engineer_features(self, df):
        cols = df.columns

        for c in cols:
            df[f"{c}_lag1"] = df[c].shift(1)
            df[f"{c}_rm6"]   = df[c].rolling(6).mean()

        df = df.dropna().reset_index(drop=True)
        return df

    # ------------------------------------------------------------
    def _scale(self, df):
        numeric = df.select_dtypes(include=["number"])
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(numeric), columns=numeric.columns)

        self.scalers["main"] = scaler
        return df_scaled

    # ------------------------------------------------------------
    def _make_windows(self, df):
        lookback = self.config.get("training")["main_lookback"]
        horizons = self.config.get("horizons")

        X_cols = df.columns
        data = df[X_cols].values
        targets = df[["speed", "bz_gsm", "density"]].values

        datasets = {}

        for h in horizons:
            X_list, y_list = [], []

            for i in range(len(data) - lookback - h):
                X_list.append(data[i:i+lookback])
                y_list.append(targets[i+lookback+h])

            datasets[h] = {"X": np.array(X_list), "y": np.array(y_list)}

        return datasets
