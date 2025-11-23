#!/usr/bin/env python3
"""
train_horizon.py
Train a single forecasting horizon for HAC v6 Solar Wind Forecaster.

Usage examples:
    python3 train_horizon.py --h 1
    python3 train_horizon.py --h 3
    python3 train_horizon.py --h 6
    python3 train_horizon.py --h 12
    python3 train_horizon.py --h 24
    python3 train_horizon.py --h 48
"""

import os
import argparse
import json
import numpy as np
from datetime import datetime
import joblib

import tensorflow as tf

from hac_v6_config import HACConfig
from hac_v6_features import HACFeatureBuilder
from hac_v6_models import create_model_builder


# ============================================================
# Argumentos CLI
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--h", type=int, required=True, help="Horizon to train (1,3,6,12,24,48).")
args = parser.parse_args()

H = args.h


# ============================================================
# Main Training Function
# ============================================================
def train_horizon(horizon: int):

    print("📡 Loading configuration...")
    config = HACConfig("config.yaml")

    print("🔧 Loading feature builder...")
    fb = HACFeatureBuilder(config)

    print("🧠 Initializing model builder...")
    model_builder = create_model_builder(config)

    model_dir = config.get("paths")["model_dir"]
    os.makedirs(model_dir, exist_ok=True)

    print("\n🚀 Building dataset...")
    datasets = fb.build_all()

    if horizon not in datasets:
        raise ValueError(f"Horizon {horizon}h does not exist in config.")

    X = datasets[horizon]["X"]
    y = datasets[horizon]["y"]

    print(f"Dataset shape for {horizon}h: {X.shape}")

    # -----------------------------
    # Train / Val / Test Split
    # -----------------------------
    n = len(X)
    val_split = config.get("training")["val_split"]
    test_split = config.get("training")["test_split"]

    n_train = int(n * (1 - val_split - test_split))
    n_val = int(n * val_split)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # -----------------------------
    # Build Model
    # -----------------------------
    lookback = config.get("training")["main_lookback"]
    feature_count = X.shape[2]
    output_dim = len(config.get("targets")["primary"])

    model_type = config.get("models")["train_single"]  # ← novo campo no config.yaml

    print(f"\n🧠 Training model = {model_type.upper()}  |  Horizon = {horizon}h")

    model = model_builder.build_model(
        model_type=model_type,
        input_shape=(lookback, feature_count),
        output_dim=output_dim
    )

    callbacks = model_builder.create_advanced_callbacks(model_type, horizon)

    # -----------------------------
    # Fit
    # -----------------------------
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.get("training")["max_epochs"],
        batch_size=config.get("training")["batch_size"],
        verbose=1,
        callbacks=callbacks
    )

    # -----------------------------
    # Evaluate
    # -----------------------------
    loss, mae, mse, rmse, dir_acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"\n🎯 Final Test Metrics (h={horizon}):")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE : {mae:.4f}")
    print(f"   DIR : {dir_acc:.3f}")

    # -----------------------------
    # Save Artifacts
    # -----------------------------
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(model_dir, f"{model_type}_h{horizon}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    model.save(os.path.join(out_dir, "model.h5"))

    # Save scalers
    scaler = fb.scalers["main"]
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))

    # Save metadata
    metadata = {
        "horizon": horizon,
        "model_type": model_type,
        "timestamp": stamp,
        "lookback": lookback,
        "feature_count": feature_count,
        "targets": config.get("targets")["primary"],
        "history": {k: list(v) for k, v in history.history.items()}
    }

    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n💾 Saved model artifacts to: {out_dir}")
    print("✅ Training completed successfully!")


# ============================================================
# EXECUTE
# ============================================================
if __name__ == "__main__":
    train_horizon(H)
