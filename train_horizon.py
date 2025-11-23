#!/usr/bin/env python3
"""
train_horizon.py
Train HAC v6 for a single horizon only.
"""

import argparse
import sys
from hac_v6_config import HACConfig
from hac_v6_features import HACFeatureBuilder
from hac_v6_models import create_model_builder
import tensorflow as tf
import numpy as np
import os


def train_single_horizon(h):
    print("📡 Loading configuration...")
    config = HACConfig("config.yaml")

    print("🧬 Loading dataset builder...")
    features = HACFeatureBuilder(config)

    print("🧠 Initializing model builder...")
    model_builder = create_model_builder(config)

    print("📁 Loading dataset: data_real/omni_prepared.csv")
    datasets = features.build_all()

    if h not in datasets:
        raise ValueError(f"Horizon {h}h not found in dataset.")

    X = datasets[h]["X"]
    y = datasets[h]["y"]

    print(f"📊 Dataset for horizon {h}h: X={X.shape}, y={y.shape}")

    # splits
    n = len(X)
    val_split = config.get("training")["val_split"]
    test_split = config.get("training")["test_split"]

    n_train = int(n * (1 - val_split - test_split))
    n_val = int(n * val_split)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

    print(f"→ Train = {len(X_train)}, Val = {len(X_val)}, Test = {len(X_test)}")

    model_type = config.get("models")["default"][0]

    print(f"🧠 Building model {model_type.upper()} for horizon {h}h...")
    model = model_builder.build_model(
        model_type=model_type,
        input_shape=X_train.shape[1:],
        output_dim=y_train.shape[1],
    )

    callbacks = model_builder.create_advanced_callbacks(model_type, h)

    print("🚀 Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.get("training")["max_epochs"],
        batch_size=config.get("training")["batch_size"],
        verbose=1,
        callbacks=callbacks
    )

    print("🧪 Evaluating...")
    loss, mae, mse, rmse, dir_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"🎯 RMSE={rmse:.3f} MAE={mae:.3f} DIR_ACC={dir_acc:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train HAC v6 model for a single horizon."
    )
    parser.add_argument(
        "--h", "--horizon", dest="horizon", type=int, required=True,
        help="Horizon to train (1,3,6,12,24,48)"
    )

    args = parser.parse_args()

    train_single_horizon(args.horizon)
